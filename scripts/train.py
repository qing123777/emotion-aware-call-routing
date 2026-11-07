"""Train one configuration and record the result.

Every run writes results/<run_id>.json containing the full config, the parameter
counts, the per-epoch history and the final metrics. Nothing goes in the report
without a JSON file behind it.

    python scripts/train.py --config configs/crema_d_baseline.yaml
    python scripts/train.py --config configs/crema_d_baseline.yaml --run-id lora_r16 \
        --override model.lora.r=16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import EMOTIONS, SpeechEmotionDataset, class_weights, read_manifest  # noqa: E402
from src.data.labels import EMOTION_TO_INDEX, INTENSITIES  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import MultiTaskLoss, evaluate, fit  # noqa: E402
from src.utils import set_seed  # noqa: E402

# Predicting any of these while the caller is actually calm or happy is a false alarm.
ALARM_EMOTIONS = ("ANG", "DIS")


def apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply `--override a.b=value` so a sweep does not need a file per run."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        path, raw = item.split("=", 1)
        node = config
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)  # parses ints, floats, bools, null
    return config


def build_loaders(config: Dict[str, Any], batch_size: int, num_workers: int):
    data_cfg = config["data"]
    common = {
        "window_seconds": float(data_cfg.get("window_seconds", 4.0)),
        "sample_rate": int(data_cfg.get("sample_rate", 16000)),
    }
    manifest = data_cfg["manifest"]

    train_ds = SpeechEmotionDataset.from_manifest(
        manifest,
        "train",
        gain_jitter_db=float(data_cfg.get("gain_jitter_db", 0.0)),
        noise_snr_db=data_cfg.get("noise_snr_db"),
        **common,
    )
    val_ds = SpeechEmotionDataset.from_manifest(manifest, "val", **common)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, train_ds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None, help="defaults to run_id in the config")
    parser.add_argument("--override", action="append", default=[], help="e.g. train.lr=5e-5")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoint-dir", default=None, help="defaults to train.checkpoint_dir")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test", action="store_true", help="also score the test split -- ONCE, at the end")
    parser.add_argument("--max-batches", type=int, default=None, help="cap batches per epoch, for quick checks")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = apply_overrides(config, args.override)

    run_id = args.run_id or config.get("run_id", "run")
    seed = int(config.get("seed", 1337))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))

    loaders, train_ds = build_loaders(config, batch_size, args.num_workers)

    model = build_model(config, num_emotions=len(EMOTIONS), num_intensities=len(INTENSITIES)).to(device)
    params = model.parameter_summary()
    print(f"[{run_id}] {params['trainable']:,} trainable of {params['total']:,} "
          f"({params['trainable_percent']}%) on {device}")

    loss_cfg = config.get("loss", {})
    weights = None
    if loss_cfg.get("use_class_weights", True):
        weights = class_weights(train_ds.clips, len(EMOTIONS)).to(device)

    criterion = MultiTaskLoss(
        emotion_weight=float(loss_cfg.get("emotion_weight", 1.0)),
        intensity_weight=float(loss_cfg.get("intensity_weight", 0.0)) if config.get("heads", {}).get("intensity") else 0.0,
        class_weights=weights,
    )

    # Only optimise what is actually trainable -- with LoRA that is a tiny slice.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.01))
    )

    # Mixed precision roughly halves memory and speeds up training, but only on a
    # GPU -- on CPU it is not supported, so it switches itself off.
    use_amp = bool(train_cfg.get("fp16", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"[{run_id}] mixed precision: {'on' if use_amp else 'off'}")

    # Linear warmup then decay. Fine-tuning a pre-trained model with a cold, full
    # learning rate can wreck what it already knows in the first few steps.
    steps_per_epoch = max(1, len(loaders["train"]) // max(1, int(train_cfg.get("grad_accum_steps", 1))))
    total_steps = steps_per_epoch * int(train_cfg.get("epochs", 10))
    scheduler = None
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    if warmup_ratio > 0:
        from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save only the parameters that actually learned. Under LoRA that is well
    # under 1% of the model -- everything else is the public backbone, which
    # reloads from HuggingFace by name. Saving the full state dict would produce
    # a ~400MB file per run and throw away LoRA's main practical advantage.
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}

    def trainable_state(current_model) -> Dict[str, Any]:
        full = current_model.state_dict()
        return {key: value for key, value in full.items() if key in trainable_names}

    # Everything a teammate needs to load these weights and get the same numbers:
    # the preprocessing settings, the label order, and the backbone the adapter attaches to.
    def checkpoint_payload(state_dict, epoch):
        return {
            "epoch": epoch,
            "state_dict": state_dict,
            "trainable_only": True,  # load with strict=False after rebuilding the model
            "config": config,
            "emotion_labels": list(EMOTIONS),
            "intensity_labels": list(INTENSITIES),
            "backbone": config.get("model", {}).get("backbone"),
        }

    def save_checkpoint(result, current_model):
        # Save every epoch: Colab sessions are killed without warning, so a
        # disconnect should cost one epoch rather than the whole run.
        torch.save(
            checkpoint_payload(trainable_state(current_model), result.epoch),
            checkpoint_dir / f"{run_id}_last.pt",
        )

    summary = fit(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=EMOTIONS,
        epochs=int(train_cfg.get("epochs", 10)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 3)),
        on_epoch_end=save_checkpoint,
        max_batches=args.max_batches,
        scheduler=scheduler,
        scaler=scaler,
    )

    # fit() restores the best epoch's weights into the model, but until now that
    # only existed in memory -- the file on disk was whatever the LAST epoch
    # produced, which is usually worse. Save the best explicitly; this is the
    # file that gets handed to teammates and linked from the readme.
    best_path = checkpoint_dir / f"{run_id}_best.pt"
    torch.save(checkpoint_payload(trainable_state(model), summary["best_epoch"]), best_path)
    print(f"[{run_id}] best weights saved to {best_path} "
          f"({best_path.stat().st_size / 1e6:.1f} MB)")

    alarm_indices = [EMOTION_TO_INDEX[e] for e in ALARM_EMOTIONS]
    final = {
        "val": evaluate(
            model, loaders["val"], criterion, device, EMOTIONS,
            benign_class=EMOTION_TO_INDEX["NEU"], alarm_classes=alarm_indices,
            max_batches=args.max_batches,
        )
    }

    if args.test:
        # Deliberately gated behind a flag: the test split is opened once, at the end.
        test_ds = SpeechEmotionDataset.from_manifest(
            config["data"]["manifest"], "test",
            window_seconds=float(config["data"].get("window_seconds", 4.0)),
            sample_rate=int(config["data"].get("sample_rate", 16000)),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        final["test"] = evaluate(
            model, test_loader, criterion, device, EMOTIONS,
            benign_class=EMOTION_TO_INDEX["NEU"], alarm_classes=alarm_indices,
        )

    layer_weights = model.layer_weights()
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "device": str(device),
        "config": config,
        "parameters": params,
        "training": summary,
        "metrics": final,
        "layer_weights": layer_weights.tolist() if layer_weights is not None else None,
        "checkpoint": str(best_path),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[{run_id}] best epoch {summary['best_epoch']}, "
          f"val unweighted average recall {summary['best_val_uar']:.4f}")
    print(f"[{run_id}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
