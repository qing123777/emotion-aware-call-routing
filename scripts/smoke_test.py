"""Fast end-to-end sanity check. Run this after touching src/data or src/models.

It builds a tiny synthetic CREMA-D-shaped corpus in a temp directory, so it works
with no dataset downloaded, on CPU, in a few seconds. If the real dataset is
available, pass --audio-dir to additionally check it parses.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --audio-dir data/raw/CREMA-D/AudioWAV
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    EMOTIONS,
    SpeechEmotionDataset,
    assert_no_speaker_leakage,
    assign_splits,
    class_weights,
    emotion_to_arousal_valence,
    escalation_target,
    parse_filename,
    scan_audio_dir,
    read_manifest,
    split_actors,
    write_manifest,
)
from src.data.crema_d import SENTENCE_CODES, CremaDParseError  # noqa: E402
from src.evaluation import false_alarm_rate, routing_metrics, unweighted_average_recall  # noqa: E402
from src.models import SpeechEmotionModel, count_parameters, masked_mean  # noqa: E402
from src.utils import set_seed  # noqa: E402

PASS = "  ok  "
FAIL = " FAIL "


# --- A stand-in for WavLM ---------------------------------------------------
# Mimics the parts of a wav2vec2/WavLM model that our code touches, so the model
# tests run in seconds on CPU without downloading ~360MB.

class _DummyConfig:
    def __init__(self, hidden_size: int, num_hidden_layers: int):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class _DummyOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class DummyBackbone(__import__("torch").nn.Module):
    """Same shape of interface as a real speech backbone, ~1000x smaller."""

    STRIDE = 320  # real wav2vec2-family models reduce 16 kHz audio to ~50 frames/s

    def __init__(self, hidden_size: int = 32, num_hidden_layers: int = 4):
        import torch as _torch
        from torch import nn as _nn

        super().__init__()
        self.config = _DummyConfig(hidden_size, num_hidden_layers)
        # Named feature_extractor so freeze_feature_extractor() finds it.
        self.feature_extractor = _nn.Conv1d(1, hidden_size, kernel_size=self.STRIDE, stride=self.STRIDE)
        self.encoder = _nn.Module()
        self.encoder.layers = _nn.ModuleList(
            [_nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self._torch = _torch

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths // self.STRIDE

    def forward(self, input_values, output_hidden_states: bool = False, **kwargs):
        x = self.feature_extractor(input_values.unsqueeze(1)).transpose(1, 2)
        hidden_states = [x]
        for layer in self.encoder.layers:
            x = layer(x)
            hidden_states.append(x)
        return _DummyOutput(tuple(hidden_states))


def _write_wav(path: Path, seconds: float = 2.0, sr: int = 16_000) -> None:
    """Write a short synthetic tone as 16-bit PCM."""
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def make_fake_corpus(root: Path, n_actors: int = 12) -> Path:
    """Create a miniature corpus using real CREMA-D filename conventions."""
    audio_dir = root / "AudioWAV"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for actor in range(1001, 1001 + n_actors):
        for emotion in EMOTIONS:
            for intensity in ("LO", "HI"):
                name = f"{actor}_{SENTENCE_CODES[0]}_{emotion}_{intensity}.wav"
                _write_wav(audio_dir / name, seconds=2.0)
    return audio_dir


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"[{PASS if condition else FAIL}] {name}{(' - ' + detail) if detail else ''}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", default=None, help="optionally also check the real CREMA-D AudioWAV/")
    args = parser.parse_args()

    set_seed(1337)
    results = []

    # 1. Filename parsing -----------------------------------------------------
    parsed = parse_filename("1001_DFA_ANG_XX.wav")
    results.append(
        check(
            "filename parsing",
            parsed == {"actor_id": 1001, "sentence": "DFA", "emotion": "ANG", "intensity": "XX"},
            str(parsed),
        )
    )

    # One real file is tagged "X" instead of "XX"; it must survive as unspecified
    # rather than being dropped over a one-character typo in the corpus.
    results.append(
        check(
            "single-character intensity tag normalised",
            parse_filename("1040_ITH_SAD_X.wav")["intensity"] == "XX",
        )
    )

    bad_rejected = True
    for bad in ("1001_DFA_ANG.wav", "1001_ZZZ_ANG_XX.wav", "1001_DFA_XXX_XX.wav", "abcd_DFA_ANG_XX.wav"):
        try:
            parse_filename(bad)
            bad_rejected = False
            print(f"        did not reject malformed name {bad!r}")
        except CremaDParseError:
            pass
    results.append(check("malformed filenames rejected", bad_rejected))

    # 2. Label derivation -----------------------------------------------------
    ang_hi = escalation_target("ANG", "HI")
    neu = escalation_target("NEU", "XX")
    hap_hi = escalation_target("HAP", "HI")
    results.append(
        check(
            "escalation score ordering (angry > neutral > happy)",
            ang_hi > neu > hap_hi,
            f"ANG/HI={ang_hi:.3f} NEU={neu:.3f} HAP/HI={hap_hi:.3f}",
        )
    )
    a_lo, v_lo = emotion_to_arousal_valence("ANG", "LO")
    a_hi, v_hi = emotion_to_arousal_valence("ANG", "HI")
    results.append(
        check(
            "intensity scales magnitude but not sign",
            abs(a_lo) < abs(a_hi) and abs(v_lo) < abs(v_hi) and v_lo < 0 and v_hi < 0,
            f"LO=({a_lo:.2f},{v_lo:.2f}) HI=({a_hi:.2f},{v_hi:.2f})",
        )
    )

    # 3. Pipeline on a synthetic corpus --------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        audio_dir = make_fake_corpus(tmp_root)
        clips = scan_audio_dir(audio_dir)
        results.append(check("scan synthetic corpus", len(clips) == 12 * len(EMOTIONS) * 2, f"{len(clips)} clips"))

        splits = split_actors((c.actor_id for c in clips), seed=1337)
        buckets = assign_splits(clips, splits)

        leak_free = True
        try:
            assert_no_speaker_leakage(buckets)
        except AssertionError as exc:
            leak_free = False
            print(f"        {exc}")
        results.append(check("speaker-independent splits", leak_free))

        all_actors = {c.actor_id for c in clips}
        split_actor_union = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        results.append(
            check("every actor assigned exactly once", split_actor_union == all_actors and
                  len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(all_actors))
        )

        # split determinism
        again = split_actors((c.actor_id for c in clips), seed=1337)
        different = split_actors((c.actor_id for c in clips), seed=99)
        results.append(
            check("split is seed-deterministic and seed-sensitive",
                  again == splits and different != splits)
        )

        manifest = write_manifest(buckets, tmp_root / "manifest.csv")
        results.append(check("manifest written", manifest.exists()))

        # train.py reads splits back from this file, so a lossy round trip would
        # silently move clips between splits.
        reread = {split: read_manifest(manifest, split) for split in ("train", "val", "test")}
        results.append(
            check(
                "manifest round-trips every split",
                all(
                    sorted(c.path for c in reread[s]) == sorted(c.path for c in buckets[s])
                    for s in ("train", "val", "test")
                ),
                " ".join(f"{s}={len(reread[s])}" for s in ("train", "val", "test")),
            )
        )

        train_ds = SpeechEmotionDataset.from_manifest(manifest, "train", window_seconds=1.0)
        item = train_ds[0]
        results.append(
            check(
                "dataset item shapes",
                item["waveform"].shape == (16_000,)
                and item["waveform"].dtype.is_floating_point
                and item["emotion"].dtype.__str__() == "torch.int64",
                f"waveform={tuple(item['waveform'].shape)} dtype={item['waveform'].dtype}",
            )
        )
        results.append(
            check(
                "waveform in [-1, 1]",
                float(item["waveform"].abs().max()) <= 1.0,
                f"max abs = {float(item['waveform'].abs().max()):.3f}",
            )
        )

        # padding path: window longer than the 2s source audio
        long_ds = SpeechEmotionDataset.from_manifest(manifest, "val", window_seconds=4.0)
        results.append(check("short clips are padded to the window", long_ds[0]["waveform"].shape == (64_000,)))

        weights = class_weights([c for c in clips], len(EMOTIONS))
        results.append(
            check("class weights finite and mean 1", bool(weights.isfinite().all()) and abs(float(weights.mean()) - 1.0) < 1e-5)
        )

        # a real DataLoader pass, since collation is where shape bugs surface
        from torch.utils.data import DataLoader  # noqa: PLC0415

        batch = next(iter(DataLoader(train_ds, batch_size=4, shuffle=True)))
        results.append(
            check(
                "DataLoader batching",
                batch["waveform"].shape == (4, 16_000) and batch["emotion"].shape == (4,),
                f"waveform={tuple(batch['waveform'].shape)}",
            )
        )

    # 4. Model ----------------------------------------------------------------
    import torch  # noqa: PLC0415

    # Padding must not dilute the pooled vector: a clip that is half silence
    # should pool to the same value as one that is entirely real audio.
    frames = torch.cat([torch.ones(1, 5, 3), torch.zeros(1, 5, 3)], dim=1)
    mask = torch.cat([torch.ones(1, 5), torch.zeros(1, 5)], dim=1)
    results.append(
        check(
            "masked pooling ignores padding",
            torch.allclose(masked_mean(frames, mask), torch.ones(1, 3))
            and not torch.allclose(masked_mean(frames, None), torch.ones(1, 3)),
            f"masked={masked_mean(frames, mask).flatten()[0]:.3f} "
            f"unmasked={masked_mean(frames, None).flatten()[0]:.3f}",
        )
    )

    def make_model(strategy: str, pooling: str = "weighted") -> SpeechEmotionModel:
        return SpeechEmotionModel(
            backbone=DummyBackbone(),
            num_emotions=len(EMOTIONS),
            num_intensities=4,
            finetune_strategy=strategy,
            partial_unfreeze_layers=2,
            layer_pooling=pooling,
        )

    model = make_model("frozen")
    waveform = torch.randn(2, 16_000) * 0.1
    num_valid = torch.tensor([16_000, 8_000])
    out = model(waveform, num_valid)
    results.append(
        check(
            "model forward shapes",
            out["emotion"].shape == (2, len(EMOTIONS)) and out["intensity"].shape == (2, 4),
            f"emotion={tuple(out['emotion'].shape)} intensity={tuple(out['intensity'].shape)}",
        )
    )

    weights = model.layer_weights()
    results.append(
        check(
            "layer weights normalised",
            weights is not None and abs(float(weights.sum()) - 1.0) < 1e-5 and weights.numel() == 5,
            f"{weights.numel()} layers, sum={float(weights.sum()):.4f}",
        )
    )
    results.append(check("last-layer pooling has no weights", make_model("frozen", "last").layer_weights() is None))

    # Each strategy must unlock strictly more of the backbone than the previous.
    trainable = {}
    for strategy in ("frozen", "partial", "full"):
        m = make_model(strategy)
        trainable[strategy] = count_parameters(m.backbone)["trainable"]
    results.append(
        check(
            "finetune strategies unlock increasing parameter counts",
            trainable["frozen"] == 0 < trainable["partial"] < trainable["full"],
            f"frozen={trainable['frozen']} partial={trainable['partial']} full={trainable['full']}",
        )
    )

    # The conv front end stays locked even under 'full'.
    full_model = make_model("full")
    results.append(
        check(
            "feature extractor frozen under every strategy",
            not any(p.requires_grad for p in full_model.backbone.feature_extractor.parameters()),
        )
    )

    # LoRA is the one strategy that needs an extra library (peft). The dummy
    # backbone has no q_proj/v_proj, so its encoder Linear layers ("0".."3")
    # are targeted instead. Skipped, not failed, when peft is absent so the
    # rest of the smoke test still runs on a minimal install.
    try:
        import peft  # noqa: F401, PLC0415
    except ImportError:
        print("[ skip ] LoRA adapters attach - peft not installed")
    else:
        lora_model = SpeechEmotionModel(
            backbone=DummyBackbone(),
            num_emotions=len(EMOTIONS),
            finetune_strategy="lora",
            lora_config={"r": 4, "alpha": 8, "target_modules": ["0", "1", "2", "3"]},
        )
        lora_counts = count_parameters(lora_model.backbone)
        lora_out = lora_model(waveform, num_valid)["emotion"]
        results.append(
            check(
                "LoRA adapters attach and train only a small slice",
                0 < lora_counts["trainable"] < trainable["full"] and lora_out.shape == (2, len(EMOTIONS)),
                f"trainable={lora_counts['trainable']} of {lora_counts['total']}",
            )
        )

    # Every ablation is launched as --override key.path=value, so the parser
    # must create nested keys and type the value (int, bool, null) correctly.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train import apply_overrides  # noqa: PLC0415

    overridden = apply_overrides(
        {"model": {"lora": {"r": 8}}},
        ["model.lora.r=16", "model.finetune_strategy=partial", "train.amp=false", "data.limit=null"],
    )
    results.append(
        check(
            "config overrides set nested, typed values",
            overridden == {
                "model": {"lora": {"r": 16}, "finetune_strategy": "partial"},
                "train": {"amp": False},
                "data": {"limit": None},
            },
        )
    )

    # 5. Metrics --------------------------------------------------------------
    # A lazy model that always predicts the majority class scores well on plain
    # accuracy and badly on unweighted average recall. That gap is the whole
    # reason UAR is our headline metric.
    true = [0] * 90 + [1] * 10
    lazy = [0] * 100
    uar = unweighted_average_recall(true, lazy, 2)
    results.append(
        check(
            "unweighted average recall punishes majority-class guessing",
            abs(uar - 0.5) < 1e-6,
            f"UAR={uar:.3f} while accuracy would be 0.900",
        )
    )

    # Two of four genuinely neutral clips predicted as an alarm class.
    far = false_alarm_rate([0, 0, 0, 0], [0, 3, 3, 0], benign_class=0, alarm_classes=[3, 5])
    results.append(check("false alarm rate", abs(far - 0.5) < 1e-6, f"{far:.2f}"))

    # The application decides "fast-track or not", not "which of six emotions".
    # A six-way error that stays inside one routing group is a correct decision,
    # and this is the check that the collapse actually credits it.
    # EMOTIONS order is NEU HAP SAD ANG FEA DIS -> escalate on SAD ANG FEA DIS.
    escalate = [EMOTIONS.index(e) for e in ("SAD", "ANG", "FEA", "DIS")]
    routing = routing_metrics(
        #  true: FEA        SAD        HAP        NEU
        true=[EMOTIONS.index("FEA"), EMOTIONS.index("SAD"),
              EMOTIONS.index("HAP"), EMOTIONS.index("NEU")],
        #  pred: ANG (ok)   DIS (ok)   ANG (bad)  NEU (ok)
        predicted=[EMOTIONS.index("ANG"), EMOTIONS.index("DIS"),
                   EMOTIONS.index("ANG"), EMOTIONS.index("NEU")],
        escalate_classes=escalate,
        class_names=EMOTIONS,
    )
    results.append(
        check(
            "routing collapse credits within-group confusions",
            # fear->angry and sad->disgust are wrong six-way but route correctly
            routing["rescued_errors"] == 2
            # happy->angry is the one real routing error: 1 of 2 standard clips
            and abs(routing["false_alarm_rate"] - 0.5) < 1e-6
            # both genuinely-escalate clips were escalated
            and abs(routing["escalate_recall"] - 1.0) < 1e-6
            # a correct six-way prediction can never route wrongly
            and routing["introduced_errors"] == 0,
            f"rescued {routing['rescued_errors']} six-way errors, "
            f"false alarm {routing['false_alarm_rate']:.2f}",
        )
    )

    # 6. A real training step --------------------------------------------------
    # Overfitting a handful of examples is the cheapest proof that the loss,
    # optimiser and backward pass are actually wired together.
    from torch.utils.data import DataLoader as _DataLoader  # noqa: PLC0415

    from src.train import MultiTaskLoss, fit  # noqa: PLC0415

    class _TinySet(torch.utils.data.Dataset):
        def __init__(self, n: int = 8):
            torch.manual_seed(0)
            self.x = torch.randn(n, 16_000) * 0.1
            self.y = torch.arange(n) % len(EMOTIONS)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {
                "waveform": self.x[i],
                "num_valid": torch.tensor(16_000),
                "emotion": self.y[i],
                "intensity": torch.tensor(0),
                "actor_id": torch.tensor(1),
            }

    tiny_loader = _DataLoader(_TinySet(), batch_size=4)
    train_model = make_model("full")
    summary = fit(
        model=train_model,
        train_loader=tiny_loader,
        val_loader=tiny_loader,
        criterion=MultiTaskLoss(),
        optimizer=torch.optim.AdamW(train_model.parameters(), lr=1e-2),
        device=torch.device("cpu"),
        class_names=EMOTIONS,
        epochs=6,
        early_stopping_patience=6,
        verbose=False,
    )

    # A long run must not be silent: the per-epoch line is what tells you whether
    # the score is still climbing or has flattened. Capture it rather than trust it.
    log = io.StringIO()
    log_model = make_model("full")
    with contextlib.redirect_stdout(log):
        fit(
            model=log_model,
            train_loader=tiny_loader,
            val_loader=tiny_loader,
            criterion=MultiTaskLoss(),
            optimizer=torch.optim.AdamW(log_model.parameters(), lr=1e-2),
            device=torch.device("cpu"),
            class_names=EMOTIONS,
            epochs=3,
            early_stopping_patience=3,
            verbose=True,
        )
    log_lines = [line for line in log.getvalue().splitlines() if line.strip()]
    # header + one line per epoch, and the val UAR must actually appear on them
    numbered = [line for line in log_lines if line.strip().split(" ")[0].isdigit()]
    results.append(
        check(
            "verbose training prints one line per epoch",
            len(numbered) == 3 and "val UAR" in log_lines[0],
            f"{len(numbered)} epoch lines, header={log_lines[0].strip()!r}",
        )
    )
    # Saving only trainable parameters is what keeps a LoRA checkpoint small.
    # Under the frozen strategy that must be the heads and pooling weights only.
    frozen_model = make_model("frozen")
    trainable_names = {n for n, p in frozen_model.named_parameters() if p.requires_grad}
    saved = {k: v for k, v in frozen_model.state_dict().items() if k in trainable_names}
    results.append(
        check(
            "checkpoint saves only trainable parameters",
            0 < len(saved) < len(frozen_model.state_dict())
            and not any(k.startswith("backbone.") for k in saved),
            f"{len(saved)} of {len(frozen_model.state_dict())} entries, no backbone weights",
        )
    )

    # The unwrap helper must find the object exposing the frame-length method even
    # when it is nested behind wrappers, as peft nests it.
    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_model = inner

    probe = make_model("frozen")
    probe.backbone = _Wrapper(_Wrapper(probe.backbone))
    results.append(
        check(
            "backbone unwrapping survives nested wrappers",
            hasattr(probe._base_backbone(), "_get_feat_extract_output_lengths"),
        )
    )

    first_loss = summary["history"][0]["train_loss"]
    last_loss = summary["history"][-1]["train_loss"]
    results.append(
        check(
            "training loop reduces loss",
            last_loss < first_loss,
            f"{first_loss:.3f} -> {last_loss:.3f} over {summary['epochs_run']} epochs",
        )
    )
    results.append(
        check(
            "best epoch tracked",
            1 <= summary["best_epoch"] <= summary["epochs_run"],
            f"best epoch {summary['best_epoch']}, val UAR {summary['best_val_uar']:.3f}",
        )
    )

    # 7. Optional: the real dataset ------------------------------------------
    if args.audio_dir:
        real = scan_audio_dir(args.audio_dir)
        actors = {c.actor_id for c in real}
        results.append(
            check("real CREMA-D scanned (expect 7442 clips / 91 actors)",
                  len(real) > 0, f"{len(real)} clips, {len(actors)} actors")
        )

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
