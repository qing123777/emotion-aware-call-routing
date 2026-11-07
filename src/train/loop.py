"""Training and evaluation loop, shared by the script and the Kaggle notebook."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import false_alarm_rate, most_confused_pairs, summarise_classification


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_uar: float
    seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


class MultiTaskLoss(nn.Module):
    """Cross-entropy on each active head, added together with weights.

    Class weights are applied to the emotion head only. CREMA-D has roughly four
    times fewer neutral clips than the other emotions, so without them the model
    quietly learns to under-predict neutral.
    """

    def __init__(
        self,
        emotion_weight: float = 1.0,
        intensity_weight: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.emotion_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.intensity_loss = nn.CrossEntropyLoss()
        self.emotion_weight = emotion_weight
        self.intensity_weight = intensity_weight

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = self.emotion_weight * self.emotion_loss(outputs["emotion"], batch["emotion"])
        if "intensity" in outputs and self.intensity_weight > 0:
            total = total + self.intensity_weight * self.intensity_loss(
                outputs["intensity"], batch["intensity"]
            )
        return total


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _format_duration(seconds: float) -> str:
    """Seconds as a short human-readable string, for the progress log only."""
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any = None,
    grad_accum_steps: int = 1,
    max_batches: int | None = None,
    scaler: Any = None,
) -> float:
    """One pass over the training data. Returns the mean loss.

    ``grad_accum_steps`` lets a large effective batch run on a small GPU by
    summing gradients over several small batches before stepping. Needed for
    WavLM-Large on a 16GB card.

    ``scaler`` enables mixed precision. Half precision alone would let small
    gradients underflow to zero, so the scaler multiplies the loss up before the
    backward pass and divides it out before the step.
    """
    model.train()
    running, seen = 0.0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _to_device(batch, device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch["waveform"], batch.get("num_valid"))
            loss = criterion(outputs, batch)

        scaled = loss / grad_accum_steps
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.detach()) * batch["emotion"].size(0)
        seen += batch["emotion"].size(0)

    return running / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    class_names: Sequence[str],
    benign_class: int | None = None,
    alarm_classes: Sequence[int] = (),
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """Run over a split and return metrics plus raw predictions.

    Predictions and actor ids come back too, so per-speaker error analysis can
    be done later without re-running the model.
    """
    model.eval()
    running, seen = 0.0, 0
    true: List[int] = []
    predicted: List[int] = []
    actors: List[int] = []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _to_device(batch, device)
        outputs = model(batch["waveform"], batch.get("num_valid"))

        if criterion is not None:
            loss = criterion(outputs, batch)
            running += float(loss) * batch["emotion"].size(0)
            seen += batch["emotion"].size(0)

        true.extend(batch["emotion"].cpu().tolist())
        predicted.extend(outputs["emotion"].argmax(dim=-1).cpu().tolist())
        if "actor_id" in batch:
            actors.extend(batch["actor_id"].cpu().tolist())

    metrics: Dict[str, Any] = summarise_classification(true, predicted, class_names)
    metrics["loss"] = running / seen if seen else None
    metrics["most_confused"] = most_confused_pairs(
        torch.tensor(metrics["confusion_matrix"]).numpy(), class_names
    )

    # The error the application actually cares about: a calm or happy caller
    # being flagged as escalating.
    if benign_class is not None and alarm_classes:
        metrics["false_alarm_rate"] = false_alarm_rate(true, predicted, benign_class, alarm_classes)

    metrics["predictions"] = {"true": true, "predicted": predicted, "actor_id": actors}
    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: Sequence[str],
    epochs: int = 10,
    scheduler: Any = None,
    grad_accum_steps: int = 1,
    early_stopping_patience: int = 3,
    on_epoch_end: Any = None,
    max_batches: int | None = None,
    scaler: Any = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train, tracking the best epoch by validation score.

    Model selection uses unweighted average recall on validation, never the test
    split. Early stopping avoids burning GPU hours on epochs that no longer help,
    which matters on a 30-hour weekly budget.

    ``verbose`` prints one line per epoch. Without it a run is completely silent
    for its whole duration, so there is no way to tell a model that has stopped
    improving from one that is still climbing until it is already over. Tests
    pass ``verbose=False`` to keep their output readable.
    """
    history: List[EpochResult] = []
    best_uar = float("-inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] = {}
    since_improved = 0

    if verbose:
        # flush=True on every print below: when this script runs under a notebook
        # (`!python scripts/train.py`) stdout is block-buffered, so unflushed lines
        # sit in the buffer until the process exits -- which is exactly the silence
        # this logging exists to remove.
        print(f"{'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val UAR':>8}  "
              f"{'time':>7}  {'remaining':>9}", flush=True)

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler,
            grad_accum_steps, max_batches, scaler,
        )
        val = evaluate(model, val_loader, criterion, device, class_names, max_batches=max_batches)
        elapsed = time.time() - started

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val["loss"],
            val_uar=val["unweighted_average_recall"],
            seconds=round(elapsed, 1),
        )
        history.append(result)

        improved = result.val_uar > best_uar
        if improved:
            best_uar, best_epoch, since_improved = result.val_uar, epoch, 0
            # Keep a CPU copy so the best weights survive later epochs.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improved += 1

        if verbose:
            # Estimate from the mean of all epochs so far rather than the last one,
            # because the first epoch is slower (model download, cache warm-up).
            mean_seconds = sum(r.seconds for r in history) / len(history)
            remaining = _format_duration(mean_seconds * (epochs - epoch))
            val_loss_text = f"{result.val_loss:.4f}" if result.val_loss is not None else "-"
            # "best so far" is marked inline: an unbroken run of stars means the
            # model was still improving when training stopped, which is the signal
            # that the epoch budget was the binding constraint rather than the model.
            marker = "  <- best" if improved else ""
            print(f"{epoch:>5}  {result.train_loss:>10.4f}  {val_loss_text:>9}  "
                  f"{result.val_uar:>8.4f}  {result.seconds:>6.0f}s  {remaining:>9}{marker}",
                  flush=True)

        if on_epoch_end is not None:
            on_epoch_end(result, model)

        if since_improved >= early_stopping_patience:
            if verbose:
                print(f"stopping early: no improvement for {early_stopping_patience} epochs",
                      flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)

    if verbose and best_epoch == len(history) and len(history) == epochs:
        # Worth shouting about: the run ended on its best epoch, so the model was
        # still getting better. The score is a floor, not a ceiling.
        print(f"note: best epoch was the last one ({best_epoch} of {epochs}) -- the model was "
              f"still improving. Consider raising train.epochs.", flush=True)

    return {
        "history": [vars(r) for r in history],
        "best_epoch": best_epoch,
        "best_val_uar": best_uar,
        "epochs_run": len(history),
    }
