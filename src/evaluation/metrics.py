"""Scoring. Every number reported in the write-up comes from here."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(true: Sequence[int], predicted: Sequence[int], num_classes: int) -> np.ndarray:
    """Rows are the true class, columns the predicted one."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, predicted):
        matrix[int(t), int(p)] += 1
    return matrix


def per_class_recall(matrix: np.ndarray) -> np.ndarray:
    """Of the clips that really were class i, what fraction did we get right.

    Classes with no examples come back as NaN rather than 0, so they can be
    skipped instead of silently dragging the average down.
    """
    totals = matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(matrix) / totals
    return np.where(totals > 0, recall, np.nan)


def unweighted_average_recall(true: Sequence[int], predicted: Sequence[int], num_classes: int) -> float:
    """The primary metric: average of the per-class recalls.

    Plain accuracy is misleading here. CREMA-D has far fewer neutral clips than
    the other emotions, so a model that never predicts neutral can still post a
    good accuracy. Averaging recall per class gives every emotion equal say,
    which is why this is the standard metric in speech emotion work.
    """
    recall = per_class_recall(confusion_matrix(true, predicted, num_classes))
    return float(np.nanmean(recall))


def accuracy(true: Sequence[int], predicted: Sequence[int]) -> float:
    """Reported alongside, never as the headline. See unweighted_average_recall."""
    if not len(true):
        return float("nan")
    return float(np.mean(np.asarray(true) == np.asarray(predicted)))


def false_alarm_rate(
    true: Sequence[int], predicted: Sequence[int], benign_class: int, alarm_classes: Sequence[int]
) -> float:
    """How often genuinely benign speech triggers an alert.

    For an escalation system this is the expensive mistake: flagging a delighted
    customer as furious costs the agent's trust in the tool, while a missed
    detection merely leaves things as they are today. Reported separately from
    overall accuracy because it is the number the application actually cares
    about.

    Returns NaN when the benign class has no examples.
    """
    true_arr = np.asarray(true)
    predicted_arr = np.asarray(predicted)
    benign = true_arr == benign_class
    if not benign.any():
        return float("nan")
    alarms = np.isin(predicted_arr[benign], list(alarm_classes))
    return float(alarms.mean())


def summarise_classification(
    true: Sequence[int],
    predicted: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Everything the results JSON needs for one classification head."""
    num_classes = len(class_names)
    matrix = confusion_matrix(true, predicted, num_classes)
    recall = per_class_recall(matrix)
    return {
        "unweighted_average_recall": float(np.nanmean(recall)),
        "accuracy": accuracy(true, predicted),
        "per_class_recall": {
            name: (None if np.isnan(r) else round(float(r), 4)) for name, r in zip(class_names, recall)
        },
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
        "support": matrix.sum(axis=1).tolist(),
    }


def routing_metrics(
    true: Sequence[int],
    predicted: Sequence[int],
    escalate_classes: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Score the application's actual decision: fast-track this caller, or don't.

    The model predicts six emotions, but the deployed system makes a binary
    choice. Collapsing the six-way prediction into that choice is not the same
    question, and answers it differently: a clip whose true label is fear and
    whose predicted label is anger is a six-way error but a *correct* routing
    decision, because both send the caller to a human.

    Reported alongside the six-way score, never instead of it. The six-way number
    measures the model; this measures the application. The gap between them is
    itself a result, and quoting only the more flattering one would be dishonest.

    ``rescued_errors`` counts exactly that gap -- six-way mistakes that land on
    the correct side of the routing boundary. ``introduced_errors`` is always 0
    by construction (a correct six-way prediction cannot route wrongly) and is
    returned anyway so the arithmetic is checkable by a reader.

    Naming: "escalate" is the positive class throughout -- the caller needs a
    human. Recall is therefore the share of callers needing help who got it, and
    the false alarm rate is the share of ordinary callers wrongly fast-tracked.
    """
    true_arr = np.asarray(true)
    predicted_arr = np.asarray(predicted)
    escalate = list(escalate_classes)

    true_escalate = np.isin(true_arr, escalate)
    predicted_escalate = np.isin(predicted_arr, escalate)

    true_positive = int((true_escalate & predicted_escalate).sum())
    false_negative = int((true_escalate & ~predicted_escalate).sum())
    false_positive = int((~true_escalate & predicted_escalate).sum())
    true_negative = int((~true_escalate & ~predicted_escalate).sum())

    def _safe(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    escalate_recall = _safe(true_positive, true_positive + false_negative)
    standard_recall = _safe(true_negative, true_negative + false_positive)

    six_way_correct = true_arr == predicted_arr
    routed_correct = true_escalate == predicted_escalate
    return {
        "escalate_classes": [class_names[i] for i in escalate],
        "standard_classes": [n for i, n in enumerate(class_names) if i not in escalate],
        "confusion": {
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "true_negative": true_negative,
        },
        "accuracy": _safe(true_positive + true_negative, len(true_arr)),
        # Average of the two recalls -- the binary analogue of the project's
        # primary metric, and immune to the escalate group being the larger one.
        "balanced_accuracy": float(np.nanmean([escalate_recall, standard_recall])),
        "escalate_recall": escalate_recall,
        "escalate_precision": _safe(true_positive, true_positive + false_positive),
        # The expensive error: an ordinary caller pushed into the priority queue.
        "false_alarm_rate": _safe(false_positive, false_positive + true_negative),
        "rescued_errors": int((~six_way_correct & routed_correct).sum()),
        "introduced_errors": int((six_way_correct & ~routed_correct).sum()),
    }


def most_confused_pairs(matrix: np.ndarray, class_names: Sequence[str], top_k: int = 3) -> List[Dict[str, object]]:
    """The biggest off-diagonal cells, for the error analysis section.

    Expected here: happy and angry. Both are loud, fast and high-pitched, so
    they are hard to separate from sound alone.
    """
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(matrix[i, j])})
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top_k]
