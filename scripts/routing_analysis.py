"""Score a finished run as the routing decision the application actually makes.

The model predicts six emotions. The deployed system makes one binary choice:
fast-track this caller to a human, or send them through the standard menu. Those
are different questions and they get different answers, so both are reported.

Runs on a saved results JSON -- no model, no GPU, no retraining. The predictions
were stored by `evaluate()` when the run finished, so this is a re-reading of
numbers that already exist.

    python scripts/routing_analysis.py --results results/baseline_lora.json
    python scripts/routing_analysis.py --results results/baseline_lora.json --split test

Two groupings are reported, because where the boundary is drawn is a design
choice rather than a fact, and the result is sensitive to it:

  distress  ANG FEA SAD DIS -- any caller in difficulty needs a person
  narrow    ANG DIS         -- only overt hostility, matching ALARM_EMOTIONS
                               in scripts/train.py

Neither is "correct". Report both and say which the system uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.labels import EMOTION_TO_INDEX  # noqa: E402
from src.evaluation import routing_metrics  # noqa: E402

GROUPINGS: Dict[str, Sequence[str]] = {
    "distress": ("ANG", "FEA", "SAD", "DIS"),
    "narrow": ("ANG", "DIS"),
}


def load_predictions(record: dict, split: str) -> tuple[List[int], List[int], List[str]]:
    """Pull the stored predictions out of a results JSON.

    Fails with a specific message rather than a KeyError: a run made before
    predictions were saved, or a --split that was never evaluated, are both
    ordinary situations and the fix differs.
    """
    metrics = record.get("metrics", {})
    if split not in metrics:
        available = ", ".join(sorted(metrics)) or "none"
        raise SystemExit(
            f"This run has no '{split}' metrics. Splits present: {available}.\n"
            "The test split is only scored when train.py is given --test."
        )
    predictions = metrics[split].get("predictions")
    if not predictions:
        raise SystemExit(
            f"No stored predictions in the '{split}' metrics, so the routing decision "
            "cannot be recomputed. Re-run training to regenerate this results file."
        )
    class_names = metrics[split].get("class_names") or list(EMOTION_TO_INDEX)
    return predictions["true"], predictions["predicted"], class_names


def report(name: str, result: Dict[str, object]) -> None:
    counts = result["confusion"]
    print(f"\n--- {name}: escalate on {' '.join(result['escalate_classes'])} ---")
    print(f"  routing balanced accuracy : {result['balanced_accuracy']:.4f}")
    print(f"  routing accuracy          : {result['accuracy']:.4f}")
    print(f"  escalate recall           : {result['escalate_recall']:.4f}"
          "   (callers needing a human who got one)")
    print(f"  escalate precision        : {result['escalate_precision']:.4f}"
          "   (fast-tracked callers who needed it)")
    print(f"  false alarm rate          : {result['false_alarm_rate']:.4f}"
          "   (ordinary callers wrongly fast-tracked)")
    print(f"  counts                    : escalated correctly {counts['true_positive']}, "
          f"missed {counts['false_negative']}, false alarms {counts['false_positive']}, "
          f"correctly routine {counts['true_negative']}")
    print(f"  six-way errors that route correctly anyway: {result['rescued_errors']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="path to results/<run_id>.json")
    parser.add_argument("--split", default="val", help="val (default) or test")
    parser.add_argument("--out", default=None, help="write the analysis alongside as JSON")
    args = parser.parse_args()

    record = json.loads(Path(args.results).read_text(encoding="utf-8"))
    true, predicted, class_names = load_predictions(record, args.split)

    six_way = record["metrics"][args.split]
    print(f"run: {record.get('run_id')}  split: {args.split}  clips: {len(true)}")
    print(f"six-way unweighted average recall: {six_way['unweighted_average_recall']:.4f}")

    analysis = {}
    for name, emotions in GROUPINGS.items():
        indices = [class_names.index(e) for e in emotions if e in class_names]
        analysis[name] = routing_metrics(true, predicted, indices, class_names)
        report(name, analysis[name])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"run_id": record.get("run_id"), "split": args.split, "routing": analysis},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
