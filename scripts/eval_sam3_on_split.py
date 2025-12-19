"""
Evaluate SAM 3 YOLO-style predictions against ground-truth labels for a split.

This script:
- locates the ground-truth YOLO labels and SAM 3 prediction files for a given split,
- calls the generic YOLO evaluation routine (evaluate_yolo_predictions),
- and prints a per-class and overall summary of detection metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

# Add project root to PYTHONPATH so "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_labels_dir, get_sam3_yolo_predictions_dir  # noqa: E402
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    evaluate_yolo_predictions,
    print_evaluation_summary,
)  # noqa: E402


def main() -> None:
    """
    Evaluate SAM 3 YOLO predictions on a specific dataset split.

    The script:
    - selects the split to evaluate (by default 'test'),
    - sets a confidence threshold to filter predictions,
    - runs YOLO-style evaluation given GT labels and prediction files,
    - and prints a human-readable summary of per-class and global metrics.
    """
    parser = argparse.ArgumentParser(description="Evaluate SAM3 predictions on a split.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (train/val/test).")
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.26, help="Minimum confidence for predictions."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "probe"],
        help="Which predictions to evaluate: 'baseline' (sam3_yolo) or 'probe' (sam3_linear_probe_yolo).",
    )
    args = parser.parse_args()

    split = args.split
    confidence_threshold = float(args.confidence_threshold)

    # Directories containing ground-truth labels and SAM 3 predictions for this split.
    labels_dir = get_labels_dir(split)

    if args.mode == "baseline":
        preds_dir = get_sam3_yolo_predictions_dir(split)
    else:
        # Keep logic minimal: same as your linear-probe eval script did.
        preds_dir = PROJECT_ROOT / "data" / "processed" / "predictions" / "sam3_linear_probe_yolo" / split

    # Number of classes is derived from the CLASS_PROMPTS mapping.
    num_classes = len(CLASS_PROMPTS)

    print(f"Evaluating split '{split}'")
    print(f"Labels directory:     {labels_dir}")
    print(f"Predictions directory:{preds_dir}")
    print(f"Confidence threshold: {confidence_threshold}\n")

    # Run YOLO-style evaluation: this function is responsible for computing
    # TP/FP/FN, per-class AP, mAP, etc., given labels and predictions.
    result = evaluate_yolo_predictions(
        labels_dir=labels_dir,
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    )

    # Build a simple mapping from class id to human-readable class name,
    # reused when printing the summary.
    class_names = {cid: name for cid, name in CLASS_PROMPTS.items()}

    # Pretty-print a summary of metrics (per class and aggregated).
    print_evaluation_summary(result, class_names=class_names)


if __name__ == "__main__":
    main()
