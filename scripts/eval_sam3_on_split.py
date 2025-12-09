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

# Add project root to PYTHONPATH so "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import get_labels_dir, get_sam3_yolo_predictions_dir  # noqa: E402
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    evaluate_yolo_predictions,
    print_evaluation_summary,
)


def main() -> None:
    """
    Evaluate SAM 3 YOLO predictions on a specific dataset split.

    The script:
    - selects the split to evaluate (by default 'test'),
    - sets a confidence threshold to filter predictions,
    - runs YOLO-style evaluation given GT labels and prediction files,
    - and prints a human-readable summary of per-class and global metrics.
    """
    # Choose which split to evaluate. Change to "val" or "train" if needed.
    split = "test"  # change here if you want to evaluate 'val' etc.

    # Minimum confidence for predictions to be considered in evaluation.
    confidence_threshold = 0.26

    # Directories containing ground-truth labels and SAM 3 predictions for this split.
    labels_dir = get_labels_dir(split)
    preds_dir = get_sam3_yolo_predictions_dir(split)

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
