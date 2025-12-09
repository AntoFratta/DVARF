from __future__ import annotations

"""
Evaluate SAM 3 detections AFTER applying the linear probe.

This script is analogous to `eval_sam3_on_split.py`, but it reads
predictions from:

    data/processed/predictions/sam3_linear_probe_yolo/<split>/*.txt

instead of:

    data/processed/predictions/sam3_yolo/<split>/*.txt

Metrics are computed using `src.eval_yolo.evaluate_yolo_predictions`,
so they are directly comparable to:
- the original SAM 3 zero-shot results, and
- the YOLO-based models from the original project.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH so that "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_labels_dir,
    PREDICTIONS_DIR,
)
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    evaluate_yolo_predictions,
    print_evaluation_summary,
)


def eval_sam3_linear_probe_on_split(
    split: str = "test",
    confidence_threshold: float = 0.26,
) -> None:
    """
    Evaluate SAM 3 + linear probe detections on a given split.

    Args:
        split:
            Dataset split to evaluate ('train', 'val', or 'test').
        confidence_threshold:
            Score threshold applied when loading the predictions.
            This should match the threshold used for zero-shot SAM 3
            to enable a fair comparison.
    """
    # Ground-truth labels (YOLO format) for this split.
    labels_dir = get_labels_dir(split)

    # Directory containing YOLO predictions AFTER applying the linear probe.
    preds_dir = PREDICTIONS_DIR / "sam3_linear_probe_yolo" / split

    # Basic run configuration logging.
    print(f"Evaluating split '{split}' (SAM 3 + linear probe)")
    print(f"Labels directory:     {labels_dir}")
    print(f"Predictions directory:{preds_dir}")
    print(f"Confidence threshold: {confidence_threshold}\n")

    # Number of classes is inferred from the prompt mapping used in the project.
    num_classes = len(CLASS_PROMPTS)

    # Run generic YOLO-style evaluation (AP, mAP, precision/recall, ...).
    result = evaluate_yolo_predictions(
        labels_dir=labels_dir,
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    )

    # Map class ids to human-readable names (the same text prompts used for SAM 3).
    class_names = {cid: name for cid, name in CLASS_PROMPTS.items()}

    # Pretty-print a per-class and global summary of the evaluation metrics.
    print_evaluation_summary(result, class_names=class_names)


def main() -> None:
    """
    CLI entry point.

    By default, this evaluates the 'test' split at the same confidence
    threshold used for the zero-shot SAM 3 experiment.
    """
    # Split and confidence threshold used for evaluation.
    split = "test"
    confidence_threshold = 0.26

    eval_sam3_linear_probe_on_split(
        split=split,
        confidence_threshold=confidence_threshold,
    )


if __name__ == "__main__":
    main()
