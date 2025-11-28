from __future__ import annotations

from pathlib import Path
import sys

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
    split = "test"  # change here if you want to evaluate 'val' etc.
    confidence_threshold = 0.26

    labels_dir = get_labels_dir(split)
    preds_dir = get_sam3_yolo_predictions_dir(split)
    num_classes = len(CLASS_PROMPTS)

    print(f"Evaluating split '{split}'")
    print(f"Labels directory:     {labels_dir}")
    print(f"Predictions directory:{preds_dir}")
    print(f"Confidence threshold: {confidence_threshold}\n")

    result = evaluate_yolo_predictions(
        labels_dir=labels_dir,
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    )

    class_names = {cid: name for cid, name in CLASS_PROMPTS.items()}
    print_evaluation_summary(result, class_names=class_names)


if __name__ == "__main__":
    main()
