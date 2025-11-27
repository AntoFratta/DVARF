from __future__ import annotations

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import get_labels_dir, get_sam3_yolo_predictions_dir  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    GTBox,
    PredBox,
    yolo_to_xyxy_norm,
    evaluate_detections,
)
from src.prompts import CLASS_PROMPTS  # noqa: E402


def load_gt_and_predictions(split: str):
    """
    Load ground-truth and prediction boxes for images that have predictions.

    Predictions are expected in the format:
    class_id cx cy w h score
    """
    labels_dir = get_labels_dir(split)
    preds_dir = get_sam3_yolo_predictions_dir(split)

    gt_boxes = []
    pred_boxes = []

    label_files = sorted(labels_dir.glob("*.txt"), key=lambda p: int(p.stem))

    for label_path in label_files:
        image_id = label_path.stem
        pred_path = preds_dir / f"{image_id}.txt"

        if not pred_path.exists():
            continue

        # Ground-truth boxes
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                box = yolo_to_xyxy_norm(cx, cy, w, h)
                gt_boxes.append(GTBox(image_id=image_id, class_id=class_id, box=box))

        # Predictions
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                if len(parts) >= 6:
                    try:
                        score = float(parts[5])
                    except ValueError:
                        score = 1.0
                else:
                    score = 1.0

                box = yolo_to_xyxy_norm(cx, cy, w, h)
                pred_boxes.append(
                    PredBox(
                        image_id=image_id,
                        class_id=class_id,
                        box=box,
                        score=score,
                    )
                )

    return gt_boxes, pred_boxes


def main() -> None:
    split = "test"
    iou_threshold = 0.5
    confidence_threshold = 0.26


    class_ids = sorted(CLASS_PROMPTS.keys())
    num_classes = max(class_ids) + 1 if class_ids else 0

    gt_boxes, pred_boxes = load_gt_and_predictions(split)

    print(
        f"Loaded {len(gt_boxes)} ground-truth boxes and "
        f"{len(pred_boxes)} predictions for split '{split}'."
    )
    if not gt_boxes or not pred_boxes:
        print("No data to evaluate. Make sure predictions exist for this split.")
        return

    aps, mAP = evaluate_detections(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    id_to_name = {cid: name for cid, name in CLASS_PROMPTS.items()}

    print(f"\nResults on split '{split}' (IoU >= {iou_threshold}, score >= {confidence_threshold}):\n")
    for class_id in sorted(aps.keys()):
        name = id_to_name.get(class_id, f"class_{class_id}")
        print(f"  Class {class_id} ({name}): AP = {aps[class_id]:.4f}")

    print(f"\n  mAP@{iou_threshold:.2f} over {num_classes} classes: {mAP:.4f}")


if __name__ == "__main__":
    main()
