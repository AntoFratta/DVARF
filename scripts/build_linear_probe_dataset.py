from __future__ import annotations

"""
Build a training dataset for a simple linear probe on top of SAM 3 detections.

For a given split (e.g. "train"), this script:
- Loads YOLO-style ground-truth boxes.
- Loads SAM 3 YOLO-style predictions for the same split.
- Matches predictions to ground truth with IoU >= 0.5 (same logic as eval_yolo).
- For each prediction, extracts a small feature vector that depends ONLY on
  the prediction itself (no GT information), e.g.:
    [score, cx, cy, w, h, area, aspect_ratio]
- Assigns a binary label:
    1 = true positive (TP), 0 = false positive (FP) according to the matching.
- Saves the resulting features and labels to:
    data/processed/linear_probe/sam3_linear_probe_<split>.npz
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to PYTHONPATH so that "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import get_labels_dir, get_sam3_yolo_predictions_dir  # noqa: E402
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    _load_yolo_dataset,
    _compute_iou,
)


def _box_features_xyxy(box_xyxy: np.ndarray, score: float) -> np.ndarray:
    """
    Compute a small feature vector from a normalized xyxy box and its score.

    Box coordinates are expected to be normalized in [0, 1], as returned by
    _load_yolo_dataset in eval_yolo.py.
    """
    # Unpack the normalized corner coordinates [x1, y1, x2, y2].
    x1, y1, x2, y2 = box_xyxy

    # Basic geometric properties of the box in normalized coordinates.
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    area = w * h
    aspect_ratio = w / max(h, 1e-6)  # avoid division by zero

    # Feature vector used by the linear probe:
    # [score, center_x, center_y, width, height, area, aspect_ratio]
    return np.asarray(
        [float(score), cx, cy, w, h, area, aspect_ratio],
        dtype=np.float32,
    )


def build_linear_probe_dataset_for_split(
    split: str,
    confidence_threshold: float = 0.26,
    iou_threshold: float = 0.5,
) -> Path:
    """
    Build the linear-probe dataset for a single split (e.g. 'train').

    Args:
        split:
            Dataset split to process ('train', 'val', or 'test').
        confidence_threshold:
            Score threshold applied when loading SAM 3 predictions.
            Only predictions with score >= threshold are considered.
            This should usually match the threshold used in evaluation.
        iou_threshold:
            IoU threshold used to decide if a prediction is TP or FP,
            consistent with the detection evaluation (typically 0.5).

    Returns:
        Path to the saved .npz file containing:
            - features: float32 array of shape (N, D)
            - targets: int64 array of shape (N,) with values {0, 1}
            - class_ids: int64 array of shape (N,) with class indices
    """
    # Get directories for ground-truth labels and SAM 3 predictions.
    labels_dir = get_labels_dir(split)
    preds_dir = get_sam3_yolo_predictions_dir(split)
    num_classes = len(CLASS_PROMPTS)

    print(f"Building linear-probe dataset for split='{split}'")
    print(f"  Labels directory:      {labels_dir}")
    print(f"  Predictions directory: {preds_dir}")
    print(f"  Confidence threshold:  {confidence_threshold}")
    print(f"  IoU threshold:         {iou_threshold}")

    # Load GT and predictions in YOLO format, grouped by class and image.
    # - gt_by_class[class_id][img_id] -> array of GT boxes (xyxy, normalized)
    # - preds_by_class[class_id] -> list of (img_id, box_xyxy, score)
    gt_by_class, preds_by_class = _load_yolo_dataset(
        labels_dir=labels_dir,
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    )

    all_features: List[np.ndarray] = []
    all_targets: List[int] = []
    all_class_ids: List[int] = []

    # For per-class statistics / debugging: class_id -> (num_pos, num_neg)
    per_class_counts: Dict[int, Tuple[int, int]] = {}

    # Loop over all classes; we build TP/FP labels independently per class.
    for class_id in range(num_classes):
        gt_dict = gt_by_class.get(class_id, {})
        preds = preds_by_class.get(class_id, [])

        # Count how many GT boxes we have for this class (over all images).
        num_gt = sum(len(v) for v in gt_dict.values())
        if num_gt == 0 and not preds:
            print(f"  Class {class_id}: no GT and no predictions, skipping.")
            continue

        print(
            f"  Class {class_id}: {num_gt} GT boxes, "
            f"{len(preds)} predictions before matching."
        )

        # For each image, keep track of which GT boxes have already been
        # "claimed" by a higher-score prediction (greedy matching).
        matched: Dict[str, np.ndarray] = {
            img_id: np.zeros(len(boxes), dtype=bool)
            for img_id, boxes in gt_dict.items()
        }

        # Sort predictions by descending score, as in standard detection eval.
        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)

        num_pos = 0  # number of true positives for this class
        num_neg = 0  # number of false positives for this class

        # Iterate over predictions from highest to lowest score.
        for img_id, box_pred, score in preds_sorted:
            gt_boxes = gt_dict.get(img_id)
            if gt_boxes is None or gt_boxes.size == 0:
                # No GT boxes for this image and class: prediction is a FP.
                target = 0
            else:
                # Compute IoU with all GT boxes in this image for this class.
                ious = _compute_iou(box_pred, gt_boxes)
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_threshold and not matched[img_id][best_idx]:
                    # This prediction becomes a TP and "claims" the GT box
                    # (so it cannot be matched again by lower-score predictions).
                    target = 1
                    matched[img_id][best_idx] = True
                else:
                    # Either IoU is too low, or the GT box was already claimed:
                    # in both cases, this prediction is treated as a FP.
                    target = 0

            # Compute the feature vector for this prediction (box + score).
            feats = _box_features_xyxy(box_pred, score)

            # Append features and corresponding target (0/1) and class id.
            all_features.append(feats)
            all_targets.append(int(target))
            all_class_ids.append(int(class_id))

            if target == 1:
                num_pos += 1
            else:
                num_neg += 1

        per_class_counts[class_id] = (num_pos, num_neg)
        print(
            f"    -> positives: {num_pos}, negatives: {num_neg} "
            f"(pos ratio: {num_pos / max(num_pos + num_neg, 1):.3f})"
        )

    # If no predictions survived the confidence threshold, we cannot build a dataset.
    if not all_features:
        raise RuntimeError(
            f"No predictions found above threshold {confidence_threshold} "
            f"for split '{split}'. Cannot build linear-probe dataset."
        )

    # Stack and convert lists to final NumPy arrays.
    features_arr = np.stack(all_features, axis=0).astype(np.float32)
    targets_arr = np.asarray(all_targets, dtype=np.int64)
    class_ids_arr = np.asarray(all_class_ids, dtype=np.int64)

    # Create output directory for linear probe datasets if it does not exist.
    out_dir = PROJECT_ROOT / "data" / "processed" / "linear_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sam3_linear_probe_{split}.npz"

    # Save compressed .npz with features, binary targets and class ids.
    np.savez_compressed(
        out_path,
        features=features_arr,
        targets=targets_arr,
        class_ids=class_ids_arr,
    )

    print(f"\nSaved linear-probe dataset to: {out_path}")
    print(f"  Total samples: {features_arr.shape[0]}")
    print(f"  Feature dimension: {features_arr.shape[1]}")
    for class_id, (num_pos, num_neg) in per_class_counts.items():
        print(
            f"  Class {class_id}: {num_pos} pos, {num_neg} neg "
            f"(ratio={num_pos / max(num_pos + num_neg, 1):.3f})"
        )

    return out_path


def main() -> None:
    """
    Entry point for command-line use.

    By default, this builds the linear-probe dataset for the 'train' split.
    You can manually change the split to 'val' or 'test' if needed.
    """
    split = "train"
    confidence_threshold = 0.26
    iou_threshold = 0.5

    build_linear_probe_dataset_for_split(
        split=split,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )


if __name__ == "__main__":
    main()
