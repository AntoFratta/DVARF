from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class GTBox:
    """Ground-truth bounding box in normalized xyxy format."""
    image_id: str
    class_id: int
    box: np.ndarray  # [x1, y1, x2, y2] in normalized coordinates


@dataclass
class PredBox:
    """Predicted bounding box in normalized xyxy format with a confidence score."""
    image_id: str
    class_id: int
    box: np.ndarray  # [x1, y1, x2, y2] in normalized coordinates
    score: float


def yolo_to_xyxy_norm(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """
    Convert YOLO (cx, cy, w, h) normalized box to xyxy normalized box.
    """
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two xyxy normalized boxes.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    denom = area_a + area_b - inter_area + 1e-6
    return float(inter_area / denom) if denom > 0.0 else 0.0


def evaluate_detections(
    gt_boxes: List[GTBox],
    pred_boxes: List[PredBox],
    num_classes: int,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> Tuple[Dict[int, float], float]:
    """
    Compute AP per class and mAP using 11-point interpolation (VOC-style).

    The evaluation follows the standard mAP@0.5 setup:
    - predictions are filtered by confidence_threshold,
    - predictions are sorted by score in descending order,
    - IoU threshold is applied to match predictions to ground truth.
    """
    # Filter predictions by score threshold (analogous to CONFIDENCE_THRESHOLD).
    pred_boxes = [p for p in pred_boxes if p.score >= confidence_threshold]

    # Ground truths grouped by (image_id, class_id)
    gt_by_image_class: Dict[Tuple[str, int], List[np.ndarray]] = {}
    for gt in gt_boxes:
        key = (gt.image_id, gt.class_id)
        gt_by_image_class.setdefault(key, []).append(gt.box)

    # Matched flags for each ground-truth list
    matched_by_image_class: Dict[Tuple[str, int], np.ndarray] = {
        key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_by_image_class.items()
    }

    # Ground-truth counts per class
    gt_counts: Dict[int, int] = {c: 0 for c in range(num_classes)}
    for gt in gt_boxes:
        gt_counts[gt.class_id] += 1

    aps: Dict[int, float] = {}

    for class_id in range(num_classes):
        preds_c = [p for p in pred_boxes if p.class_id == class_id]

        if not preds_c:
            aps[class_id] = 0.0
            continue

        preds_c.sort(key=lambda p: p.score, reverse=True)

        tps = np.zeros(len(preds_c), dtype=np.float32)
        fps = np.zeros(len(preds_c), dtype=np.float32)

        for i, pred in enumerate(preds_c):
            key = (pred.image_id, class_id)
            gt_boxes_c = gt_by_image_class.get(key, [])

            if not gt_boxes_c:
                fps[i] = 1.0
                continue

            matched = matched_by_image_class[key]
            best_iou = 0.0
            best_j = -1

            for j, gt_box in enumerate(gt_boxes_c):
                if matched[j]:
                    continue
                iou = compute_iou(pred.box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                tps[i] = 1.0
                matched[best_j] = True
            else:
                fps[i] = 1.0

        if gt_counts[class_id] == 0:
            aps[class_id] = 0.0
            continue

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)

        recalls = cum_tp / (gt_counts[class_id] + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        recall_levels = np.linspace(0.0, 1.0, 11)
        interpolated_precisions = []

        for r in recall_levels:
            mask = recalls >= r
            if np.any(mask):
                interpolated_precisions.append(float(np.max(precisions[mask])))
            else:
                interpolated_precisions.append(0.0)

        aps[class_id] = float(np.mean(interpolated_precisions))

    mAP = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps, mAP
