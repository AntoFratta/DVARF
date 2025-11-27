"""
Utility functions to convert SAM 3 predictions into YOLO-style bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .sam3_wrapper import Sam3Prediction


@dataclass
class YoloBox:
    """YOLO-format bounding box, optionally with a confidence score."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    score: Optional[float] = None


def sam3_boxes_to_yolo(
    prediction: Sam3Prediction,
    class_id: int,
    image_width: int,
    image_height: int,
    score_threshold: float = 0.0,
) -> List[YoloBox]:
    """
    Convert SAM 3 pixel-space boxes into YOLO-normalized boxes for a single class.

    SAM 3 boxes are expected in the format [x1, y1, x2, y2] in pixel coordinates.
    The output is in YOLO format (cx, cy, w, h) normalized to [0, 1].
    """
    boxes = prediction.boxes
    scores = prediction.scores

    if hasattr(boxes, "detach"):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    if hasattr(scores, "detach"):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    yolo_boxes: List[YoloBox] = []

    for box, score in zip(boxes_np, scores_np):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        cx_norm = cx / image_width
        cy_norm = cy / image_height
        bw_norm = bw / image_width
        bh_norm = bh / image_height

        yolo_boxes.append(
            YoloBox(
                class_id=class_id,
                cx=cx_norm,
                cy=cy_norm,
                w=bw_norm,
                h=bh_norm,
                score=float(score),
            )
        )

    return yolo_boxes


def yolo_boxes_to_lines(
    boxes: List[YoloBox],
    include_score_column: bool = False,
    include_score_comment: bool = False,
) -> List[str]:
    """
    Convert a list of YoloBox objects into YOLO-style text lines.

    If include_score_column is True, the confidence is written as a 6th numeric
    column: class cx cy w h score.

    If include_score_comment is True, the confidence is appended as a comment.
    """
    lines: List[str] = []

    for box in boxes:
        base = f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"

        if include_score_column and box.score is not None:
            base = f"{base} {box.score:.6f}"
        elif include_score_comment and box.score is not None:
            base = f"{base}  # score={box.score:.3f}"

        lines.append(base)

    return lines
