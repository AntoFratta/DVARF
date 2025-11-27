"""
Utilities to export SAM 3 segmentation masks to image files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .sam3_wrapper import Sam3Prediction


def _prediction_masks_to_numpy(prediction: Sam3Prediction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SAM 3 masks and scores to numpy arrays.

    Masks are returned as an array of shape (N, H, W) with float values in [0, 1].
    """
    masks = prediction.masks
    scores = prediction.scores

    if hasattr(masks, "detach"):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        # (N, 1, H, W) -> (N, H, W)
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {masks_np.shape}")

    if hasattr(scores, "detach"):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    return masks_np, scores_np


def save_sam3_masks_for_image(
    prediction: Sam3Prediction,
    class_id: int,
    image_id: str,
    output_root: Path,
    score_threshold: float = 0.0,
    max_masks: Optional[int] = None,
) -> List[Path]:
    """
    Save binary segmentation masks for a single image and class.

    Masks are saved as 8-bit PNG images in:
        output_root / f"{image_id}_c{class_id}_i{idx}.png"
    """
    masks_np, scores_np = _prediction_masks_to_numpy(prediction)

    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    count = 0

    for idx, (mask, score) in enumerate(zip(masks_np, scores_np)):
        if score < score_threshold:
            continue

        if max_masks is not None and count >= max_masks:
            break

        # Threshold mask to binary and convert to uint8 image.
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_bin)

        out_name = f"{image_id}_c{class_id}_i{idx}.png"
        out_path = output_root / out_name
        mask_img.save(out_path)

        saved_paths.append(out_path)
        count += 1

    return saved_paths

