"""
Lightweight wrapper around the official SAM 3 image model.

This module centralizes all interactions with the SAM 3 API so that the rest
of the project does not depend on the low-level interface of the original
repository. This makes the code easier to maintain and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PathLike = Union[str, Path]


@dataclass
class Sam3Prediction:
    """Masks, bounding boxes and scores returned by SAM 3."""
    masks: Any   # usually a tensor of shape [N, 1, H, W]
    boxes: Any   # usually a tensor of shape [N, 4] in pixel coordinates
    scores: Any  # usually a tensor of shape [N]


class Sam3ImageModel:
    """Wrapper for single-image inference with a text prompt."""

    def __init__(self) -> None:
        """Build the SAM 3 image model and its processor."""
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

    def predict_with_text(self, image_path: PathLike, prompt: str) -> Sam3Prediction:
        """
        Run SAM 3 on a single image using a single text prompt.
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        state = self.processor.set_image(image)
        output: Dict[str, Any] = self.processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

        return Sam3Prediction(
            masks=output["masks"],
            boxes=output["boxes"],
            scores=output["scores"],
        )
