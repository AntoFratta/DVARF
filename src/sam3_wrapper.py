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
    masks: Any      # usually a tensor of shape [N, 1, H, W]
    boxes: Any      # usually a tensor of shape [N, 4] in pixel coordinates
    scores: Any     # usually a tensor of shape [N]
    features: Any   # query embeddings (N, 256) from last decoder layer


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

        # Extract query embeddings (256-d) from last decoder layer.
        # These embeddings contain semantic information about the detected objects
        # and their compatibility with the text prompt.
        # Shape: (num_queries, 256)
        query_features = output.get("queries", None)
        
        # CRITICAL: Validate that queries align with boxes
        if query_features is not None:
            num_queries = query_features.shape[0] if hasattr(query_features, "shape") else len(query_features)
            num_boxes = output["boxes"].shape[0] if hasattr(output["boxes"], "shape") else len(output["boxes"])
            
            if num_queries != num_boxes:
                raise RuntimeError(
                    f"SAM3 queries/boxes length mismatch: {num_queries} queries but {num_boxes} boxes. "
                    f"Cannot guarantee feature alignment. This indicates a bug in SAM3 output."
                )

        return Sam3Prediction(
            masks=output["masks"],
            boxes=output["boxes"],
            scores=output["scores"],
            features=query_features,
        )
