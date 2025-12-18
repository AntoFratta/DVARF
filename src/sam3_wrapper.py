"""
Lightweight wrapper around the official SAM 3 image model.

This module centralizes all interactions with the SAM 3 API so that the rest
of the project does not depend on the low-level interface of the original
repository. This makes the code easier to maintain and test.
"""

from __future__ import annotations

import os
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
        # Explicitly load from HuggingFace to avoid bpe_path issues
        self.model = build_sam3_image_model(load_from_HF=True)
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

        # Debug utility: print output structure and shapes to understand
        # where query embeddings are exposed by the processor.
        # Enable with: export SAM3_DEBUG=1
        if os.environ.get("SAM3_DEBUG", "0") == "1":
            print("TOP KEYS:", list(output.keys()))
            bo = output.get("backbone_out", None)
            print("backbone_out type:", type(bo))
            if isinstance(bo, dict):
                print("BACKBONE_OUT KEYS:", list(bo.keys()))
                if "queries" in bo and hasattr(bo["queries"], "shape"):
                    print("backbone_out['queries'] shape:", bo["queries"].shape)
            if "boxes" in output and hasattr(output["boxes"], "shape"):
                print("boxes shape:", output["boxes"].shape)

        # Extract query embeddings (256-d) from last decoder layer.
        # These embeddings contain semantic information about the detected objects
        # and their compatibility with the text prompt.
        # Shape: (num_queries, 256)
        #
        # NOTE: When using Sam3Processor, the decoder outputs (including queries)
        # are often exposed inside `output["backbone_out"]`, not necessarily at the top level.
        query_features = output.get("queries", None)

        if query_features is None:
            backbone_out = output.get("backbone_out", None)
            if isinstance(backbone_out, dict):
                query_features = backbone_out.get("queries", None)

        # CRITICAL: Queries are REQUIRED for the linear probe pipeline
        if query_features is None:
            raise RuntimeError(
                f"SAM3 did not expose query embeddings ('queries') for image {image_path}. "
                f"The linear probe requires semantic features (query embeddings). "
                f"Expected either output['queries'] or output['backbone_out']['queries']. "
                f"Output keys: {list(output.keys())}"
            )

        # If the model returns a batch dimension (1, N, 256), remove it.
        if hasattr(query_features, "ndim") and query_features.ndim == 3 and query_features.shape[0] == 1:
            query_features = query_features[0]

        # Validate that queries align with boxes.
        # In this project we assume a strict 1:1 alignment between:
        # - the i-th detection box written to the YOLO file
        # - the i-th embedding saved to the .npz features file
        num_queries = query_features.shape[0] if hasattr(query_features, "shape") else len(query_features)
        num_boxes = output["boxes"].shape[0] if hasattr(output["boxes"], "shape") else len(output["boxes"])

        if num_queries != num_boxes:
            raise RuntimeError(
                f"SAM3 queries/boxes length mismatch: {num_queries} queries but {num_boxes} boxes. "
                f"Cannot guarantee feature alignment. "
                f"(Hint: enable SAM3_DEBUG=1 to inspect output/backbone_out keys.)"
            )

        return Sam3Prediction(
            masks=output["masks"],
            boxes=output["boxes"],
            scores=output["scores"],
            features=query_features,
        )
