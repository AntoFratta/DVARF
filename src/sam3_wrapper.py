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
from typing import Any, Dict, Union, Optional

from PIL import Image
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PathLike = Union[str, Path]


@dataclass
class Sam3Prediction:
    """Masks, bounding boxes and scores returned by SAM 3."""
    masks: Any      # usually a tensor of shape [N, 1, H, W]
    boxes: Any      # usually a tensor of shape [N, 4] in pixel coordinates
    scores: Any     # usually a tensor of shape [N]
    features: Any   # per-box semantic embeddings (N, 256) aligned to final boxes


class Sam3ImageModel:
    """Wrapper for single-image inference with a text prompt."""

    def __init__(self) -> None:
        """Build the SAM 3 image model and its processor."""
        # Explicitly load from HuggingFace to avoid bpe_path issues
        self.model = build_sam3_image_model(load_from_HF=True)
        self.processor = Sam3Processor(self.model)

    @staticmethod
    def _extract_last_stage_out(model_forward_output: Any) -> Optional[Dict[str, Any]]:
        """
        SAM3Image.forward(...) returns a nested structure like:
          previous_stages_out = [ [stage0_out, stage1_out, ...] ]  # 1 frame
        We want the last stage dict.
        """
        out = model_forward_output

        # Typical: list (frames) -> list (stages) -> dict
        if isinstance(out, list) and out:
            if isinstance(out[0], list) and out[0]:
                last = out[0][-1]
                return last if isinstance(last, dict) else None
            # Sometimes might be list of dicts
            if isinstance(out[-1], dict):
                return out[-1]

        # Already a dict
        if isinstance(out, dict):
            return out

        return None

    @staticmethod
    def _boxes_xyxy_norm_to_pixels(boxes_xyxy_norm: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        Convert normalized [0..1] XYXY boxes to pixel XYXY.
        boxes_xyxy_norm: (Q, 4)
        """
        scale = torch.tensor([w, h, w, h], device=boxes_xyxy_norm.device, dtype=boxes_xyxy_norm.dtype)
        return boxes_xyxy_norm * scale

    @staticmethod
    def _match_boxes_to_queries(
        final_boxes_xyxy_px: torch.Tensor,
        raw_boxes_xyxy_px: torch.Tensor,
        match_threshold_px: float = 5.0,
    ) -> torch.Tensor:
        """
        Match each final box (from processor) to the closest raw per-query box (from model).
        Returns indices into raw_boxes (and thus into raw queries).
        """
        if final_boxes_xyxy_px.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=raw_boxes_xyxy_px.device)

        # Pairwise L2 distances: (N, Q)
        # final: (N,4), raw:(Q,4)
        diff = final_boxes_xyxy_px[:, None, :] - raw_boxes_xyxy_px[None, :, :]
        dists = torch.sqrt((diff * diff).sum(dim=-1))  # (N, Q)

        nn_idx = torch.argmin(dists, dim=1)           # (N,)
        nn_dist = dists[torch.arange(dists.size(0), device=dists.device), nn_idx]

        # If any match is too far, better to fail loudly: alignment would be wrong.
        if torch.any(nn_dist > match_threshold_px):
            bad = torch.nonzero(nn_dist > match_threshold_px).flatten().tolist()
            raise RuntimeError(
                f"SAM3 feature alignment failed: {len(bad)} boxes could not be matched "
                f"within {match_threshold_px}px. Bad indices: {bad[:10]}. "
                f"Try increasing match_threshold_px or inspect box scaling."
            )

        return nn_idx

    def predict_with_text(self, image_path: PathLike, prompt: str) -> Sam3Prediction:
        """
        Run SAM 3 on a single image using a single text prompt.
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        # Enable with: export SAM3_DEBUG=1
        debug = os.environ.get("SAM3_DEBUG", "0") == "1"

        # We need semantic embeddings for the linear probe.
        # Sam3Processor does NOT expose 'queries' in its output dict in the current SAM3 version.
        # Solution: capture the raw model forward output via a forward hook, then align queries
        # to the final boxes returned by the processor.
        captured: Dict[str, Any] = {"raw_out": None}

        def _forward_hook(_module, _inputs, output):
            captured["raw_out"] = self._extract_last_stage_out(output)

        hook_handle = self.model.register_forward_hook(_forward_hook)

        try:
            state = self.processor.set_image(image)
            output: Dict[str, Any] = self.processor.set_text_prompt(
                state=state,
                prompt=prompt,
            )
        finally:
            hook_handle.remove()

        if debug:
            print("TOP KEYS:", list(output.keys()))
            bo = output.get("backbone_out", None)
            print("backbone_out type:", type(bo))
            if isinstance(bo, dict):
                print("BACKBONE_OUT KEYS:", list(bo.keys()))
            if "boxes" in output and hasattr(output["boxes"], "shape"):
                print("boxes shape:", output["boxes"].shape)

        raw_out = captured.get("raw_out", None)
        if raw_out is None:
            raise RuntimeError(
                "Could not capture raw SAM3 model output via forward hook. "
                "This indicates the processor did not execute the model forward pass as expected."
            )

        # Raw per-query embeddings from the last decoder layer (semantic features).
        query_features = raw_out.get("queries", None)
        if query_features is None:
            raise RuntimeError(
                f"Raw model output did not contain 'queries'. Raw keys: {list(raw_out.keys())}"
            )

        # query_features shape typically: (B, Q, 256) or (Q, 256)
        if hasattr(query_features, "ndim") and query_features.ndim == 3:
            # Use batch 0 (we run single-image inference)
            query_features = query_features[0]

        # Raw predicted boxes per query (normalized xyxy in [0..1]) to match features.
        raw_boxes_xyxy = raw_out.get("pred_boxes_xyxy", None)
        if raw_boxes_xyxy is None:
            raise RuntimeError(
                f"Raw model output did not contain 'pred_boxes_xyxy'. Raw keys: {list(raw_out.keys())}"
            )

        if hasattr(raw_boxes_xyxy, "ndim") and raw_boxes_xyxy.ndim == 3:
            raw_boxes_xyxy = raw_boxes_xyxy[0]  # (Q, 4)

        # Final boxes returned by the processor are in pixel coordinates (XYXY).
        final_boxes = output["boxes"]
        final_scores = output["scores"]
        final_masks = output["masks"]

        # If no detections, return empty aligned features.
        if hasattr(final_boxes, "shape") and final_boxes.shape[0] == 0:
            empty_feats = query_features[:0]  # (0, 256)
            return Sam3Prediction(
                masks=final_masks,
                boxes=final_boxes,
                scores=final_scores,
                features=empty_feats,
            )

        # Convert raw boxes to pixels using original image size from processor output.
        orig_h = int(output["original_height"])
        orig_w = int(output["original_width"])
        raw_boxes_px = self._boxes_xyxy_norm_to_pixels(raw_boxes_xyxy, orig_w, orig_h).to(final_boxes.device)

        # Match final boxes to raw queries by nearest raw box.
        matched_idx = self._match_boxes_to_queries(
            final_boxes_xyxy_px=final_boxes,
            raw_boxes_xyxy_px=raw_boxes_px,
            match_threshold_px=5.0,
        )

        # Aligned semantic embeddings for each final detection box.
        aligned_features = query_features.to(final_boxes.device)[matched_idx]

        # Final sanity check: strict 1:1 alignment
        if aligned_features.shape[0] != final_boxes.shape[0]:
            raise RuntimeError(
                f"Internal alignment error: features N={aligned_features.shape[0]} "
                f"but boxes N={final_boxes.shape[0]}."
            )

        return Sam3Prediction(
            masks=final_masks,
            boxes=final_boxes,
            scores=final_scores,
            features=aligned_features,
        )
