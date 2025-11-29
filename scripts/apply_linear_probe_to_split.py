from __future__ import annotations

"""
Apply the trained linear probe to SAM 3 YOLO-style detections.

This script:
- Loads the original SAM 3 predictions for a given split (e.g. "test"),
  stored in YOLO format in:

      data/processed/predictions/sam3_yolo/<split>/*.txt

  Each line is expected to be:
      class_id  cx  cy  w  h  score

- For each prediction, builds the same feature vector used during training:
      [score, cx, cy, w, h, area, aspect_ratio]
  where:
      area = w * h
      aspect_ratio = w / h

- Applies the class-specific logistic regression weights learned by
  `train_linear_probe.py`, stored in:

      data/processed/linear_probe/sam3_linear_probe_weights.npz

- Writes new YOLO files, with the SAME boxes but UPDATED scores, to:

      data/processed/predictions/sam3_linear_probe_yolo/<split>/*.txt

The updated scores can then be evaluated with `eval_yolo.py` in the same
way as the original SAM 3 predictions.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_sam3_yolo_predictions_dir,
    PREDICTIONS_DIR,
)
from src.prompts import CLASS_PROMPTS  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _features_from_yolo(
    cx: float,
    cy: float,
    w: float,
    h: float,
    score: float,
) -> np.ndarray:
    """
    Build the feature vector used by the linear probe from YOLO-normalized
    coordinates and the original SAM 3 score.

    Coordinates (cx, cy, w, h) are assumed to be normalized in [0, 1].
    """
    area = w * h
    aspect_ratio = w / max(h, 1e-6)

    return np.asarray(
        [float(score), cx, cy, w, h, area, aspect_ratio],
        dtype=np.float32,
    )


def apply_linear_probe_to_split(split: str = "test") -> Path:
    """
    Re-score SAM 3 predictions on a given split using the trained linear probe.

    Args:
        split:
            Dataset split to process ('train', 'val', or 'test').

    Returns:
        Path to the directory containing the re-scored YOLO prediction files.
    """
    # Where the original SAM 3 predictions are stored
    in_dir = get_sam3_yolo_predictions_dir(split)

    # Where we will write the re-scored predictions
    out_dir = PREDICTIONS_DIR / "sam3_linear_probe_yolo" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class-wise weights and biases learned by train_linear_probe.py
    weights_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "linear_probe"
        / "sam3_linear_probe_weights.npz"
    )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Linear probe weights not found: {weights_path}. "
            "Please run 'train_linear_probe.py' first."
        )

    weights_data = np.load(weights_path)
    all_weights = weights_data["weights"]  # shape (num_classes, D)
    all_biases = weights_data["biases"]    # shape (num_classes,)

    num_classes = len(CLASS_PROMPTS)
    if all_weights.shape[0] != num_classes:
        raise ValueError(
            f"Mismatch between number of classes in CLASS_PROMPTS ({num_classes}) "
            f"and weights shape ({all_weights.shape[0]})."
        )

    feature_dim = all_weights.shape[1]
    print(f"Applying linear probe on split='{split}'")
    print(f"  Input predictions dir:  {in_dir}")
    print(f"  Output predictions dir: {out_dir}")
    print(f"  Num classes:           {num_classes}")
    print(f"  Feature dimension:     {feature_dim}")

    pred_files: List[Path] = sorted(
        in_dir.glob("*.txt"),
        key=lambda p: int(p.stem),
    )
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {in_dir}")

    for idx, pred_path in enumerate(pred_files, start=1):
        image_id = pred_path.stem
        out_path = out_dir / f"{image_id}.txt"

        new_lines: List[str] = []

        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    # Ignore invalid class ids
                    continue

                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                score_original = float(parts[5]) if len(parts) >= 6 else 1.0

                # Build feature vector and compute new score
                feats = _features_from_yolo(cx, cy, w, h, score_original)
                w_c = all_weights[class_id]  # (D,)
                b_c = float(all_biases[class_id])

                logit = float(feats @ w_c + b_c)
                score_new = float(_sigmoid(np.asarray(logit)))

                # Ensure score is in [0, 1]
                score_new = float(np.clip(score_new, 0.0, 1.0))

                new_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score_new:.6f}"
                new_lines.append(new_line)

        # Write updated predictions for this image
        with out_path.open("w", encoding="utf-8") as f_out:
            for ln in new_lines:
                f_out.write(ln + "\n")

        if idx % 50 == 0 or idx == len(pred_files):
            print(f"  [{idx}/{len(pred_files)}] Processed {pred_path.name} -> {out_path.name}")

    print("Done applying linear probe.")
    return out_dir


def main() -> None:
    """
    Entry point for command-line use.

    By default, this applies the linear probe to the 'test' split.
    """
    split = "test"
    apply_linear_probe_to_split(split=split)


if __name__ == "__main__":
    main()
