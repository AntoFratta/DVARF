from __future__ import annotations

"""
Train a simple linear probe on top of SAM 3 detection features.

This script expects a dataset created by `build_linear_probe_dataset.py`,
stored as:

    data/processed/linear_probe/sam3_linear_probe_<split>.npz

The .npz file must contain:
    - features: float32 array of shape (N, D)
    - targets: int64 array of shape (N,) with values {0, 1}
    - class_ids: int64 array of shape (N,) with values in [0, num_classes-1]

For each class c, we select all samples with class_ids == c and train a
separate logistic regression model:

    p(y=1 | x, c) = sigmoid( w_c^T x + b_c )

The resulting weight matrix and biases are saved to:

    data/processed/linear_probe/sam3_linear_probe_weights.npz

This file can later be used to re-score SAM 3 detections on any split.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.prompts import CLASS_PROMPTS  # noqa: E402


@dataclass
class LogisticRegressionWeights:
    """Weights and bias for a binary logistic regression model."""
    weights: np.ndarray  # shape (D,)
    bias: float          # scalar


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 500,
    learning_rate: float = 0.1,
    l2_weight: float = 1e-4,
) -> LogisticRegressionWeights:
    """
    Train a binary logistic regression model with L2 regularization.

    Args:
        x:
            Input features of shape (N, D).
        y:
            Binary targets of shape (N,), values in {0, 1}.
        num_epochs:
            Number of gradient descent iterations.
        learning_rate:
            Step size for gradient updates.
        l2_weight:
            L2 regularization coefficient (lambda).

    Returns:
        LogisticRegressionWeights with learned parameters.
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    n_samples, n_features = x.shape
    w = np.zeros(n_features, dtype=np.float32)
    b = 0.0

    for epoch in range(num_epochs):
        # Forward pass
        logits = x @ w + b
        probs = _sigmoid(logits)

        # Compute gradients
        error = probs - y  # shape (N,)
        grad_w = (x.T @ error) / float(n_samples) + l2_weight * w
        grad_b = float(np.mean(error))

        # Parameter update
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return LogisticRegressionWeights(weights=w, bias=float(b))


def train_linear_probe(
    split: str = "train",
    num_epochs: int = 500,
    learning_rate: float = 0.1,
    l2_weight: float = 1e-4,
) -> Path:
    """
    Train one logistic regression model per class on the specified split.

    Args:
        split:
            Dataset split used for training ('train' by default).
        num_epochs:
            Number of gradient descent iterations for each class model.
        learning_rate:
            Learning rate for gradient descent.
        l2_weight:
            L2 regularization strength.

    Returns:
        Path to the saved .npz file with all class-wise weights.
    """
    in_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "linear_probe"
        / f"sam3_linear_probe_{split}.npz"
    )

    if not in_path.exists():
        raise FileNotFoundError(
            f"Linear probe dataset not found: {in_path}. "
            f"Please run 'build_linear_probe_dataset.py' first for split='{split}'."
        )

    data = np.load(in_path)
    features = data["features"]          # (N, D)
    targets = data["targets"]            # (N,)
    class_ids = data["class_ids"]        # (N,)

    num_samples, feature_dim = features.shape
    num_classes = len(CLASS_PROMPTS)

    print(f"Training linear probe on split='{split}'")
    print(f"  Input file:      {in_path}")
    print(f"  Num samples:     {num_samples}")
    print(f"  Feature dim:     {feature_dim}")
    print(f"  Num classes:     {num_classes}")
    print(f"  Num epochs:      {num_epochs}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  L2 weight:       {l2_weight}")

    # Storage for class-wise weights and biases
    all_weights = np.zeros((num_classes, feature_dim), dtype=np.float32)
    all_biases = np.zeros((num_classes,), dtype=np.float32)

    # For reporting purposes
    class_stats: Dict[int, Tuple[int, int]] = {}  # class_id -> (num_pos, num_neg)

    for class_id in range(num_classes):
        mask = class_ids == class_id
        x_c = features[mask]
        y_c = targets[mask]

        num_c = x_c.shape[0]
        num_pos = int(np.sum(y_c))
        num_neg = int(num_c - num_pos)
        class_stats[class_id] = (num_pos, num_neg)

        print(
            f"\nClass {class_id} ({CLASS_PROMPTS[class_id]}): "
            f"{num_c} samples -> {num_pos} pos, {num_neg} neg"
        )

        if num_pos == 0 or num_neg == 0:
            print(
                "  WARNING: class has only one label type (all pos or all neg). "
                "Skipping training and leaving weights at zero."
            )
            continue

        model = train_logistic_regression(
            x_c,
            y_c,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
        )

        all_weights[class_id] = model.weights
        all_biases[class_id] = model.bias

        # Simple training accuracy at threshold 0.5 (for curiosity)
        logits = x_c @ model.weights + model.bias
        preds = (logits >= 0.0).astype(np.int64)
        acc = float(np.mean(preds == y_c))
        print(f"  Training accuracy (thr=0.5): {acc:.4f}")

    out_dir = PROJECT_ROOT / "data" / "processed" / "linear_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "sam3_linear_probe_weights.npz"
    np.savez_compressed(
        out_path,
        weights=all_weights,
        biases=all_biases,
    )

    print(f"\nSaved linear-probe weights to: {out_path}")
    for class_id, (num_pos, num_neg) in class_stats.items():
        total = max(num_pos + num_neg, 1)
        ratio = num_pos / float(total)
        print(
            f"  Class {class_id}: {num_pos} pos, {num_neg} neg "
            f"(pos ratio = {ratio:.3f})"
        )

    return out_path


def main() -> None:
    """
    Entry point for command-line use.

    By default, this trains the linear probe on the 'train' split
    using a fixed set of hyperparameters.
    """
    split = "train"
    num_epochs = 500
    learning_rate = 0.1
    l2_weight = 1e-4

    train_linear_probe(
        split=split,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
    )


if __name__ == "__main__":
    main()
