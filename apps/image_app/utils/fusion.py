"""
Fusion Utilities - Model ensemble methods

Provides mean fusion and optional export of fused predictions.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.export.model_exporter import export_predictions
from src.data.label_mapping import CANONICAL_CLASSES


def mean_fusion(probs_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute mean fusion of probability predictions.

    Args:
        probs_list: List of probability arrays, each of shape (N, num_classes)

    Returns:
        Mean fused probabilities of shape (N, num_classes)
    """
    if len(probs_list) == 0:
        raise ValueError("probs_list cannot be empty")

    # Stack and compute mean along model axis
    stacked = np.stack(probs_list, axis=0)  # Shape: (num_models, N, num_classes)
    fused = np.mean(stacked, axis=0)  # Shape: (N, num_classes)

    return fused


def export_fusion(
    out_dir: str,
    split_name: str,
    idx: np.ndarray,
    split_signature: str,
    probs: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export fused predictions using the official export contract.

    Args:
        out_dir: Output directory (typically "artifacts/exports")
        split_name: Split name (e.g., "val", "test")
        idx: Sample indices
        split_signature: Split signature for alignment
        probs: Fused probabilities
        y_true: Ground truth labels (optional)
        extra_meta: Additional metadata (optional)

    Returns:
        Export result dict from export_predictions()
    """
    # Always write to fusion_mean subdirectory
    model_name = "fusion_mean"

    # Merge extra metadata with fusion info
    fusion_meta = {
        "fusion_method": "mean",
        "is_fusion": True
    }

    if extra_meta:
        fusion_meta.update(extra_meta)

    return export_predictions(
        out_dir=out_dir,
        model_name=model_name,
        split_name=split_name,
        idx=idx,
        split_signature=split_signature,
        probs=probs,
        classes=CANONICAL_CLASSES,
        y_true=y_true,
        extra_meta=fusion_meta
    )
