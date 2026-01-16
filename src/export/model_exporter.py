"""
Model Exporter - Unified Export Contract (Phase 3)

Provides standardized export and loading of model predictions with validation.
All exports include: idx, split_signature, classes, probs, metadata.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import numpy as np

from src.data.label_mapping import (
    CANONICAL_CLASSES,
    CANONICAL_CLASSES_FP,
    classes_fp,
    verify_canonical_classes
)


def export_predictions(
    out_dir: Union[str, Path],
    model_name: str,
    split_name: str,
    idx: np.ndarray,
    split_signature: str,
    probs: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export model predictions with full validation metadata.
    Supports both probability exports (legacy) and raw logits exports (recommended for fusion).

    Args:
        out_dir: Output directory
        model_name: Model identifier (e.g., "swin_v2_logits", "convnext_logits")
        split_name: Split identifier (e.g., "val", "test")
        idx: Sample indices (original row numbers)
        split_signature: Split signature for alignment verification
        probs: Prediction probabilities of shape (N, 27) - optional, for legacy compatibility
        logits: Raw model logits of shape (N, 27) - recommended for model fusion
        classes: Classes array (defaults to CANONICAL_CLASSES, must match)
        y_true: Ground truth labels (optional)
        extra_meta: Additional metadata (optional)

    Returns:
        Dict with keys: npz_path, meta_json_path, classes_fp, split_signature, num_samples

    Raises:
        AssertionError: If validation fails (single-line error)

    Notes:
        - At least one of probs or logits must be provided
        - If logits is provided, output_type will be set to "logits"
        - If only probs is provided, output_type will be set to "probs"
        - The .npz file will contain separate fields for probs and logits (whichever is provided)
    """
    # Validate at least one of probs or logits is provided
    if probs is None and logits is None:
        raise AssertionError("At least one of probs or logits must be provided")

    # Use canonical classes if not provided
    if classes is None:
        classes = CANONICAL_CLASSES

    # Validate split_signature
    if not isinstance(split_signature, str) or split_signature.strip() == "":
        raise AssertionError("split_signature is required and must be a non-empty string")

    # Directory structure: out_dir/model_name/
    model_dir = Path(out_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Force dtypes
    idx = np.asarray(idx, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float32)
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float32)
    if y_true is not None:
        y_true = np.asarray(y_true, dtype=np.int64)

    # Validate array dimensions (defensive checks after dtype conversion)
    if idx.ndim != 1:
        raise AssertionError(f"idx.ndim must be 1, got {idx.ndim}")

    if probs is not None and probs.ndim != 2:
        raise AssertionError(f"probs.ndim must be 2, got {probs.ndim}")

    if logits is not None and logits.ndim != 2:
        raise AssertionError(f"logits.ndim must be 2, got {logits.ndim}")

    if y_true is not None and y_true.ndim != 1:
        raise AssertionError(f"y_true.ndim must be 1, got {y_true.ndim}")

    # Validate classes against canonical
    verify_canonical_classes(classes, raise_on_error=True)

    # Compute and validate classes_fp (explicit check for alignment)
    fp_classes = classes_fp(classes)
    if fp_classes != CANONICAL_CLASSES_FP:
        raise AssertionError(f"classes_fp={fp_classes} != CANONICAL_CLASSES_FP={CANONICAL_CLASSES_FP}")

    # Determine primary output array (logits preferred) for shape validation
    primary_output = logits if logits is not None else probs
    n_samples = len(idx)

    # Validate shapes
    if primary_output.shape[0] != n_samples:
        raise AssertionError(f"primary_output.shape[0]={probs.shape[0] if probs is not None else logits.shape[0]} != len(idx)={n_samples}")

    if probs is not None and probs.shape[0] != n_samples:
        raise AssertionError(f"probs.shape[0]={probs.shape[0]} != len(idx)={n_samples}")

    if probs is not None and probs.shape[1] != 27:
        raise AssertionError(f"probs.shape[1]={probs.shape[1]} != 27 (expected num_classes)")

    if logits is not None and logits.shape[0] != n_samples:
        raise AssertionError(f"logits.shape[0]={logits.shape[0]} != len(idx)={n_samples}")

    if logits is not None and logits.shape[1] != 27:
        raise AssertionError(f"logits.shape[1]={logits.shape[1]} != 27 (expected num_classes)")

    if y_true is not None and len(y_true) != n_samples:
        raise AssertionError(f"len(y_true)={len(y_true)} != len(idx)={n_samples}")

    # Paths
    npz_path = model_dir / f"{split_name}.npz"
    json_path = model_dir / f"{split_name}_meta.json"

    # Determine output type (logits preferred)
    output_type = "logits" if logits is not None else "probs"

    # Save .npz (data arrays) - store logits and probs in separate fields
    npz_data = {
        "idx": idx,
        "classes": classes,
    }
    if logits is not None:
        npz_data["logits"] = logits
    if probs is not None:
        npz_data["probs"] = probs
    if y_true is not None:
        npz_data["y_true"] = y_true

    np.savez_compressed(npz_path, **npz_data)

    # Save metadata JSON
    meta = {
        "model_name": model_name,
        "split_name": split_name,
        "split_signature": split_signature,
        "classes_fp": fp_classes,
        "num_classes": len(classes),
        "num_samples": n_samples,
        "has_y_true": y_true is not None,
        "has_logits": logits is not None,
        "has_probs": probs is not None,
        "output_type": output_type,
        "created_at": datetime.now().isoformat(),
    }

    # Add shape info for available arrays
    if logits is not None:
        meta["logits_shape"] = list(logits.shape)
        meta["logits_dtype"] = str(logits.dtype)
    if probs is not None:
        meta["probs_shape"] = list(probs.shape)
        meta["probs_dtype"] = str(probs.dtype)

    # Merge extra metadata
    if extra_meta:
        meta["extra"] = extra_meta

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Compact output (single line)
    msg = (
        f"[OK] Exported model={model_name} split={split_name} npz={npz_path} "
        f"sig={split_signature} fp={fp_classes} n={n_samples}"
    )
    print(msg)

    return {
        "npz_path": str(npz_path),
        "meta_json_path": str(json_path),
        "classes_fp": fp_classes,
        "split_signature": split_signature,
        "num_samples": n_samples,
    }


def load_predictions(
    npz_path: Union[str, Path],
    verify_split_signature: Optional[str] = None,
    verify_classes_fp: Optional[str] = None,
    require_y_true: bool = False
) -> Dict[str, Any]:
    """
    Load model predictions with validation.
    Supports both legacy probability-only exports and new logits exports.

    Args:
        npz_path: Path to .npz file
        verify_split_signature: Expected split signature (raises if mismatch)
        verify_classes_fp: Expected classes fingerprint (raises if mismatch)
        require_y_true: If True, raise if y_true not present

    Returns:
        Dictionary with keys: idx, classes, y_true (if present), metadata
        Optional keys: logits (if present), probs (if present)
        Convenience key: scores (alias to logits if present, else probs)

    Raises:
        FileNotFoundError: If files not found (single-line)
        AssertionError: If validation fails (single-line)

    Notes:
        - Backward compatible: automatically reads probs if logits not present
        - Use result["scores"] for generic access to primary output
        - Check metadata["output_type"] to determine which field is primary
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {npz_path}")

    # Load .npz
    data = np.load(npz_path, allow_pickle=False)

    result = {
        "idx": data["idx"],
        "classes": data["classes"],
    }

    # Load logits if present (preferred)
    if "logits" in data:
        result["logits"] = data["logits"]

    # Load probs if present (legacy compatibility)
    if "probs" in data:
        result["probs"] = data["probs"]

    # Provide scores alias (logits preferred, fallback to probs)
    if "logits" in result:
        result["scores"] = result["logits"]
    elif "probs" in result:
        result["scores"] = result["probs"]
    else:
        raise AssertionError(f"Neither logits nor probs found in {npz_path}")

    if "y_true" in data:
        result["y_true"] = data["y_true"]
    elif require_y_true:
        raise AssertionError(f"y_true required but not found in {npz_path}")

    # Load metadata JSON (same directory, split_name_meta.json)
    json_path = npz_path.with_name(npz_path.stem + "_meta.json")

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, "r") as f:
        result["metadata"] = json.load(f)

    # Validation
    if verify_split_signature is not None:
        actual_sig = result["metadata"].get("split_signature", "")
        if actual_sig != verify_split_signature:
            msg = f"Split signature mismatch: expected={verify_split_signature} got={actual_sig} file={npz_path}"
            raise AssertionError(msg)

    if verify_classes_fp is not None:
        actual_fp = result["metadata"].get("classes_fp", "")
        if actual_fp != verify_classes_fp:
            msg = f"Classes fingerprint mismatch: expected={verify_classes_fp} got={actual_fp} file={npz_path}"
            raise AssertionError(msg)

    # Verify classes against canonical
    verify_canonical_classes(result["classes"], raise_on_error=True)

    return result
