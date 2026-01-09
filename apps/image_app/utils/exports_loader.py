"""
Exports Loader - Read-only consumer for model predictions

Scans and loads model predictions with strict validation against canonical classes.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.data.label_mapping import CANONICAL_CLASSES, CANONICAL_CLASSES_FP, classes_fp
from src.export.model_exporter import load_predictions


def scan_exports(exports_root: str = "artifacts/exports", split: str = "val") -> List[Dict[str, Any]]:
    """
    Scan exports directory for model predictions.

    Args:
        exports_root: Root directory containing model exports
        split: Split name to look for (e.g., "val", "test")

    Returns:
        List of dicts with keys: model_name, model_dir, npz_path, meta_path, is_skipped, skip_reason
    """
    exports_path = Path(exports_root)

    if not exports_path.exists():
        return []

    results = []

    for model_dir in sorted(exports_path.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Filter: ignore directories starting with "_", "test", "debug"
        if model_name.startswith("_") or model_name.startswith("test") or model_name.startswith("debug"):
            continue

        # Check for required files
        npz_path = model_dir / f"{split}.npz"
        meta_path = model_dir / f"{split}_meta.json"

        if not npz_path.exists() or not meta_path.exists():
            results.append({
                "model_name": model_name,
                "model_dir": str(model_dir),
                "npz_path": str(npz_path) if npz_path.exists() else None,
                "meta_path": str(meta_path) if meta_path.exists() else None,
                "is_skipped": True,
                "skip_reason": f"Missing {split}.npz or {split}_meta.json"
            })
            continue

        results.append({
            "model_name": model_name,
            "model_dir": str(model_dir),
            "npz_path": str(npz_path),
            "meta_path": str(meta_path),
            "is_skipped": False,
            "skip_reason": ""
        })

    return results


def load_model_export(model_name: str, npz_path: str) -> Dict[str, Any]:
    """
    Load model predictions with validation.

    Args:
        model_name: Model identifier
        npz_path: Path to .npz file (metadata file found automatically)

    Returns:
        Dict with keys:
            - status: "PASS" or "FAIL"
            - fail_reason: str (empty if PASS)
            - idx: np.ndarray (if loaded)
            - probs: np.ndarray (if loaded)
            - y_true: np.ndarray or None (if loaded)
            - classes: np.ndarray (if loaded)
            - metadata: dict (if loaded)
            - n_samples: int
            - split_signature: str
            - classes_fp: str
    """
    result = {
        "model_name": model_name,
        "status": "FAIL",
        "fail_reason": "",
        "idx": None,
        "probs": None,
        "y_true": None,
        "classes": None,
        "metadata": None,
        "n_samples": 0,
        "split_signature": "",
        "classes_fp": ""
    }

    try:
        # Load predictions using the official loader
        data = load_predictions(
            npz_path=npz_path,
            verify_split_signature=None,  # Don't verify here, collect for comparison
            verify_classes_fp=None,  # Don't verify here, collect for comparison
            require_y_true=False
        )

        result["idx"] = data["idx"]
        result["probs"] = data["probs"]
        result["y_true"] = data.get("y_true")
        result["classes"] = data["classes"]
        result["metadata"] = data["metadata"]
        result["n_samples"] = len(data["idx"])
        result["split_signature"] = data["metadata"].get("split_signature", "")
        result["classes_fp"] = data["metadata"].get("classes_fp", "")

        # Validation 1: split_signature must be non-empty
        if not result["split_signature"]:
            result["fail_reason"] = "split_signature missing or empty in metadata"
            return result

        # Validation 2: metadata must contain classes_fp
        if not result["classes_fp"]:
            result["fail_reason"] = "classes_fp missing in metadata (export contract violation)"
            return result

        # Validation 3: Compute classes_fp from loaded classes and verify
        computed_fp = classes_fp(result["classes"])

        if computed_fp != CANONICAL_CLASSES_FP:
            result["fail_reason"] = f"Computed classes_fp mismatch: {computed_fp} != {CANONICAL_CLASSES_FP}"
            return result

        if result["classes_fp"] != computed_fp:
            result["fail_reason"] = f"Metadata classes_fp ({result['classes_fp']}) != computed ({computed_fp})"
            return result

        # Validation 4: classes must exactly equal CANONICAL_CLASSES
        if not np.array_equal(result["classes"], CANONICAL_CLASSES):
            result["fail_reason"] = f"classes array mismatch with CANONICAL_CLASSES"
            return result

        # Validation 5: probs shape contract
        if result["probs"] is None:
            result["fail_reason"] = "probs is None"
            return result

        if result["probs"].ndim != 2:
            result["fail_reason"] = f"probs shape contract violation: ndim={result['probs'].ndim}, expected 2"
            return result

        if result["probs"].shape[0] != len(result["idx"]):
            result["fail_reason"] = f"probs shape contract violation: shape[0]={result['probs'].shape[0]} != len(idx)={len(result['idx'])}"
            return result

        if result["probs"].shape[1] != 27:
            result["fail_reason"] = f"probs shape contract violation: shape[1]={result['probs'].shape[1]}, expected 27"
            return result

        # All validations passed
        result["status"] = "PASS"

    except Exception as e:
        result["fail_reason"] = f"Load error: {str(e)}"

    return result


def build_global_alignment(models_loaded: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check global alignment across all PASS models.

    Args:
        models_loaded: List of loaded model dicts from load_model_export()

    Returns:
        Dict with keys:
            - all_pass_models: list[str] - names of PASS models
            - sig_ok: bool - all have same split_signature
            - sig_value: str - the common signature (if sig_ok)
            - per_model_sig: dict - map of model_name -> split_signature
            - idx_ok: bool - all have identical idx (same values AND order)
            - idx_mismatch_reason: str - first mismatch detail
    """
    result = {
        "all_pass_models": [],
        "sig_ok": False,
        "sig_value": "",
        "per_model_sig": {},
        "idx_ok": False,
        "idx_mismatch_reason": ""
    }

    # Filter to PASS models only
    pass_models = [m for m in models_loaded if m["status"] == "PASS"]

    if len(pass_models) == 0:
        result["idx_mismatch_reason"] = "No PASS models to compare"
        return result

    result["all_pass_models"] = [m["model_name"] for m in pass_models]

    # Check split_signature alignment
    signatures = {}
    for m in pass_models:
        sig = m["split_signature"]
        signatures[m["model_name"]] = sig

    result["per_model_sig"] = signatures

    unique_sigs = set(signatures.values())
    if len(unique_sigs) == 1:
        result["sig_ok"] = True
        result["sig_value"] = list(unique_sigs)[0]
    else:
        result["sig_ok"] = False
        result["sig_value"] = f"Multiple signatures: {sorted(unique_sigs)}"

    # Check idx alignment (same values AND same order)
    if len(pass_models) < 2:
        # Single model always aligned
        result["idx_ok"] = True
    else:
        reference_idx = pass_models[0]["idx"]
        reference_name = pass_models[0]["model_name"]

        all_match = True
        for m in pass_models[1:]:
            if not np.array_equal(m["idx"], reference_idx):
                all_match = False
                # Check if it's a value mismatch or order mismatch
                if np.array_equal(np.sort(m["idx"]), np.sort(reference_idx)):
                    result["idx_mismatch_reason"] = f"Order mismatch: {m['model_name']} vs {reference_name}"
                else:
                    result["idx_mismatch_reason"] = f"Value mismatch: {m['model_name']} vs {reference_name}"
                break

        result["idx_ok"] = all_match

    return result
