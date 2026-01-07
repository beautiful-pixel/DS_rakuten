"""
Unified split loading entry point. Ensures all notebooks use the same canonical splits.

Core principles:
1. Load from data/splits/*.txt if files exist (single source of truth)
2. Only call generate_splits() and save if files do not exist
3. Provide alignment checks and signature mechanism for export and fusion assertions
"""

import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, Tuple

from .splits import generate_splits


# Define SPLITS_DIR locally (project convention: data/splits relative to project root)
MODULE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = (MODULE_DIR / "../../data/splits").resolve()


def load_splits(verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Load canonical splits (train/val/test indices).

    Priority:
    1. If data/splits/*.txt exist -> load directly (single source of truth)
    2. Otherwise -> call generate_splits() and save to files

    Args:
        verbose: If True, print loading/generation messages

    Returns:
        dict: {
            "train_idx": np.ndarray,
            "val_idx": np.ndarray,
            "test_idx": np.ndarray
        }
    """
    train_file = SPLITS_DIR / "train_idx.txt"
    val_file = SPLITS_DIR / "val_idx.txt"
    test_file = SPLITS_DIR / "test_idx.txt"

    # Check if all files exist
    if train_file.exists() and val_file.exists() and test_file.exists():
        if verbose:
            print(f"[split_manager] Loading canonical splits from {SPLITS_DIR}")
        train_idx = np.loadtxt(train_file, dtype=int)
        val_idx = np.loadtxt(val_file, dtype=int)
        test_idx = np.loadtxt(test_file, dtype=int)
    else:
        if verbose:
            print(f"[split_manager] Splits files not found, generating and saving to {SPLITS_DIR}")
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            splits = generate_splits()
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate splits using generate_splits(). "
                f"This function depends on DATA_DIR configuration and raw data files. "
                f"Preferred path: provide pre-generated data/splits/*.txt files. "
                f"Original error: {e}"
            ) from e

        train_idx = splits["train_idx"]
        val_idx = splits["val_idx"]
        test_idx = splits["test_idx"]

        # Save to files
        np.savetxt(train_file, train_idx, fmt="%d")
        np.savetxt(val_file, val_idx, fmt="%d")
        np.savetxt(test_file, test_idx, fmt="%d")
        if verbose:
            print(f"[split_manager] Splits saved to {SPLITS_DIR}")

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }


def print_split_summary(splits: Dict[str, np.ndarray]) -> None:
    """
    Print split summary including sizes and overlap checks.

    Args:
        splits: Dictionary returned by load_splits()
    """
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    print("\n" + "="*60)
    print("CANONICAL SPLIT SUMMARY")
    print("="*60)
    print(f"Train size: {len(train_idx):6d}")
    print(f"Val size:   {len(val_idx):6d}")
    print(f"Test size:  {len(test_idx):6d}")
    print(f"Total:      {len(train_idx) + len(val_idx) + len(test_idx):6d}")
    print("-"*60)

    # Overlap checks (must all be 0)
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)

    print("Overlap checks (must all be 0):")
    print(f"  train & val:  {overlap_train_val}")
    print(f"  train & test: {overlap_train_test}")
    print(f"  val & test:   {overlap_val_test}")

    if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
        print("OK: All overlaps are 0 - splits are valid!")
    else:
        print("WARNING: Non-zero overlap detected!")

    print("="*60 + "\n")


def split_signature(splits: Dict[str, np.ndarray]) -> str:
    """
    Calculate SHA256 signature of splits for export and fusion alignment assertions.

    Hashes the sorted concatenation of all three index arrays to ensure
    identical content yields identical signature regardless of loading order.

    Args:
        splits: Dictionary returned by load_splits()

    Returns:
        str: 16-character hexadecimal signature (first 8 bytes of SHA256)
    """
    train_idx = np.sort(splits["train_idx"])
    val_idx = np.sort(splits["val_idx"])
    test_idx = np.sort(splits["test_idx"])

    # Concatenate byte representation of all three arrays
    combined = np.concatenate([train_idx, val_idx, test_idx])
    content = combined.tobytes()

    # Calculate SHA256, take first 8 bytes as short signature
    hash_obj = hashlib.sha256(content)
    signature = hash_obj.hexdigest()[:16]

    return signature


def get_split_info(verbose: bool = True) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Convenience function: load splits + calculate signature + print summary.

    Args:
        verbose: If True, print loading messages

    Returns:
        tuple: (splits dict, signature string)
    """
    splits = load_splits(verbose=verbose)
    print_split_summary(splits)
    sig = split_signature(splits)
    print(f"Split signature: {sig}\n")
    return splits, sig
