from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from .split_manager import load_splits, split_signature


def _build_image_path(df: pd.DataFrame, img_root: Path) -> pd.Series:
    """
    Build image paths from imageid / productid using a runtime image root.
    """
    file_names = (
        "image_" + df["imageid"].astype(str) +
        "_product_" + df["productid"].astype(str) + ".jpg"
    )
    return file_names.apply(lambda x: img_root / x)


def load_data_colab(
    raw_dir: Path,
    img_root: Optional[Path] = None,
    splitted: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Colab/runtime-only data loader.

    - Does NOT rely on repo-relative data/raw
    - Explicitly uses provided raw_dir (CSV) and img_root (images)
    - Keeps canonical split logic via split_manager

    Args:
        raw_dir: Path to directory containing CSV files
        img_root: Path to image directory (optional, None for text-only models)
        splitted: Whether to return split data
        verbose: Print debug information

    This function is intentionally separate from load_data()
    to avoid breaking local / server workflows.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()

    # Handle optional img_root for text-only models
    if img_root is not None:
        img_root = Path(img_root).expanduser().resolve()

    x_path = raw_dir / "X_train_update.csv"
    y_path = raw_dir / "Y_train_CVw08PX.csv"

    if verbose:
        print("[load_data_colab] raw_dir:", raw_dir)
        print("[load_data_colab] img_root:", img_root)
        print("[load_data_colab] X:", x_path)
        print("[load_data_colab] Y:", y_path)

    if not x_path.exists():
        raise FileNotFoundError(f"Missing X_train_update.csv at {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing Y_train_CVw08PX.csv at {y_path}")

    X = pd.read_csv(x_path)

    # Only build image paths if img_root is provided (for image models)
    if img_root is not None:
        X["image_path"] = _build_image_path(X, img_root)

    y = pd.read_csv(y_path)["prdtypecode"]

    if not splitted:
        return {
            "X": X,
            "y": y,
        }

    # Phase 1: canonical splits (single source of truth)
    splits = load_splits(verbose=verbose)
    sig = split_signature(splits)

    return {
        "X": X,
        "y": y,
        "X_train": X.iloc[splits["train_idx"]],
        "X_val": X.iloc[splits["val_idx"]],
        "X_test": X.iloc[splits["test_idx"]],
        "y_train": y.iloc[splits["train_idx"]],
        "y_val": y.iloc[splits["val_idx"]],
        "y_test": y.iloc[splits["test_idx"]],
        "train_idx": splits["train_idx"],
        "val_idx": splits["val_idx"],
        "test_idx": splits["test_idx"],
        "split_signature": sig,
    }
