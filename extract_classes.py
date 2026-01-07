#!/usr/bin/env python
"""
Extract canonical classes from Y_train and save as project-level truth.
"""
import argparse
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np


def classes_fp(classes: np.ndarray) -> str:
    """Compute classes fingerprint (SHA256 first 16 chars)."""
    classes_json = json.dumps(classes.tolist())
    return hashlib.sha256(classes_json.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Extract canonical classes from Y_train")
    parser.add_argument("--y-path", type=str, default="data/raw/Y_train_CVw08PX.csv",
                        help="Path to Y_train CSV file")
    parser.add_argument("--out-json", type=str, default="artifacts/canonical_classes.json",
                        help="Output JSON path")
    parser.add_argument("--out-npy", type=str, default="artifacts/canonical_classes.npy",
                        help="Output NPY path")
    parser.add_argument("--expected", type=int, default=27,
                        help="Expected number of classes")

    args = parser.parse_args()

    print("="*80)
    print("EXTRACTING CANONICAL CLASSES")
    print("="*80)

    # Read Y_train
    print(f"\nReading: {args.y_path}")
    y = pd.read_csv(args.y_path, index_col=0)

    # Force int dtype to avoid str/int sorting differences
    y["prdtypecode"] = y["prdtypecode"].astype(int)

    # Get unique classes and sort
    classes = np.array(sorted(y["prdtypecode"].unique()), dtype=np.int64)

    # Validate
    num_classes = len(classes)
    print(f"\nUnique classes found: {num_classes}")
    assert num_classes == args.expected, f"Expected {args.expected} classes, got {num_classes}"
    print(f"[OK] Validation passed: {num_classes} classes")

    # Compute fingerprint
    fp = classes_fp(classes)
    print(f"\nClasses fingerprint (SHA256[:16]): {fp}")

    # Display classes
    print(f"\nCanonical classes (sorted): {classes.tolist()}")

    # Save JSON
    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    json_data = {
        "classes": classes.tolist(),
        "classes_fp": fp,
        "num_classes": num_classes
    }

    with open(out_json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n[OK] Saved JSON: {out_json_path}")

    # Save NPY
    out_npy_path = Path(args.out_npy)
    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, classes)

    print(f"[OK] Saved NPY: {out_npy_path}")

    print("\n" + "="*80)
    print("CANONICAL CLASSES EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nNext step: Use these files as the single source of truth in")
    print(f"          src/data/label_mapping.py")


if __name__ == "__main__":
    main()
