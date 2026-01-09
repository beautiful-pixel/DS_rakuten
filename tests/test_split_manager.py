"""
Quick test script for Phase 0: split_manager functionality.

Run from repository root: python test_split_manager.py
"""

import sys
from pathlib import Path

# Add repo root to path so that "import src..." works
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.split_manager import get_split_info

if __name__ == "__main__":
    print("Testing split_manager.py functionality...\n")
    splits, signature = get_split_info(verbose=True)

    print(f"\nVerification complete.")
    print(f"Signature: {signature}")
    print(f"train_idx shape: {splits['train_idx'].shape}")
    print(f"val_idx shape: {splits['val_idx'].shape}")
    print(f"test_idx shape: {splits['test_idx'].shape}")
