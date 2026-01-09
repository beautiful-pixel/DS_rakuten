"""Test enhanced validation in model_exporter (Phase 3 P0 validation)"""
from pathlib import Path
import shutil
import numpy as np
from src.export import export_predictions, load_predictions
from src.data.label_mapping import CANONICAL_CLASSES

# Test output directory (isolated, auto-cleaned)
OUT_DIR = Path("artifacts/exports/_tmp_validation_tests")

def setup():
    """Clean and create test directory"""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def test_idx_wrong_ndim():
    """Test idx with ndim=2 should fail"""
    try:
        idx_bad = np.array([[0,1,2]])  # ndim=2
        probs = np.zeros((3,27))
        export_predictions(OUT_DIR, 'test_idx_ndim2', 'val', idx_bad, 'sig_idx', probs)
        return False  # Should not reach here
    except AssertionError as e:
        if "idx.ndim must be 1" in str(e):
            return True
        print(f"  Unexpected error: {e}")
        return False

def test_probs_wrong_ndim():
    """Test probs with ndim=1 should fail"""
    try:
        idx = np.array([0,1,2])
        probs_bad = np.zeros(27)  # ndim=1
        export_predictions(OUT_DIR, 'test_probs_ndim1', 'val', idx, 'sig_probs', probs_bad)
        return False
    except AssertionError as e:
        if "probs.ndim must be 2" in str(e):
            return True
        print(f"  Unexpected error: {e}")
        return False

def test_ytrue_wrong_ndim():
    """Test y_true with ndim=2 should fail"""
    try:
        idx = np.array([0,1,2])
        probs = np.zeros((3,27))
        y_true_bad = np.array([[10,20,30]])  # ndim=2
        export_predictions(OUT_DIR, 'test_ytrue_ndim2', 'val', idx, 'sig_ytrue', probs, y_true=y_true_bad)
        return False
    except AssertionError as e:
        if "y_true.ndim must be 1" in str(e):
            return True
        print(f"  Unexpected error: {e}")
        return False

def test_classes_order_mismatch():
    """Test wrong classes order should fail"""
    try:
        idx = np.array([0,1,2])
        probs = np.zeros((3,27))
        # Swap two elements to create order mismatch
        wrong_classes = CANONICAL_CLASSES.copy()
        wrong_classes[0], wrong_classes[1] = wrong_classes[1], wrong_classes[0]
        export_predictions(OUT_DIR, 'test_classes_order', 'val', idx, 'sig_classes', probs, classes=wrong_classes)
        return False
    except AssertionError as e:
        if "Classes mismatch" in str(e) or "classes_fp" in str(e):
            return True
        print(f"  Unexpected error: {e}")
        return False

def test_classes_set_mismatch():
    """Test wrong classes set should fail"""
    try:
        idx = np.array([0,1,2])
        probs = np.zeros((3,27))
        # Use completely different classes
        wrong_classes = np.array(list(range(100, 127)))
        export_predictions(OUT_DIR, 'test_classes_set', 'val', idx, 'sig_set', probs, classes=wrong_classes)
        return False
    except AssertionError as e:
        if "Classes mismatch" in str(e) or "classes_fp" in str(e):
            return True
        print(f"  Unexpected error: {e}")
        return False

def test_valid_export_load():
    """Test valid inputs should pass"""
    try:
        idx = np.array([10, 20, 30])
        probs = np.random.rand(3, 27).astype(np.float32)
        y_true = CANONICAL_CLASSES[:3]  # Use valid classes

        result = export_predictions(
            OUT_DIR, 'test_valid', 'val', idx, 'sig_valid_123', probs, y_true=y_true
        )

        # Verify export result
        if result['num_samples'] != 3:
            print(f"  Export failed: wrong num_samples")
            return False

        # Test load
        loaded = load_predictions(result['npz_path'])

        # Verify loaded data
        if not np.array_equal(loaded['idx'], idx):
            print(f"  Load failed: idx mismatch")
            return False

        if not np.array_equal(loaded['classes'], CANONICAL_CLASSES):
            print(f"  Load failed: classes mismatch")
            return False

        if loaded['metadata']['split_signature'] != 'sig_valid_123':
            print(f"  Load failed: split_signature mismatch")
            return False

        return True
    except Exception as e:
        print(f"  Unexpected exception: {e}")
        return False

def main():
    """Run all tests and report summary"""
    setup()

    tests = [
        ("idx wrong ndim", test_idx_wrong_ndim),
        ("probs wrong ndim", test_probs_wrong_ndim),
        ("y_true wrong ndim", test_ytrue_wrong_ndim),
        ("classes order mismatch", test_classes_order_mismatch),
        ("classes set mismatch", test_classes_set_mismatch),
        ("valid export and load", test_valid_export_load),
    ]

    print("="*80)
    print("EXPORTER VALIDATION TESTS (Phase 3 P0)")
    print("="*80)

    results = []
    for name, test_func in tests:
        print(f"\n[Test] {name}")
        passed = test_func()
        status = "PASS" if passed else "FAIL"
        print(f"  Result: {status}")
        results.append((name, passed))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed_count = sum(1 for _, p in results if p)
    failed_count = len(results) - passed_count

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    print(f"\nPassed: {passed_count}/{len(results)}")
    print(f"Failed: {failed_count}/{len(results)}")

    if failed_count > 0:
        print("\n[ERROR] Some tests failed!")
        raise SystemExit(1)
    else:
        print("\n[OK] All tests passed!")

if __name__ == "__main__":
    main()
