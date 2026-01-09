"""
Validation Script for Model Exports

Scans and validates model exports with deterministic reporting.
Provides proof of validation contract compliance.
"""

import sys
import argparse
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from apps.image_app.utils.exports_loader import scan_exports, load_model_export, build_global_alignment
from apps.image_app.utils.metrics import accuracy, macro_f1
from src.data.label_mapping import CANONICAL_CLASSES_FP


def _write_output_file(out_path: str, lines: list):
    """Write captured output to file"""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate model exports")
    parser.add_argument("--split", type=str, default="val", help="Split to validate (default: val)")
    parser.add_argument("--exports-root", type=str, default="artifacts/exports", help="Exports root directory")
    parser.add_argument("--out", type=str, default=None, help="Output file path (optional, writes same report to file)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if alignment fails or any FAIL model exists")
    args = parser.parse_args()

    # Capture output for both stdout and file
    output_lines = []

    def print_and_capture(line=""):
        """Print to stdout and capture for file output"""
        print(line)
        output_lines.append(line)

    print_and_capture("="*80)
    print_and_capture(f"DS_RAKUTEN IMAGE EXPORTS VALIDATION - Split: {args.split}")
    print_and_capture("="*80)
    print_and_capture(f"Exports Root: {args.exports_root}")
    print_and_capture(f"Canonical Classes FP: {CANONICAL_CLASSES_FP}")
    print_and_capture()

    # Scan exports
    scanned = scan_exports(exports_root=args.exports_root, split=args.split)

    if len(scanned) == 0:
        print_and_capture(f"ERROR: No model directories found in {args.exports_root}")
        print_and_capture(f"Expected structure: {args.exports_root}/<model_name>/{args.split}.npz + {args.split}_meta.json")

        # Write output if requested
        if args.out:
            _write_output_file(args.out, output_lines)

        return 1

    print_and_capture(f"Found {len(scanned)} model directories")
    print_and_capture()

    # Load all models
    models_loaded = []
    for scan_result in scanned:
        if scan_result["is_skipped"]:
            models_loaded.append({
                "model_name": scan_result["model_name"],
                "status": "SKIPPED",
                "fail_reason": scan_result["skip_reason"],
                "idx": None,
                "probs": None,
                "y_true": None,
                "classes": None,
                "metadata": None,
                "n_samples": 0,
                "split_signature": "",
                "classes_fp": ""
            })
        else:
            result = load_model_export(
                model_name=scan_result["model_name"],
                npz_path=scan_result["npz_path"]
            )
            models_loaded.append(result)

    # Per-model validation report
    print_and_capture("-"*80)
    print_and_capture("PER-MODEL VALIDATION")
    print_and_capture("-"*80)

    for m in sorted(models_loaded, key=lambda x: x["model_name"]):
        print_and_capture(f"\nModel: {m['model_name']}")
        print_and_capture(f"  Status:          {m['status']}")
        print_and_capture(f"  Fail Reason:     {m['fail_reason'] if m['fail_reason'] else 'N/A'}")
        print_and_capture(f"  Samples:         {m['n_samples']}")
        print_and_capture(f"  Split Signature: {m['split_signature'] if m['split_signature'] else 'N/A'}")
        print_and_capture(f"  Classes FP:      {m['classes_fp'] if m['classes_fp'] else 'N/A'}")

    # Global alignment check
    alignment = build_global_alignment(models_loaded)

    print_and_capture()
    print_and_capture("-"*80)
    print_and_capture("GLOBAL ALIGNMENT")
    print_and_capture("-"*80)
    print_and_capture(f"PASS Models:   {len(alignment['all_pass_models'])} - {alignment['all_pass_models']}")
    print_and_capture(f"Sig OK:        {alignment['sig_ok']}")
    print_and_capture(f"Sig Value:     {alignment['sig_value']}")
    print_and_capture(f"Idx OK:        {alignment['idx_ok']}")
    print_and_capture(f"Idx Mismatch:  {alignment['idx_mismatch_reason'] if alignment['idx_mismatch_reason'] else 'N/A'}")

    if alignment["per_model_sig"]:
        print_and_capture("\nPer-Model Signatures:")
        for model_name, sig in sorted(alignment["per_model_sig"].items()):
            print_and_capture(f"  {model_name}: {sig}")

    # Metrics (if y_true exists for PASS models)
    pass_models = [m for m in models_loaded if m["status"] == "PASS"]
    has_y_true = any(m["y_true"] is not None for m in pass_models)

    if has_y_true and len(pass_models) > 0:
        print_and_capture()
        print_and_capture("-"*80)
        print_and_capture("METRICS (PASS MODELS WITH Y_TRUE)")
        print_and_capture("-"*80)

        for m in sorted(pass_models, key=lambda x: x["model_name"]):
            if m["y_true"] is not None:
                acc = accuracy(m["y_true"], m["probs"])
                f1 = macro_f1(m["y_true"], m["probs"], num_classes=27)
                print_and_capture(f"\n{m['model_name']}:")
                print_and_capture(f"  Accuracy:  {acc:.4f}")
                print_and_capture(f"  Macro F1:  {f1:.4f}")
                print_and_capture(f"  Samples:   {m['n_samples']:,}")
    elif len(pass_models) > 0:
        print_and_capture()
        print_and_capture("-"*80)
        print_and_capture("METRICS: N/A (no y_true available for PASS models)")
        print_and_capture("-"*80)

    # Summary
    print_and_capture()
    print_and_capture("="*80)
    print_and_capture("SUMMARY")
    print_and_capture("="*80)
    num_pass = len([m for m in models_loaded if m["status"] == "PASS"])
    num_fail = len([m for m in models_loaded if m["status"] == "FAIL"])
    num_skip = len([m for m in models_loaded if m["status"] == "SKIPPED"])

    print_and_capture(f"Total Models:  {len(models_loaded)}")
    print_and_capture(f"PASS:          {num_pass}")
    print_and_capture(f"FAIL:          {num_fail}")
    print_and_capture(f"SKIPPED:       {num_skip}")
    print_and_capture(f"Alignment OK:  {alignment['sig_ok'] and alignment['idx_ok']}")
    print_and_capture(f"Fusion Ready:  {num_pass >= 2 and alignment['sig_ok'] and alignment['idx_ok']}")
    print_and_capture("="*80)

    # Write output to file if requested
    if args.out:
        _write_output_file(args.out, output_lines)

    # Determine exit code based on --strict flag
    alignment_ok = alignment['sig_ok'] and alignment['idx_ok']
    has_failures = num_fail > 0

    if args.strict and (not alignment_ok or has_failures):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
