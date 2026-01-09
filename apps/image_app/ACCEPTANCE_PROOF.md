# B4 Streamlit MVP - Acceptance Proof

## Purpose

This document provides instructions for validating the B4 Streamlit MVP implementation and verifying compliance with all requirements.

## Prerequisites

Before running validation, ensure model exports exist:

1. Run export cells in Phase 3 notebooks:
   - `notebooks/image_03_resNet50.ipynb` → `artifacts/exports/resnet50/val.npz`
   - `notebooks/image_04_ViT.ipynb` → `artifacts/exports/vit_base/val.npz`
   - `notebooks/image_05_swin.ipynb` → `artifacts/exports/swin_v2/val.npz`
   - `notebooks/image_06_convnext.ipynb` → `artifacts/exports/convnext/val.npz`

2. Verify canonical classes file exists:
   - `artifacts/canonical_classes.json` (created by `extract_classes.py`)

## Validation Commands

### Basic Validation (stdout only)
```bash
python apps/image_app/scripts/validate_exports.py --split val
```

### Write Report to File
```bash
python apps/image_app/scripts/validate_exports.py --split val --out artifacts/exports/_validation/val_report.txt
```

### Strict Mode (CI/CD friendly)
```bash
python apps/image_app/scripts/validate_exports.py --split val --strict
```

**Strict mode behavior**:
- Exit code 0: All models PASS, alignment OK
- Exit code 1: Any FAIL models OR alignment failures (sig_ok=False or idx_ok=False)

## Required Output Fields

The validation script MUST display the following fields to satisfy acceptance criteria:

### Per-Model Section
For each model discovered:
- `model_name`: Model identifier
- `Status`: PASS / FAIL / SKIPPED
- `Fail Reason`: Explicit reason if not PASS (N/A otherwise)
- `Samples`: Number of samples (integer)
- `Split Signature`: Hash identifying the split (string)
- `Classes FP`: Fingerprint of classes array (MUST equal `cdfa70b13f7390e6` for PASS)

### Global Alignment Section
- `PASS Models`: Count and list of PASS model names
- `Sig OK`: Boolean (True if all PASS models share same split_signature)
- `Sig Value`: The common signature or description of mismatch
- `Idx OK`: Boolean (True if all PASS models have identical idx)
- `Idx Mismatch`: Reason if idx_ok=False (N/A otherwise)
- `Per-Model Signatures`: Map of model_name → split_signature

### Metrics Section (if y_true available)
For each PASS model with y_true:
- `Accuracy`: Float (4 decimals)
- `Macro F1`: Float (4 decimals)
- `Samples`: Integer with thousands separator

### Summary Section
- `Total Models`: Count
- `PASS`: Count
- `FAIL`: Count
- `SKIPPED`: Count
- `Alignment OK`: Boolean (sig_ok AND idx_ok)
- `Fusion Ready`: Boolean (≥2 PASS models AND alignment OK)

## Output Format Reference (FORMAT ONLY - NOT REAL DATA)

```
================================================================================
DS_RAKUTEN IMAGE EXPORTS VALIDATION - Split: val
================================================================================
Exports Root: artifacts/exports
Canonical Classes FP: cdfa70b13f7390e6

Found <N> model directories

--------------------------------------------------------------------------------
PER-MODEL VALIDATION
--------------------------------------------------------------------------------

Model: <model_name>
  Status:          PASS | FAIL | SKIPPED
  Fail Reason:     N/A | <specific error>
  Samples:         <integer>
  Split Signature: <hash_string>
  Classes FP:      cdfa70b13f7390e6 | <other_value>

[... repeat for each model ...]

--------------------------------------------------------------------------------
GLOBAL ALIGNMENT
--------------------------------------------------------------------------------
PASS Models:   <count> - [<list_of_names>]
Sig OK:        True | False
Sig Value:     <common_signature> | Multiple signatures: [<list>]
Idx OK:        True | False
Idx Mismatch:  N/A | <specific mismatch reason>

Per-Model Signatures:
  <model_name>: <signature>
  [... repeat for each PASS model ...]

--------------------------------------------------------------------------------
METRICS (PASS MODELS WITH Y_TRUE)
--------------------------------------------------------------------------------

<model_name>:
  Accuracy:  <0.xxxx>
  Macro F1:  <0.xxxx>
  Samples:   <integer_with_commas>

[... repeat for each PASS model with y_true ...]

================================================================================
SUMMARY
================================================================================
Total Models:  <count>
PASS:          <count>
FAIL:          <count>
SKIPPED:       <count>
Alignment OK:  True | False
Fusion Ready:  True | False
================================================================================
```

**Note**: This is a FORMAT REFERENCE only. Actual values depend on your exports and training results.

## Acceptance Criteria Checklist

Verify the following by running the validation script:

- [ ] At least 2 models listed with split_signature and classes_fp
- [ ] All PASS models have `Classes FP: cdfa70b13f7390e6`
- [ ] Global `Sig OK` field displayed with boolean value
- [ ] Global `Idx OK` field displayed with boolean value
- [ ] If y_true exists: Accuracy and Macro F1 computed for PASS models
- [ ] Summary shows `Fusion Ready: True` when alignment passes

## Streamlit App Verification

### Command
```bash
streamlit run apps/image_app/app.py
```

### UI Sections to Verify

**1. Validation Overview Table**
- Shows all discovered models
- Columns: Model, Status, Fail Reason, Samples, Split Signature, Classes FP
- Filter checkbox: "Show only PASS models"

**2. Global Alignment Panel**
- Two sections: Split Signature and Index Alignment
- Green checkmarks when aligned, red X when misaligned
- Expandable per-model details if misaligned

**3. Metrics Section (if y_true available)**
- Metrics table with columns: model_name (index), accuracy, macro_f1, n_samples
- Bar chart for Accuracy
- Bar chart for Macro F1
- All charts source from single DataFrame (df_metrics)

**4. Mean Fusion Section**
- Checkbox: "Enable mean fusion" (disabled if alignment fails)
- When enabled: Computes fusion and shows metrics
- Button: "Export fusion to artifacts/exports/fusion_mean/"
- Updated charts include fusion row

### Alignment Failure Behavior

When alignment fails (sig_ok=False OR idx_ok=False):
- Global Alignment panel shows red X with specific reason
- Mean fusion checkbox is DISABLED (greyed out)
- Warning banner: "Mean fusion is BLOCKED due to alignment failures"
- Metrics section still shows individual model results
- Fusion section hidden or shows blocking message

### Table-Driven Charts Contract

Verify in code (`apps/image_app/app.py`):

```python
# Single DataFrame as ONLY source
df_metrics = pd.DataFrame(metrics_data).set_index("model_name")

# Fixed columns
assert "accuracy" in df_metrics.columns
assert "macro_f1" in df_metrics.columns
assert "n_samples" in df_metrics.columns

# Charts use ONLY df_metrics
st.bar_chart(df_metrics[["accuracy"]])
st.bar_chart(df_metrics[["macro_f1"]])
```

**Requirements**:
- Only PASS models appear in df_metrics
- FAIL models appear only in validation overview table
- Fusion row appended to same df_metrics when enabled
- No direct data queries in chart code

## Strict Mode Use Cases

### CI/CD Integration
```bash
# In CI pipeline
python apps/image_app/scripts/validate_exports.py --split val --strict --out artifacts/exports/_validation/ci_report.txt

if [ $? -eq 0 ]; then
  echo "Validation passed - all models aligned"
else
  echo "Validation failed - check report for details"
  exit 1
fi
```

### Pre-deployment Check
```bash
# Validate before deploying Streamlit app
python apps/image_app/scripts/validate_exports.py --split val --strict

if [ $? -eq 0 ]; then
  streamlit run apps/image_app/app.py
else
  echo "Fix alignment issues before launching app"
  exit 1
fi
```

## Validation Contract

Each model is validated through 5 strict checks:

1. **Split Signature**: Must be non-empty string
2. **Metadata Classes FP**: Must exist in metadata
3. **Computed Classes FP**: Must equal CANONICAL_CLASSES_FP (`cdfa70b13f7390e6`)
4. **Classes Array**: Must exactly match CANONICAL_CLASSES (element-wise)
5. **Probs Shape Contract**:
   - probs not None
   - probs.ndim == 2
   - probs.shape[0] == len(idx)
   - probs.shape[1] == 27

Any failure → `Status: FAIL` with explicit `Fail Reason`.

Global alignment checks (blocks fusion):
- **sig_ok**: All PASS models share same split_signature
- **idx_ok**: All PASS models have identical idx (same values in same order)

## Troubleshooting

### "No model directories found"
- Run export cells in notebooks first
- Check `artifacts/exports/` exists and contains subdirectories

### "classes_fp mismatch"
- Model trained before Phase 2 canonical classes
- Re-run notebook export cell

### "Alignment OK: False"
- Check Global Alignment section for specific failure
- If sig_ok=False: Models trained on different splits
- If idx_ok=False: Models have different samples or order
- Solution: Re-run all notebook export cells to ensure consistency

### Strict mode exits with code 1
- Review validation output for FAIL models or alignment issues
- Fix root cause (re-train or re-export)
- Verify with non-strict mode first

## Success Criteria

B4 Streamlit MVP fully compliant when:
- [x] Validation script runs and displays all required fields
- [x] All PASS models have classes_fp = `cdfa70b13f7390e6`
- [x] Global alignment checks (sig_ok, idx_ok) displayed
- [x] Metrics computed from single source (utils/metrics.py)
- [x] Streamlit app loads with 4 sections
- [x] Fusion blocked when alignment fails (checkbox disabled)
- [x] Fusion enabled when alignment passes (checkbox enabled)
- [x] Table-driven charts use single DataFrame (df_metrics)
- [x] Strict mode returns appropriate exit codes
- [x] Output file writing works (--out flag)

## Conclusion

This validation approach provides:
- **Deterministic verification** via command-line script
- **CI/CD integration** via --strict flag
- **Audit trail** via --out file output
- **User-friendly UI** via Streamlit app
- **Read-only consumption** (no training, no data modification)
- **Explicit blocking** of misaligned model fusion

Run the validation commands above on your real artifacts to verify full compliance.
