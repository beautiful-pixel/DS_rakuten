# DS Rakuten - Image Model Comparison App

Read-only Streamlit consumer for DS_rakuten image classification model predictions with strict validation.

## Features

- **Strict Validation**: Enforces canonical classes and split alignment
- **Model Comparison**: Side-by-side metrics and charts
- **Mean Fusion**: Automatic ensemble of aligned models
- **Read-Only**: No training, no data modification, no silent fixes
- **Consistent Charts**: Single DataFrame source for all visualizations

## Installation

Ensure you have the DS_rakuten repository set up with Phase 1-3 complete:
- Phase 1: Canonical splits (split_manager.py)
- Phase 2: Canonical classes (label_mapping.py)
- Phase 3: Export contract (model_exporter.py)

Install Streamlit:
```bash
pip install streamlit pandas numpy
```

## Directory Structure

Expected structure for model exports:
```
artifacts/
├── exports/
│   ├── resnet50/
│   │   ├── val.npz
│   │   └── val_meta.json
│   ├── vit_base/
│   │   ├── val.npz
│   │   └── val_meta.json
│   ├── swin_v2/
│   │   ├── val.npz
│   │   └── val_meta.json
│   └── convnext/
│       ├── val.npz
│       └── val_meta.json
└── canonical_classes.json
```

**Important**: Model exports must be created by running the export cells in notebooks (Phase 3).

## Running the App

From the repository root:

```bash
streamlit run apps/image_app/app.py
```

The app will automatically:
1. Scan `artifacts/exports/` for model predictions
2. Load and validate each model against canonical classes
3. Check global alignment (split_signature + idx)
4. Display validation results, metrics, and enable fusion if aligned

## UI Sections

### 1. Validation Overview
- Shows all discovered models with their validation status
- PASS: All validations passed (canonical classes, split signature, etc.)
- FAIL: Validation failed (shows specific reason)
- SKIPPED: Missing required files

### 2. Global Alignment
- **Split Signature**: Checks all PASS models share the same split_signature
- **Index Alignment**: Checks all PASS models have identical idx (values + order)
- Fusion is BLOCKED if either check fails

### 3. Metrics (if y_true available)
- Accuracy and Macro F1 for each PASS model
- Bar charts for visual comparison
- Metrics table with detailed values

### 4. Mean Fusion
- Enabled only when:
  - At least 2 PASS models exist
  - Global alignment checks pass (sig_ok + idx_ok)
- Computes average of probabilities across models
- Shows fusion metrics
- Optional: Export fusion artifacts to `artifacts/exports/fusion_mean/`

## Controls (Sidebar)

- **Split**: Select `val` or `test` split
- **Show only PASS models**: Filter validation table
- **Enable mean fusion**: Toggle fusion computation (auto-disabled if alignment fails)

## Validation Rules

The app enforces strict validation (READ-ONLY, no silent fixes):

1. **Canonical Classes FP**: `cdfa70b13f7390e6`
   - Computed from loaded classes must match
   - Metadata classes_fp must match
   - Classes array must exactly equal CANONICAL_CLASSES

2. **Split Signature**: Must be non-empty and consistent across all PASS models

3. **Index Alignment**: All PASS models must have identical idx (same values in same order)

4. **No Modifications**: App never reorders probabilities or modifies data

## Filtering Logic

The scanner:
- ✅ Includes: Normal model directories with {split}.npz + {split}_meta.json
- ❌ Ignores: Directories starting with `_`, `test`, or `debug`
- ⚠️ Skips: Directories missing required files (shown in validation table)

## Troubleshooting

### "No model directories found"
- Ensure notebooks with export cells have been run
- Check `artifacts/exports/` exists and contains model subdirectories
- Verify each model has both `.npz` and `_meta.json` files

### "Fusion is BLOCKED"
- Check Global Alignment section for specific failure
- Split signature mismatch: Models trained on different splits
- Index mismatch: Models have different samples or order (re-run exports)

### "classes_fp mismatch"
- Model was trained before Phase 2 canonical classes
- Re-run notebook export cell to ensure alignment

## Example Terminal Output

```bash
$ streamlit run apps/image_app/app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

Navigate to the URL to view the app.

## Dependencies

- `streamlit`: Web UI framework
- `pandas`: DataFrame for metrics table
- `numpy`: Array operations
- `src.data.label_mapping`: Canonical classes
- `src.export.model_exporter`: Load predictions
- `apps.image_app.utils.*`: Metrics, fusion, exports loader

## License

Part of DS_rakuten project.
