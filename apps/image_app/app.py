"""
Streamlit Image Classification Model Comparison App

Read-only consumer for DS_rakuten image exports with strict validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Add repo root to path for imports
import sys
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from apps.image_app.utils.exports_loader import scan_exports, load_model_export, build_global_alignment
from apps.image_app.utils.metrics import accuracy, macro_f1
from apps.image_app.utils.fusion import mean_fusion, export_fusion
from src.data.label_mapping import CANONICAL_CLASSES_FP


# Page config
st.set_page_config(
    page_title="DS Rakuten - Image Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DS Rakuten - Image Model Comparison")
st.markdown("**Read-only consumer for model predictions with strict validation**")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.header("Controls")

split = st.sidebar.selectbox(
    "Split",
    options=["val", "test"],
    index=0,
    help="Select which split to analyze"
)

show_only_pass = st.sidebar.checkbox(
    "Show only PASS models",
    value=False,
    help="Filter validation table to show only PASS models"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Canonical Classes FP:** `{CANONICAL_CLASSES_FP}`")

# ============================================================================
# LOAD DATA
# ============================================================================

# Scan exports directory
scanned = scan_exports(exports_root="artifacts/exports", split=split)

if len(scanned) == 0:
    st.error(f"No model directories found in `artifacts/exports/` for split `{split}`")
    st.info("**Expected structure:** `artifacts/exports/<model_name>/{split}.npz` + `{split}_meta.json`")
    st.stop()

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

# Build global alignment check
alignment = build_global_alignment(models_loaded)

# ============================================================================
# SECTION A: VALIDATION OVERVIEW
# ============================================================================
st.header("🔍 Validation Overview")

# Build validation table
validation_data = []
for m in models_loaded:
    validation_data.append({
        "Model": m["model_name"],
        "Status": m["status"],
        "Fail Reason": m["fail_reason"] if m["fail_reason"] else "-",
        "Samples": m["n_samples"] if m["n_samples"] > 0 else "-",
        "Split Signature": m["split_signature"] if m["split_signature"] else "-",
        "Classes FP": m["classes_fp"] if m["classes_fp"] else "-"
    })

df_validation = pd.DataFrame(validation_data)

# Filter if requested
if show_only_pass:
    df_validation = df_validation[df_validation["Status"] == "PASS"]

st.dataframe(df_validation, use_container_width=True)

# ============================================================================
# SECTION B: GLOBAL ALIGNMENT
# ============================================================================
st.header("🔗 Global Alignment")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Split Signature")
    if alignment["sig_ok"]:
        st.success(f"✅ All PASS models aligned: `{alignment['sig_value']}`")
    else:
        st.error(f"❌ Signature mismatch: {alignment['sig_value']}")
        with st.expander("Per-model signatures"):
            for model_name, sig in alignment["per_model_sig"].items():
                st.text(f"{model_name}: {sig}")

with col2:
    st.subheader("Index Alignment")
    if alignment["idx_ok"]:
        st.success("✅ All PASS models have identical idx (values + order)")
    else:
        st.error(f"❌ Index mismatch: {alignment['idx_mismatch_reason']}")

# Check if fusion is possible
fusion_blocked = not (alignment["sig_ok"] and alignment["idx_ok"])
num_pass_models = len(alignment["all_pass_models"])

if fusion_blocked and num_pass_models > 0:
    st.warning("⚠️ Mean fusion is **BLOCKED** due to alignment failures")

# ============================================================================
# SECTION C: METRICS (only if y_true exists for PASS models)
# ============================================================================

# Check if any PASS model has y_true
pass_models = [m for m in models_loaded if m["status"] == "PASS"]
has_y_true = any(m["y_true"] is not None for m in pass_models)

if has_y_true and len(pass_models) > 0:
    st.header("📈 Metrics")

    # Build metrics DataFrame (single source of truth)
    metrics_data = []

    for m in pass_models:
        if m["y_true"] is not None:
            acc = accuracy(m["y_true"], m["probs"])
            f1 = macro_f1(m["y_true"], m["probs"], num_classes=27)

            metrics_data.append({
                "model_name": m["model_name"],
                "accuracy": acc,
                "macro_f1": f1,
                "n_samples": m["n_samples"]
            })

    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.set_index("model_name")

    # Show metrics table
    st.subheader("Metrics Table")
    st.dataframe(df_metrics.style.format({
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "n_samples": "{:,.0f}"
    }), use_container_width=True)

    # Bar charts (using the single DataFrame)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy")
        st.bar_chart(df_metrics[["accuracy"]])

    with col2:
        st.subheader("Macro F1")
        st.bar_chart(df_metrics[["macro_f1"]])

    # Store df_metrics in session state for fusion section
    st.session_state["df_metrics"] = df_metrics

elif len(pass_models) > 0:
    st.info("ℹ️ Metrics hidden: No y_true available for PASS models")

# ============================================================================
# SECTION D: MEAN FUSION
# ============================================================================
st.header("🔀 Mean Fusion")

enable_fusion = st.sidebar.checkbox(
    "Enable mean fusion",
    value=False,
    disabled=fusion_blocked or num_pass_models < 2,
    help="Compute mean fusion of PASS models (requires global alignment)"
)

if num_pass_models < 2:
    st.info(f"ℹ️ Fusion requires at least 2 PASS models (found {num_pass_models})")
elif fusion_blocked:
    st.warning("⚠️ Fusion is BLOCKED due to alignment failures (see Global Alignment section)")
elif not enable_fusion:
    st.info("ℹ️ Enable fusion in the sidebar to compute mean fusion")
else:
    # Compute fusion
    st.subheader("Computing Mean Fusion...")

    # Collect probs from PASS models
    probs_list = [m["probs"] for m in pass_models]
    fused_probs = mean_fusion(probs_list)

    # Get reference data (all PASS models have same idx due to alignment check)
    reference = pass_models[0]
    idx = reference["idx"]
    split_signature = reference["split_signature"]
    y_true = reference.get("y_true")  # May be None

    st.success(f"✅ Fused {len(probs_list)} models")

    # Compute fusion metrics if y_true exists
    if y_true is not None:
        fusion_acc = accuracy(y_true, fused_probs)
        fusion_f1 = macro_f1(y_true, fused_probs, num_classes=27)

        st.metric("Fusion Accuracy", f"{fusion_acc:.4f}")
        st.metric("Fusion Macro F1", f"{fusion_f1:.4f}")

        # Add fusion row to metrics DataFrame for consistent charting
        if "df_metrics" in st.session_state:
            df_metrics_with_fusion = st.session_state["df_metrics"].copy()
            df_metrics_with_fusion.loc["fusion_mean"] = {
                "accuracy": fusion_acc,
                "macro_f1": fusion_f1,
                "n_samples": len(idx)
            }

            st.subheader("Updated Metrics with Fusion")
            st.dataframe(df_metrics_with_fusion.style.format({
                "accuracy": "{:.4f}",
                "macro_f1": "{:.4f}",
                "n_samples": "{:,.0f}"
            }), use_container_width=True)

            # Updated charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Accuracy (with Fusion)")
                st.bar_chart(df_metrics_with_fusion[["accuracy"]])

            with col2:
                st.subheader("Macro F1 (with Fusion)")
                st.bar_chart(df_metrics_with_fusion[["macro_f1"]])

    # Export fusion artifacts
    st.subheader("Export Fusion Artifacts")

    if st.button("Export fusion to artifacts/exports/fusion_mean/"):
        try:
            export_result = export_fusion(
                out_dir="artifacts/exports",
                split_name=split,
                idx=idx,
                split_signature=split_signature,
                probs=fused_probs,
                y_true=y_true,
                extra_meta={
                    "fused_models": [m["model_name"] for m in pass_models],
                    "num_models": len(pass_models)
                }
            )

            st.success(f"✅ Fusion exported successfully!")
            st.code(f"NPZ: {export_result['npz_path']}\nMeta: {export_result['meta_json_path']}")
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(f"Split: `{split}` | PASS models: {num_pass_models} | Canonical Classes FP: `{CANONICAL_CLASSES_FP}`")
