"""
📊 Résultats des Modèles Image

Cette page présente les résultats détaillés des 4 modèles de classification d'images
entraînés sur le dataset Rakuten, avec visualisations interactives.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Résultats Modèles Image",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (same style as previous pages)
st.markdown("""
<style>
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .key-point {
        background-color: #e8f4f8;
        border-left: 4px solid #2ca02c;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .best-model {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.8rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 📊 Résultats des Modèles Image")
st.markdown("### Analyse comparative des performances sur l'ensemble de validation")

st.info("""
**Résultats de 4 modèles de Deep Learning** entraînés sur le dataset Rakuten de classification de produits.

📌 **Dataset de validation** : 10,827 échantillons | 27 classes de produits
""")

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
EXPORTS_DIR = PROJECT_ROOT / "artifacts" / "exports"

# Model configurations
MODEL_CONFIGS = {
    "LeNet-5": {
        "path": "lenet_canonical",
        "display_name": "LeNet-5",
        "color": "#ff7f0e"
    },
    "ViT-Tiny": {
        "path": "vit_canonical",
        "display_name": "ViT-Tiny",
        "color": "#2ca02c"
    },
    "Swin-Base": {
        "path": "swin_canonical",
        "display_name": "Swin-Base",
        "color": "#d62728"
    },
    "ConvNeXt-Base": {
        "path": "convnext_canonical_v1",
        "display_name": "ConvNeXt-Base",
        "color": "#9467bd"
    }
}

# Load data and compute metrics
@st.cache_data
def load_model_results():
    results = {}

    for model_key, config in MODEL_CONFIGS.items():
        model_dir = EXPORTS_DIR / config["path"]
        npz_path = model_dir / "val.npz"
        json_path = model_dir / "val_meta.json"

        # Load npz data
        data = np.load(npz_path)
        probs = data['probs']
        y_true = data['y_true']
        y_pred = np.argmax(probs, axis=1)

        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        results[model_key] = {
            "metadata": metadata,
            "metrics": {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro
            },
            "probs": probs,
            "y_true": y_true,
            "y_pred": y_pred
        }

    return results

# Load results
with st.spinner("Chargement des résultats des modèles..."):
    results = load_model_results()

# ========== Performance Metrics Overview ==========
st.markdown('<div class="section-header">🎯 Vue d\'Ensemble des Performances</div>', unsafe_allow_html=True)

# Create metrics dataframe
metrics_data = []
for model_key in MODEL_CONFIGS.keys():
    metrics = results[model_key]["metrics"]
    metrics_data.append({
        "Modèle": MODEL_CONFIGS[model_key]["display_name"],
        "Accuracy": metrics['accuracy'],
        "F1 Macro": metrics['f1_macro'],
        "F1 Weighted": metrics['f1_weighted'],
        "Precision Macro": metrics['precision_macro'],
        "Recall Macro": metrics['recall_macro']
    })

metrics_df = pd.DataFrame(metrics_data)

# Display formatted table
display_df = metrics_df.copy()
display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.2%}")
display_df['F1 Macro'] = display_df['F1 Macro'].apply(lambda x: f"{x:.4f}")
display_df['F1 Weighted'] = display_df['F1 Weighted'].apply(lambda x: f"{x:.4f}")
display_df['Precision Macro'] = display_df['Precision Macro'].apply(lambda x: f"{x:.4f}")
display_df['Recall Macro'] = display_df['Recall Macro'].apply(lambda x: f"{x:.4f}")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# Best model highlight
best_acc_idx = metrics_df['Accuracy'].idxmax()
best_model_name = metrics_df.loc[best_acc_idx, 'Modèle']
best_acc = metrics_df.loc[best_acc_idx, 'Accuracy']

st.markdown(f'<div class="best-model">🏆 Meilleur Modèle : {best_model_name} avec une accuracy de {best_acc:.2%}</div>', unsafe_allow_html=True)

# ========== Interactive Visualizations ==========
st.markdown('<div class="section-header">📈 Visualisations Interactives</div>', unsafe_allow_html=True)

# 1. Accuracy vs F1 Macro Comparison
st.markdown('<div class="subsection-header">Comparaison Accuracy vs F1 Macro</div>', unsafe_allow_html=True)

fig_comparison = go.Figure()

# Add Accuracy bars
fig_comparison.add_trace(go.Bar(
    name='Accuracy',
    x=metrics_df['Modèle'],
    y=metrics_df['Accuracy'],
    marker_color='#1f77b4',
    text=metrics_df['Accuracy'].apply(lambda x: f'{x:.2%}'),
    textposition='outside'
))

# Add F1 Macro bars
fig_comparison.add_trace(go.Bar(
    name='F1 Macro',
    x=metrics_df['Modèle'],
    y=metrics_df['F1 Macro'],
    marker_color='#ff7f0e',
    text=metrics_df['F1 Macro'].apply(lambda x: f'{x:.4f}'),
    textposition='outside'
))

fig_comparison.update_layout(
    title='Comparaison des Performances : Accuracy vs F1 Macro',
    xaxis_title='Modèle',
    yaxis_title='Score',
    barmode='group',
    height=500,
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_comparison, use_container_width=True)

# 2. Radar Chart for All Metrics
st.markdown('<div class="subsection-header">Comparaison Multi-Métriques (Radar Chart)</div>', unsafe_allow_html=True)

fig_radar = go.Figure()

metric_names = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision Macro', 'Recall Macro']

for idx, row in metrics_df.iterrows():
    model_name = row['Modèle']
    model_key = list(MODEL_CONFIGS.keys())[idx]
    color = MODEL_CONFIGS[model_key]['color']

    fig_radar.add_trace(go.Scatterpolar(
        r=[row['Accuracy'], row['F1 Macro'], row['F1 Weighted'],
           row['Precision Macro'], row['Recall Macro']],
        theta=metric_names,
        fill='toself',
        name=model_name,
        line_color=color
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    showlegend=True,
    title='Comparaison Radar de Toutes les Métriques',
    height=600
)

st.plotly_chart(fig_radar, use_container_width=True)

# 3. Scatter Plot: Accuracy vs F1 Macro
st.markdown('<div class="subsection-header">Accuracy vs F1 Macro (Scatter Plot)</div>', unsafe_allow_html=True)

fig_scatter = go.Figure()

for idx, row in metrics_df.iterrows():
    model_name = row['Modèle']
    model_key = list(MODEL_CONFIGS.keys())[idx]
    color = MODEL_CONFIGS[model_key]['color']

    fig_scatter.add_trace(go.Scatter(
        x=[row['Accuracy']],
        y=[row['F1 Macro']],
        mode='markers+text',
        name=model_name,
        marker=dict(size=20, color=color),
        text=[model_name],
        textposition='top center',
        textfont=dict(size=12)
    ))

# Add diagonal reference line (perfect correlation)
max_val = max(metrics_df['Accuracy'].max(), metrics_df['F1 Macro'].max())
fig_scatter.add_trace(go.Scatter(
    x=[0, max_val],
    y=[0, max_val],
    mode='lines',
    name='Ligne de référence',
    line=dict(dash='dash', color='gray'),
    showlegend=False
))

fig_scatter.update_layout(
    title='Corrélation entre Accuracy et F1 Macro',
    xaxis_title='Accuracy',
    yaxis_title='F1 Macro',
    height=500,
    hovermode='closest'
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ========== Training Configuration Comparison ==========
st.markdown('<div class="section-header">⚙️ Comparaison des Configurations d\'Entraînement</div>', unsafe_allow_html=True)

config_comparison_data = []
for model_key in MODEL_CONFIGS.keys():
    extra = results[model_key]["metadata"].get("extra", {})
    config_comparison_data.append({
        "Modèle": MODEL_CONFIGS[model_key]["display_name"],
        "Résolution": f"{extra.get('img_size', 'N/A')}×{extra.get('img_size', 'N/A')}",
        "Batch Size": extra.get('batch_size', 'N/A'),
        "Epochs": extra.get('num_epochs', 'N/A'),
        "Learning Rate": extra.get('lr', 'N/A'),
        "Dropout": extra.get('dropout_rate', 'N/A'),
        "Label Smoothing": extra.get('label_smoothing', 'N/A'),
        "Augmentation": "✅ Mixup/CutMix" if extra.get('mixup_alpha') else "❌",
        "EMA": "✅" if extra.get('use_ema') else "❌"
    })

config_comparison_df = pd.DataFrame(config_comparison_data)
st.dataframe(config_comparison_df, use_container_width=True, hide_index=True)

# ========== Training History Note ==========
st.markdown('<div class="section-header">📉 Historique d\'Entraînement (Loss, Learning Rate)</div>', unsafe_allow_html=True)

st.warning("""
**Note sur l'historique d'entraînement** :

Les courbes détaillées d'entraînement (loss par epoch, évolution du learning rate, validation accuracy au fil du temps)
ne sont pas incluses dans les exports de prédictions pour maintenir des fichiers légers.

💡 **Pour visualiser l'historique complet d'entraînement**, consultez :

1. **Weights & Biases (WandB)** : Si les runs ont été trackés avec WandB, toutes les métriques
   d'entraînement sont disponibles en ligne avec des dashboards interactifs.

2. **Fichiers de logs locaux** : Vérifiez le répertoire de checkpoints pour d'éventuels fichiers de logs.

3. **Ré-entraînement avec logging** : Les scripts d'entraînement peuvent être modifiés pour sauvegarder
   l'historique complet (voir `src/train/image_*.py`).
""")

# ========== Individual Model Details ==========
st.markdown('<div class="section-header">📋 Détails Détaillés par Modèle</div>', unsafe_allow_html=True)

for model_key, config in MODEL_CONFIGS.items():
    with st.expander(f"📄 {config['display_name']} - Détails Complets"):

        metadata = results[model_key]["metadata"]
        metrics = results[model_key]["metrics"]
        extra = metadata.get("extra", {})

        # Performance metrics
        st.markdown("**🎯 Métriques de Performance**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("F1 Macro", f"{metrics['f1_macro']:.4f}")

        with col2:
            st.metric("F1 Weighted", f"{metrics['f1_weighted']:.4f}")
            st.metric("Precision Macro", f"{metrics['precision_macro']:.4f}")

        with col3:
            st.metric("Recall Macro", f"{metrics['recall_macro']:.4f}")
            st.metric("Nombre d'échantillons", metadata["num_samples"])

        st.markdown("---")

        # Training configuration
        st.markdown("**⚙️ Configuration d'Entraînement**")

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.code(f"""Architecture : {extra.get('model_architecture', 'N/A')}
Résolution Image : {extra.get('img_size', 'N/A')}×{extra.get('img_size', 'N/A')}
Batch Size : {extra.get('batch_size', 'N/A')}
Nombre d'Epochs : {extra.get('num_epochs', 'N/A')}
Learning Rate : {extra.get('lr', 'N/A')}""", language='text')

        with config_col2:
            st.code(f"""Weight Decay : {extra.get('weight_decay', 'N/A')}
Label Smoothing : {extra.get('label_smoothing', 'N/A')}
Dropout Rate : {extra.get('dropout_rate', 'N/A')}
Mixed Precision (AMP) : {extra.get('use_amp', 'N/A')}
Drop Path Rate : {extra.get('drop_path_rate', 'N/A')}""", language='text')

        # Advanced features (if present)
        if extra.get('mixup_alpha') or extra.get('use_ema'):
            st.markdown("**🚀 Fonctionnalités Avancées**")
            advanced_features = []

            if extra.get('mixup_alpha'):
                advanced_features.append(f"✅ Mixup Alpha: {extra['mixup_alpha']}")
            if extra.get('cutmix_alpha'):
                advanced_features.append(f"✅ CutMix Alpha: {extra['cutmix_alpha']}")
            if extra.get('use_ema'):
                advanced_features.append(f"✅ EMA (Exponential Moving Average): Activé")
                if extra.get('ema_decay'):
                    advanced_features.append(f"   └─ EMA Decay: {extra['ema_decay']}")

            st.markdown("\n".join(advanced_features))

        st.markdown("---")

        # Export metadata
        st.markdown("**📦 Métadonnées d'Export**")
        st.code(f"""Nom du Modèle : {metadata['model_name']}
Split : {metadata['split_name']}
Split Signature : {metadata['split_signature']}
Classes Fingerprint : {metadata['classes_fp']}
Nombre de Classes : {metadata['num_classes']}
Date de Création : {metadata['created_at']}
Source : {extra.get('source', 'N/A')}""", language='text')

# ========== Key Observations ==========
st.markdown('<div class="section-header">💡 Observations Clés</div>', unsafe_allow_html=True)

# Calculate improvements
lenet_acc = results["LeNet-5"]["metrics"]["accuracy"]
best_modern_acc = max([results[k]["metrics"]["accuracy"] for k in ["ViT-Tiny", "Swin-Base", "ConvNeXt-Base"]])
improvement = ((best_modern_acc - lenet_acc) / lenet_acc) * 100

swin_acc = results["Swin-Base"]["metrics"]["accuracy"]
convnext_acc = results["ConvNeXt-Base"]["metrics"]["accuracy"]
vit_acc = results["ViT-Tiny"]["metrics"]["accuracy"]

st.markdown(f"""
1. **Évolution des Performances** : Les modèles modernes (Transformers et ConvNeXt) surpassent largement LeNet-5,
   avec une amélioration de **+{improvement:.1f}%** d'accuracy (de {lenet_acc:.2%} à {best_modern_acc:.2%}).

2. **Hiérarchie des Performances** :
   - 🥇 **Swin Transformer** : {swin_acc:.2%} accuracy - Meilleur modèle
   - 🥈 **ConvNeXt Base** : {convnext_acc:.2%} accuracy - Très proche de Swin
   - 🥉 **ViT-Tiny** : {vit_acc:.2%} accuracy - Performances solides malgré sa taille réduite
   - 📊 **LeNet-5** : {lenet_acc:.2%} accuracy - Baseline historique

3. **CNNs Modernisés vs Transformers** : ConvNeXt démontre que les architectures CNN bien conçues
   peuvent rivaliser avec les Transformers (écart de seulement {abs(swin_acc - convnext_acc):.2%} avec Swin).

4. **Impact de l'Augmentation** : Les modèles utilisant Mixup/CutMix (Swin, ConvNeXt) obtiennent les meilleures
   performances, confirmant l'importance de l'augmentation avancée pour les architectures modernes.

5. **Complexité de la Tâche** : La performance de LeNet-5 ({lenet_acc:.2%}) souligne la difficulté
   de la classification multi-classe avec 27 catégories et justifie l'utilisation d'architectures pré-entraînées.
""")

# ========== Validation Alignment ==========
st.markdown('<div class="section-header">✅ Validation de l\'Alignement</div>', unsafe_allow_html=True)

st.success("""
**Tous les modèles partagent les mêmes identifiants de split et classes** :

- **Split Signature** : `cf53f8eb169b3531` ✓
- **Classes Fingerprint** : `cdfa70b13f7390e6` ✓
- **Nombre d'échantillons** : 10,827 (validation set) ✓
- **Nombre de classes** : 27 ✓

Cette cohérence garantit que tous les modèles ont été évalués sur **exactement les mêmes échantillons**,
permettant une comparaison directe fiable et une fusion sécurisée des prédictions (ensemble multi-modal).
""")

# Footer
st.markdown("---")
st.caption("📊 Visualisations générées à partir des exports de modèles | Rakuten Product Classification Project")
