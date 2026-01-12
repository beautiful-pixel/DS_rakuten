"""
🖼️ Modèles de Classification d'Images

Cette page présente les 5 modèles de deep learning utilisés pour la classification
des images de produits Rakuten, leur histoire, caractéristiques et configurations.
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Modèles Image",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS (same style as Index Alignment page)
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
    .model-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
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
    .history-box {
        background-color: #fff9e6;
        border-left: 4px solid #ffa500;
        padding: 1rem;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🖼️ Modèles de Classification d'Images")
st.markdown("### Vue d'ensemble des architectures de Deep Learning pour la classification de produits")

st.info("""
**5 architectures de réseaux de neurones convolutifs** ont été explorées dans ce projet,
représentant l'évolution des approches de computer vision de 1998 à 2022.
""")

# ========== LeNet-5 ==========
st.markdown('<div class="section-header">1. LeNet-5 (1998)</div>', unsafe_allow_html=True)

st.markdown('<div class="history-box">', unsafe_allow_html=True)
st.markdown("""
**📚 Histoire et Contexte**

LeNet-5, développé par Yann LeCun et ses collègues aux Bell Labs en 1998, est l'un des premiers
réseaux de neurones convolutifs (CNN) réussis. Conçu initialement pour la reconnaissance de
chiffres manuscrits (MNIST), il a posé les bases de l'architecture CNN moderne.

**🎯 Caractéristiques Principales**
- **Architecture simple** : Seulement 2 couches convolutives et 3 couches fully-connected
- **Faible nombre de paramètres** : ~60K paramètres (très léger)
- **Taille d'entrée réduite** : Conçu pour des images 32×32 pixels
- **Activation** : Utilise tanh (dans la version originale)
- **Pooling** : Average pooling après chaque convolution
- **Contexte historique** : Révolutionnaire pour son époque, mais considéré comme basique aujourd'hui
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="key-point">✅ <b>Utilisation dans le projet</b> : Modèle de référence (baseline) pour évaluer les gains des architectures modernes</div>', unsafe_allow_html=True)

with st.expander("⚙️ Configuration LeNet-5 (src/train/image_lenet.py)"):
    st.code('''@dataclass
class LeNetConfig:
    # Data / IO
    raw_dir: str                      # Chemin vers les fichiers CSV bruts
    img_dir: str                      # Chemin vers le répertoire d'images
    out_dir: str                      # Répertoire d'export des prédictions
    ckpt_dir: str                     # Répertoire des checkpoints

    # Training
    img_size: int = 32                # LeNet utilise typiquement 32x32
    batch_size: int = 128
    num_workers: int = 4
    num_epochs: int = 30
    lr: float = 1e-3                  # Learning rate
    weight_decay: float = 1e-4
    use_amp: bool = True              # Mixed precision training

    # Regularization
    dropout_rate: float = 0.5

    # Scheduler
    plateau_factor: float = 0.5       # Réduction du LR sur plateau
    plateau_patience: int = 3

    # Runtime
    device: Optional[str] = None      # "cuda" ou "cpu"
    model_name: str = "lenet_v1"
    export_split: str = "val"''', language='python')

    st.markdown("**Architecture du Modèle**")
    st.code('''class LeNet5(nn.Module):
    """
    LeNet-5 adapté pour la classification de produits Rakuten.

    Architecture :
    - Conv1 : 3 → 6 canaux, kernel 5×5
    - MaxPool1 : 2×2
    - Conv2 : 6 → 16 canaux, kernel 5×5
    - MaxPool2 : 2×2
    - FC1 : → 120
    - FC2 : 120 → 84
    - FC3 : 84 → num_classes (27)
    """''', language='python')

    st.caption("📄 Source : `src/train/image_lenet.py`")

# ========== ResNet50 ==========
st.markdown('<div class="section-header">2. ResNet50 (2015)</div>', unsafe_allow_html=True)

st.markdown('<div class="history-box">', unsafe_allow_html=True)
st.markdown("""
**📚 Histoire et Contexte**

ResNet (Residual Network), développé par Microsoft Research en 2015, a remporté le concours
ImageNet ILSVRC 2015. L'innovation clé est l'introduction des **connexions résiduelles** (skip connections)
qui permettent d'entraîner des réseaux très profonds (50, 101, 152 couches ou plus).

**🎯 Caractéristiques Principales**
- **Connexions résiduelles** : F(x) + x, résout le problème de dégradation des réseaux profonds
- **Architecture modulaire** : Blocs résiduels empilables
- **50 couches** : Équilibre entre profondeur et efficacité computationnelle
- **Batch Normalization** : Après chaque convolution
- **Pré-entraînement ImageNet** : Transfer learning sur 1000 classes
- **~25M paramètres** : Plus lourd que LeNet mais gérable
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="key-point">✅ <b>Utilisation dans le projet</b> : Modèle CNN classique robuste avec transfer learning ImageNet</div>', unsafe_allow_html=True)

with st.expander("⚙️ Configuration ResNet50 (src/train/image_resnet50.py)"):
    st.code('''@dataclass
class ResNet50Config:
    # Data / IO
    raw_dir: str                      # Chemin vers les CSV (données brutes)
    img_dir: str                      # Chemin vers les images
    out_dir: str                      # Export des prédictions
    ckpt_dir: str                     # Checkpoints du modèle

    # Training
    img_size: int = 224               # Résolution ImageNet standard
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 1
    lr: float = 1e-4
    use_amp: bool = True              # Mixed precision

    # Regularization
    label_smoothing: float = 0.1      # Label smoothing
    dropout_rate: float = 0.3         # Dropout dans la tête

    # Scheduler
    plateau_factor: float = 0.1       # Réduction du LR
    plateau_patience: int = 3

    # Runtime
    device: Optional[str] = None
    model_name: str = "resnet50_rerun_canonical"
    export_split: str = "val"''', language='python')

    st.markdown("**Construction du Modèle**")
    st.code('''def _build_resnet50(num_classes: int, dropout_rate: float) -> nn.Module:
    """
    ResNet50 avec torchvision et tête personnalisée dropout + linear.
    """
    model = torchvision.models.resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2  # Poids pré-entraînés
    )
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model''', language='python')

    st.caption("📄 Source : `src/train/image_resnet50.py`")

# ========== Vision Transformer (ViT) ==========
st.markdown('<div class="section-header">3. Vision Transformer - ViT (2020)</div>', unsafe_allow_html=True)

st.markdown('<div class="history-box">', unsafe_allow_html=True)
st.markdown("""
**📚 Histoire et Contexte**

Vision Transformer (ViT), publié par Google Research en 2020, a marqué un tournant en appliquant
l'architecture **Transformer** (initialement conçue pour le NLP) à la vision par ordinateur.
ViT démontre qu'on peut obtenir des performances excellentes **sans convolution**, uniquement avec
des mécanismes d'attention.

**🎯 Caractéristiques Principales**
- **Patch-based** : Découpe l'image en patches 16×16, traités comme des tokens
- **Self-attention multi-têtes** : Capture les relations globales entre patches
- **Position embeddings** : Encodage de la position spatiale des patches
- **Architecture pure Transformer** : Pas de convolution
- **Scalabilité** : Performance augmente avec la taille du dataset et du modèle
- **Variants** : ViT-Tiny, Base, Large, Huge (86M → 632M paramètres)
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="key-point">✅ <b>Utilisation dans le projet</b> : Représente l\'approche Transformer pure pour la vision, alternative aux CNNs</div>', unsafe_allow_html=True)

with st.expander("⚙️ Configuration ViT (src/train/image_vit.py)"):
    st.code('''@dataclass
class ViTConfig:
    # Data / IO
    raw_dir: str                      # Données CSV brutes
    img_dir: str                      # Images
    out_dir: str                      # Export des prédictions
    ckpt_dir: str                     # Checkpoints

    # Training
    img_size: int = 224               # Résolution standard ViT
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 1
    lr: float = 1e-4
    use_amp: bool = True

    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.1         # Dropout léger (ViT régularisé par nature)

    # Scheduler
    plateau_factor: float = 0.1
    plateau_patience: int = 3

    # Model
    vit_model_name: str = "vit_tiny_patch16_224"  # Modèle timm
    vit_pretrained: bool = True                   # Poids pré-entraînés

    # Runtime
    device: Optional[str] = None
    model_name: str = "vit_rerun_canonical_smoke"
    export_split: str = "val"
    force_colab_loader: bool = False''', language='python')

    st.markdown("**Principe de Fonctionnement**")
    st.markdown("""
    1. **Patchification** : Image 224×224 → 196 patches de 16×16
    2. **Linear projection** : Chaque patch → embedding de dimension D
    3. **Position embedding** : Ajoute l'information de position spatiale
    4. **Transformer encoder** : N couches d'attention multi-têtes
    5. **Classification head** : MLP sur le token [CLS]
    """)

    st.caption("📄 Source : `src/train/image_vit.py`")

# ========== Swin Transformer ==========
st.markdown('<div class="section-header">4. Swin Transformer (2021)</div>', unsafe_allow_html=True)

st.markdown('<div class="history-box">', unsafe_allow_html=True)
st.markdown("""
**📚 Histoire et Contexte**

Swin Transformer (Shifted Window Transformer), développé par Microsoft Research en 2021,
améliore ViT en introduisant une **hiérarchie multi-échelle** et une **attention locale par fenêtres**.
Il combine les avantages des CNNs (inductive bias spatial) et des Transformers (self-attention).

**🎯 Caractéristiques Principales**
- **Attention par fenêtres décalées** : Réduit la complexité de O(n²) à O(n)
- **Hiérarchie pyramidale** : Feature maps de résolutions décroissantes (comme CNN)
- **Patch merging** : Réduit progressivement la résolution spatiale
- **Efficacité computationnelle** : Plus rapide que ViT grâce à l'attention locale
- **Performances SOTA** : Meilleur que ViT sur plusieurs benchmarks
- **Versatilité** : Excellente pour classification, détection, segmentation
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="key-point">✅ <b>Utilisation dans le projet</b> : Transformer hiérarchique optimisé pour la vision, meilleur compromis précision/vitesse</div>', unsafe_allow_html=True)

with st.expander("⚙️ Configuration Swin Transformer (src/train/image_swin.py)"):
    st.code('''@dataclass
class SwinConfig:
    # Data / IO
    raw_dir: str                      # CSV bruts
    img_dir: str                      # Images
    out_dir: str                      # Prédictions exportées
    ckpt_dir: str                     # Checkpoints

    # Training
    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 4
    num_epochs: int = 30
    lr: float = 5e-5                  # LR plus faible (fine-tuning)
    weight_decay: float = 0.05
    use_amp: bool = True

    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.5         # Head dropout
    head_dropout2: float = 0.3        # Second head dropout
    drop_path_rate: float = 0.3       # Stochastic depth

    # Mixup/CutMix augmentation
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5

    # Scheduler
    cosine_eta_min: float = 1e-6      # Cosine annealing

    # Model
    swin_model_name: str = "swin_base_patch4_window7_224"
    swin_pretrained: bool = True

    # Runtime
    device: Optional[str] = None
    model_name: str = "swin_v2"
    export_split: str = "val"
    force_colab_loader: bool = False''', language='python')

    st.markdown("**Architecture en Étages**")
    st.markdown("""
    - **Stage 1** : 56×56, fenêtre 7×7
    - **Stage 2** : 28×28, fenêtre 7×7
    - **Stage 3** : 14×14, fenêtre 7×7
    - **Stage 4** : 7×7, fenêtre 7×7
    - **Shifted Window** : Alterne attention locale et décalée entre couches
    """)

    st.caption("📄 Source : `src/train/image_swin.py`")

# ========== ConvNeXt ==========
st.markdown('<div class="section-header">5. ConvNeXt (2022)</div>', unsafe_allow_html=True)

st.markdown('<div class="history-box">', unsafe_allow_html=True)
st.markdown("""
**📚 Histoire et Contexte**

ConvNeXt, publié par Meta AI Research en 2022, est une **modernisation de ResNet** qui démontre
que les CNNs classiques peuvent rivaliser avec les Transformers si on adopte les bonnes pratiques
de design. C'est une réponse aux Transformers : "Les CNNs ne sont pas morts !"

**🎯 Caractéristiques Principales**
- **CNN modernisé** : Architecture inspirée de ResNet avec améliorations de Swin
- **Design choices** : Depthwise convolutions 7×7, GELU, Layer Normalization
- **Inductive bias** : Préserve les avantages des CNNs (translation equivariance)
- **Simplicité** : Pur CNN, pas besoin d'attention complexe
- **Performances compétitives** : Égale Swin Transformer avec moins de complexité
- **Scalabilité** : Variants Tiny, Small, Base, Large, XLarge
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="key-point">✅ <b>Utilisation dans le projet</b> : CNN de dernière génération, meilleur compromis entre simplicité (CNN) et performance (niveau Transformer)</div>', unsafe_allow_html=True)

with st.expander("⚙️ Configuration ConvNeXt (src/train/image_convnext.py)"):
    st.code('''@dataclass
class ConvNeXtConfig:
    # Data / IO
    raw_dir: str                      # CSV bruts
    img_dir: str                      # Images
    out_dir: str                      # Export prédictions
    ckpt_dir: str                     # Checkpoints

    # Training
    img_size: int = 384               # Résolution plus élevée pour ConvNeXt
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    use_amp: bool = True

    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.5         # Head dropout
    head_dropout2: float = 0.3        # Second head dropout
    drop_path_rate: float = 0.3       # Stochastic depth

    # Mixup/CutMix augmentation
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5

    # EMA (Exponential Moving Average)
    use_ema: bool = True              # EMA des poids du modèle
    ema_decay: float = 0.9999

    # Scheduler
    cosine_eta_min: float = 1e-6

    # Model
    convnext_model_name: str = "convnext_base"
    convnext_pretrained: bool = True

    # Runtime
    device: Optional[str] = None
    model_name: str = "convnext"
    export_split: str = "val"
    force_colab_loader: bool = False''', language='python')

    st.markdown("**Innovations de Design**")
    st.markdown("""
    1. **Macro design** : Stage ratios 1:1:3:1 (inspiré de Swin)
    2. **Depthwise conv 7×7** : Réceptif champ plus large
    3. **Inverted bottleneck** : Expand → Depthwise → Project
    4. **Less activation** : Une seule GELU par bloc
    5. **Layer Normalization** : Au lieu de Batch Normalization
    6. **Separate downsampling** : Couches dédiées pour réduction spatiale
    """)

    st.caption("📄 Source : `src/train/image_convnext.py`")

# ========== Tableau Comparatif ==========
st.markdown('<div class="section-header">📊 Tableau Comparatif des Modèles</div>', unsafe_allow_html=True)

comparison_data = {
    "Modèle": ["LeNet-5", "ResNet50", "ViT-Tiny", "Swin-Base", "ConvNeXt-Base"],
    "Année": ["1998", "2015", "2020", "2021", "2022"],
    "Type": ["CNN", "CNN (Residual)", "Transformer", "Transformer Hiérarchique", "CNN Modernisé"],
    "Paramètres": ["~60K", "~25M", "~5M", "~88M", "~89M"],
    "Résolution": ["32×32", "224×224", "224×224", "224×224", "384×384"],
    "Complexité": ["Très faible", "Moyenne", "Élevée (O(n²))", "Moyenne (O(n))", "Moyenne"],
    "Transfer Learning": ["❌", "✅ ImageNet", "✅ ImageNet", "✅ ImageNet", "✅ ImageNet"],
    "Innovation Clé": [
        "Premier CNN efficace",
        "Skip connections",
        "Attention globale",
        "Attention par fenêtres",
        "CNN meets Transformer"
    ]
}

import pandas as pd
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ========== Évolution Chronologique ==========
st.markdown('<div class="section-header">📈 Évolution de l\'Architecture</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Chronologie**")
    st.markdown("""
    - **1998** : LeNet-5
      └─ CNN basique
    - **2015** : ResNet
      └─ Connexions résiduelles
    - **2020** : ViT
      └─ Transformers pour la vision
    - **2021** : Swin
      └─ Transformers hiérarchiques
    - **2022** : ConvNeXt
      └─ Renaissance des CNNs
    """)

with col2:
    st.markdown("**Tendances et Leçons**")
    st.markdown("""
    1. **Profondeur** : De 7 couches (LeNet) à 100+ (ResNet, Transformers)
    2. **Inductive bias** : CNNs (local) → Transformers (global) → Hybride (Swin, ConvNeXt)
    3. **Attention mechanisms** : Révolution 2020-2022
    4. **Transfer learning** : Devenu standard depuis 2015
    5. **Augmentation** : Mixup/CutMix essentiels pour Transformers
    6. **Efficiency** : Swin et ConvNeXt optimisent le compromis performance/coût
    """)

# ========== Performances dans le Projet ==========
st.markdown('<div class="section-header">🎯 Performances dans le Projet Rakuten</div>', unsafe_allow_html=True)

st.info("""
**Résultats sur l'ensemble de validation** (10,827 échantillons, 27 classes)

Les performances détaillées de chaque modèle sont disponibles dans les pages suivantes :
- **Page "Model Performance"** : Métriques complètes (accuracy, F1, precision, recall)
- **Page "Model Comparison"** : Comparaison directe entre modèles
- **Page "Ensemble Analysis"** : Fusion optimale des modèles

**Remarque** : Les modèles plus récents (Swin, ConvNeXt) bénéficient de :
- Meilleure augmentation (Mixup/CutMix)
- Régularisation avancée (stochastic depth, EMA)
- Optimisation moderne (cosine annealing, warmup)
""")

# Footer
st.markdown("---")
st.caption("📝 Documentation générée à partir du code source | Rakuten Product Classification Project")
