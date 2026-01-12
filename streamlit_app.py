"""
Rakuten Product Classification - Model Analysis Dashboard
Main Streamlit Application
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Rakuten ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-top: 5rem;
        margin-bottom: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-title">Rakuten Classification DeepLearning</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Modal Product Classification Project</div>', unsafe_allow_html=True)

# Info box
st.info("""
**Projet de classification multi-classe utilisant des modèles de Deep Learning**

📌 **Objectif** : Classifier automatiquement les produits Rakuten en 27 catégories

🔍 **Approches** :
- 🖼️ **Modèles Image** : LeNet, ResNet50, ViT, Swin Transformer, ConvNeXt
- 📝 **Modèles Texte** : CamemBERT, XLM-RoBERTa, mDeBERTa
- 🔄 **Fusion** : Ensemble multi-modal pour maximiser les performances

📊 **Dataset** :
- 84,916 échantillons au total
- 10,827 échantillons de validation
- 27 catégories de produits

---

👈 **Naviguez dans les pages via la barre latérale pour explorer :**
- Mécanisme de garantie d'alignement des indices
- Architectures des modèles Image
- Et plus encore...
""")

st.markdown("---")
st.caption("🚀 Projet Rakuten - Classification de Produits par Deep Learning")
