"""
🔒 Mécanisme de Garantie d'Alignement des Indices

Cette page détaille le mécanisme d'alignement des indices du projet Rakuten,
qui garantit que les prédictions des modèles multi-modaux (Image + Texte) peuvent être parfaitement fusionnées.
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Mécanisme d'Alignement des Indices",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
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
    .code-section {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
    .key-point {
        background-color: #e8f4f8;
        border-left: 4px solid #2ca02c;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🔒 Mécanisme de Garantie d'Alignement des Indices")
st.markdown("### Principe fondamental : Chaîne complète de traçabilité des indices depuis les données brutes jusqu'à l'export")

st.info("""
**Explication détaillée de la garantie de cohérence des indices.**

Ce mécanisme garantit que les prédictions des modèles Image et Texte peuvent être alignées précisément par échantillon,
permettant ainsi une fusion sécurisée des modèles (Ensemble).
""")

# ========== Première Garantie : Split Manager Unifié ==========
st.markdown('<div class="section-header">📐 Première Garantie : Split Manager Unifié</div>', unsafe_allow_html=True)

st.markdown('<div class="subsection-header">1.1 Source Unique de Vérité (Single Source of Truth)</div>', unsafe_allow_html=True)

st.markdown("""
Tous les modèles (Image et Texte) utilisent **exactement les mêmes** indices de découpage des données,
qui sont sauvegardés dans des fichiers unifiés.
""")

with st.expander("📄 Voir le code source : src/data/split_manager.py (lignes 23-82)"):
    st.code('''def load_splits(verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Charge les splits canoniques (indices train/val/test)

    Priorité :
    1. Si data/splits/*.txt existe → charge directement (source unique de vérité)
    2. Sinon → appelle generate_splits() et sauvegarde dans des fichiers
    """
    train_file = SPLITS_DIR / "train_idx.txt"
    val_file = SPLITS_DIR / "val_idx.txt"
    test_file = SPLITS_DIR / "test_idx.txt"

    # Chargement depuis les fichiers
    if train_file.exists() and val_file.exists() and test_file.exists():
        train_idx = np.loadtxt(train_file, dtype=int)  # Indices de position des données brutes
        val_idx = np.loadtxt(val_file, dtype=int)
        test_idx = np.loadtxt(test_file, dtype=int)

    return {
        "train_idx": train_idx,  # Exemple : [0, 3, 7, 11, ...]
        "val_idx": val_idx,      # Exemple : [1, 5, 9, 13, ...]
        "test_idx": test_idx     # Exemple : [2, 6, 10, 14, ...]
    }''', language='python')

st.markdown('<div class="key-point">✅ Points clés :</div>', unsafe_allow_html=True)
st.markdown("""
- Retourne les **numéros de ligne des données brutes** (indices de position)
- Tous les modèles chargent **exactement les mêmes** fichiers txt
- Les valeurs d'indices sont **absolues**, par exemple `val_idx[0] = 1` signifie la ligne 1 des données brutes
""")

st.markdown('<div class="subsection-header">1.2 Vérification de la Signature du Split</div>', unsafe_allow_html=True)

st.markdown("Utilise un hash SHA256 pour garantir que tous les modèles utilisent exactement le même découpage de données.")

with st.expander("📄 Voir le code source : src/data/split_manager.py (lignes 127-152)"):
    st.code('''def split_signature(splits: Dict[str, np.ndarray]) -> str:
    """Calcule la signature SHA256 des splits"""
    train_idx = np.sort(splits["train_idx"])
    val_idx = np.sort(splits["val_idx"])
    test_idx = np.sort(splits["test_idx"])

    combined = np.concatenate([train_idx, val_idx, test_idx])
    content = combined.tobytes()

    hash_obj = hashlib.sha256(content)
    signature = hash_obj.hexdigest()[:16]  # "cf53f8eb169b3531"

    return signature''', language='python')

st.markdown('<div class="key-point">✅ Mécanisme de garantie :</div>', unsafe_allow_html=True)
st.markdown("""
- Tous les modèles doivent produire la **même signature**
- La signature est stockée dans les fichiers d'export et vérifiée lors de la fusion
- Signature actuelle du projet : `cf53f8eb169b3531`
""")

# ========== Deuxième Garantie : Chaîne de Traçabilité des Indices ==========
st.markdown('<div class="section-header">🔗 Deuxième Garantie : Chaîne de Traçabilité des Indices (Flux de Données Complet)</div>', unsafe_allow_html=True)

st.markdown('<div class="subsection-header">2.1 Flux des Indices pour les Modèles Image</div>', unsafe_allow_html=True)

st.markdown("Processus complet de traçabilité des indices depuis le chargement des données jusqu'à l'export des prédictions :")

# Étape 1 : Chargement des données brutes
with st.expander("Étape 1️⃣ : Chargement des données brutes (src/data/data_colab.py)"):
    st.code('''# === Étape 1 : Chargement des données brutes ===
pack = load_data_colab(raw_dir=..., splitted=False)
X = pack["X"]  # DataFrame, shape: (84916, n_cols)
y = pack["y"]  # Labels bruts

# État des données brutes :
# Index:  0      1      2      3      4      5    ...  84915
# Image:  img0   img1   img2   img3   img4   img5 ...  img84915
# Label:  10     40     10     50     40     60   ...  ...''', language='python')
    st.markdown("**Source** : `src/data/data_colab.py` (lignes 22-76)")

# Étape 2 : Chargement des splits unifiés
with st.expander("Étape 2️⃣ : Chargement des splits unifiés (src/data/split_manager.py)"):
    st.code('''# === Étape 2 : Chargement des splits unifiés ===
splits = load_splits(verbose=True)

# splits["train_idx"] = [0, 3, 4, 7, 8, ...]  # Numéros de ligne des données brutes
# splits["val_idx"]   = [1, 5, 9, 13, ...]
# splits["test_idx"]  = [2, 6, 10, 14, ...]''', language='python')
    st.markdown("**Source** : `src/data/split_manager.py` (lignes 23-82)")
    st.markdown("**Point clé** : Ces indices sont les **numéros de ligne absolus** du DataFrame de données brutes")

# Étape 3 : Encodage des labels
with st.expander("Étape 3️⃣ : Encodage des labels (ordre préservé) (src/train/image_convnext.py)"):
    st.code('''# === Étape 3 : Encodage des labels (ordre préservé) ===
y_encoded = encode_labels(y, CANONICAL_CLASSES).astype(int)
df_full = X.copy()
df_full["encoded_label"] = y_encoded

# État de df_full (ordre inchangé) :
# Index:  0      1      2      3      4      5    ...  84915
# Image:  img0   img1   img2   img3   img4   img5 ...  img84915
# Label:  5      15     5      20     15     25   ...  ...  (après encodage)''', language='python')
    st.markdown("**Source** : `src/train/image_convnext.py` (lignes 503-508)")
    st.markdown('<div class="key-point">✅ Point clé : L\'encodage des labels ne modifie pas l\'ordre des lignes du DataFrame</div>', unsafe_allow_html=True)

# Étape 4 : Création du Dataset complet
with st.expander("Étape 4️⃣ : Création du Dataset complet (non découpé) (src/train/image_convnext.py)"):
    st.code('''# === Étape 4 : Création du Dataset complet (non découpé) ===
full_dataset = RakutenImageDataset(
    dataframe=df_full.reset_index(drop=True),  # Réinitialise seulement l'index interne, pas l'ordre
    image_dir=img_dir,
    transform=transforms,
    label_col="encoded_label",
)

# full_dataset[i] correspond à df_full.iloc[i] correspond à la ligne i des données brutes''', language='python')
    st.markdown("**Source** : `src/train/image_convnext.py` (lignes 245-258)")
    st.markdown("**Note** : `RakutenImageDataset` est défini dans `src/data/image_dataset.py`")

# Étape 5 : Enveloppe IndexedDataset
with st.expander("Étape 5️⃣ : Enveloppe avec IndexedDataset (src/train/image_convnext.py)"):
    st.code('''# === Étape 5 : Enveloppe avec IndexedDataset ===
train_subset = IndexedDataset(full_dataset, indices=splits["train_idx"])
val_subset = IndexedDataset(full_dataset, indices=splits["val_idx"])

# Principe de fonctionnement d'IndexedDataset :
class IndexedDataset(Dataset):
    """Enveloppe de dataset qui préserve les indices d'origine"""
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices).astype(int)

    def __getitem__(self, i: int):
        real_idx = int(self.indices[i])
        img, label = self.base[real_idx]
        return img, label, real_idx  # ← Retourne l'index d'origine !

# train_subset[0] accède réellement à full_dataset[splits["train_idx"][0]]
# Retourne (image, label, index d'origine)''', language='python')
    st.markdown("**Source** : `src/train/image_convnext.py` (lignes 104-120, 263-265)")
    st.markdown('<div class="key-point">✅ Mécanisme central : IndexedDataset trace et retourne toujours le numéro de ligne des données brutes</div>', unsafe_allow_html=True)

# Étape 6 : Export des prédictions
with st.expander("Étape 6️⃣ : Export des prédictions (src/train/image_convnext.py)"):
    st.code('''# === Étape 6 : Export des prédictions ===
export_idx = splits["val_idx"]  # [1, 5, 9, 13, ...]

# Prédiction sur val_subset, ordre des probs retournés :
probs, seen_idx = _predict_probs_with_real_idx(
    model=model,
    base_dataset=full_dataset,
    indices=export_idx,  # [1, 5, 9, 13, ...]
    ...
)

# probs[0] correspond à la prédiction pour la ligne 1 des données brutes
# probs[1] correspond à la prédiction pour la ligne 5 des données brutes
# probs[2] correspond à la prédiction pour la ligne 9 des données brutes
# seen_idx = [1, 5, 9, 13, ...]  (identique à export_idx)''', language='python')
    st.markdown("**Source** : `src/train/image_convnext.py` (lignes 426-462, 688-695)")

# Étape 7 : Sauvegarde en .npz
with st.expander("Étape 7️⃣ : Sauvegarde en .npz (src/export/model_exporter.py)"):
    st.code('''# === Étape 7 : Sauvegarde en .npz ===
export_predictions(
    idx=seen_idx,                    # [1, 5, 9, 13, ...]
    probs=probs,                     # shape (n_val, 27)
    y_true=y_encoded[seen_idx],      # Labels aux positions correspondantes des données brutes
    split_signature=sig,             # "cf53f8eb169b3531"
    ...
)

# Contenu du fichier .npz :
# idx:    [1, 5, 9, 13, ...]        ← Numéros de ligne des données brutes
# probs:  [[0.1, 0.2, ...],         ← Probabilités de prédiction pour la ligne 1
#          [0.3, 0.1, ...],         ← Probabilités de prédiction pour la ligne 5
#          [0.2, 0.4, ...],         ← Probabilités de prédiction pour la ligne 9
#          ...]
# y_true: [15, 25, 20, ...]         ← Labels réels correspondants''', language='python')
    st.markdown("**Source** : `src/export/model_exporter.py` (lignes 22-150)")
    st.markdown("**Et aussi** : `src/train/image_convnext.py` (lignes 707-738)")

st.markdown('<div class="subsection-header">2.2 Flux des Indices pour les Modèles Texte (Mécanisme Identique)</div>', unsafe_allow_html=True)

st.markdown("Les modèles Texte suivent exactement le même mécanisme de traçabilité des indices :")

# Modèles Texte Étapes 1-3
with st.expander("Étapes 1️⃣-3️⃣ : Chargement et prétraitement des données (src/train/text_camembert.py)"):
    st.code('''# === Étape 1 : Chargement des données brutes ===
pack = load_data_colab(raw_dir=..., splitted=False)
X = pack["X"]  # DataFrame, shape: (84916, n_cols)
y = pack["y"]

# === Étape 2 : Chargement des mêmes splits ===
splits = load_splits(verbose=True)
# Retourne exactement les mêmes tableaux d'indices (cohérents avec les modèles Image)

# === Étape 3 : Construction de la colonne texte (point clé : pas de suppression de lignes) ===
X["text"] = build_text_column(X, "designation", "description")

# Traitement des textes vides : remplissage au lieu de suppression
valid_mask = X["text"].str.len() > 0
if not valid_mask.all():
    X.loc[~valid_mask, "text"] = "[EMPTY]"  # ✅ Préserve toutes les lignes

# État de X (nombre de lignes inchangé) :
# Index: 0      1      2         3      4    ...  84915
# Text:  "..."  "..."  "[EMPTY]" "..."  "..." ...  ...''', language='python')
    st.markdown("**Source** : `src/train/text_camembert.py` (lignes 186-230)")
    st.markdown('<div class="key-point">✅ Principe clé : Traiter les textes vides par remplissage plutôt que suppression, préserver la longueur du DataFrame</div>', unsafe_allow_html=True)

# Modèles Texte Étapes 4-5
with st.expander("Étapes 4️⃣-5️⃣ : Encodage des labels et découpage des données (src/train/text_camembert.py)"):
    st.code('''# === Étape 4 : Encodage des labels (ordre préservé) ===
y_encoded = encode_labels(y, CANONICAL_CLASSES).astype(int)
X["label"] = y_encoded

# État de X (ordre complètement inchangé) :
# Index: 0      1      2         3      4    ...  84915
# Text:  "..."  "..."  "[EMPTY]" "..."  "..." ...  ...
# Label: 5      15     5         20     15   ...  ...

# === Étape 5 : Découpage avec les indices d'origine ===
train_df = X.iloc[splits["train_idx"]].copy().reset_index(drop=True)
val_df = X.iloc[splits["val_idx"]].copy().reset_index(drop=True)

# Note : reset_index(drop=True) n'agit que sur le sous-ensemble après découpage
# Cela n'affecte pas notre traçabilité des indices d'origine

# Index interne de train_df : 0, 1, 2, 3, ...  (continu)
# Mais nous conservons splits["train_idx"] qui connaît les positions d'origine''', language='python')
    st.markdown("**Source** : `src/train/text_camembert.py` (lignes 234-247)")

# Modèles Texte Étapes 6-8
with st.expander("Étapes 6️⃣-8️⃣ : Entraînement et export (src/train/text_camembert.py)"):
    st.code('''# === Étape 6 : Création du Dataset HF ===
train_ds = Dataset.from_dict({
    "text": train_df["text"].tolist(),
    "label": train_df["label"].tolist(),
})

# train_ds[0] correspond à train_df.iloc[0]
# correspond à X.iloc[splits["train_idx"][0]]
# correspond à la ligne splits["train_idx"][0] des données brutes

# === Étape 7 : Prédiction après entraînement ===
export_idx = splits["val_idx"]  # [1, 5, 9, 13, ...]
export_ds = val_ds

predictions = trainer.predict(export_ds)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# Ordre des probs :
# probs[0] correspond à val_ds[0]
#          correspond à val_df.iloc[0]
#          correspond à X.iloc[splits["val_idx"][0]]
#          correspond à la ligne 1 des données brutes

# === Étape 8 : Export (sauvegarde des indices d'origine) ===
y_true = y_encoded[export_idx].astype(int)  # Utilise les indices d'origine pour obtenir les labels

export_predictions(
    idx=export_idx,         # [1, 5, 9, 13, ...] ← Indices d'origine
    probs=probs,            # shape (n_val, 27)
    y_true=y_true,
    split_signature=sig,
    ...
)''', language='python')
    st.markdown("**Source** : `src/train/text_camembert.py` (lignes 252-400 environ)")

# ========== Troisième Garantie : Mécanisme de Vérification des Indices ==========
st.markdown('<div class="section-header">🎯 Troisième Garantie : Mécanisme de Vérification des Indices</div>', unsafe_allow_html=True)

st.markdown('<div class="subsection-header">3.1 Vérification de Longueur</div>', unsafe_allow_html=True)
st.code('''# image_convnext.py
if len(probs) != len(export_idx):
    raise AssertionError(
        f"Longueur probs ({len(probs)}) "
        f"!= longueur export_idx "
        f"({len(export_idx)})"
    )''', language='python')
st.caption("Source : src/train/image_convnext.py")

st.markdown('<div class="subsection-header">3.2 Vérification d\'Ordre</div>', unsafe_allow_html=True)
st.code('''# image_convnext.py:698-699
if not np.array_equal(seen_idx, export_idx):
    raise AssertionError(
        "Désalignement d'ordre d'index "
        "lors de l'inférence d'export"
    )''', language='python')
st.caption("Source : src/train/image_convnext.py (ligne 698)")

st.markdown('<div class="subsection-header">3.3 Vérification de Signature</div>', unsafe_allow_html=True)
st.code('''# Sauvegarde de la signature lors de l'export
export_predictions(
    split_signature=sig,
    # "cf53f8eb169b3531"
    ...
)

# Vérification de la signature lors du chargement
loaded = load_predictions(
    npz_path=...,
    verify_split_signature=sig,
    ...
)''', language='python')
st.caption("Source : src/export/model_exporter.py")

# ========== Quatrième Garantie : Alignement lors de la Fusion ==========
st.markdown('<div class="section-header">🔄 Quatrième Garantie : Alignement lors de la Fusion</div>', unsafe_allow_html=True)

st.markdown('<div class="subsection-header">4.1 Format des Fichiers d\'Export</div>', unsafe_allow_html=True)

st.markdown("**Structure du fichier NPZ**")
st.code('''{
    "idx": np.array([1, 5, 9, 13, ...]),     # Numéros de ligne des données brutes
    "probs": np.array([[...], [...], ...]),  # Matrice de probabilités
    "y_true": np.array([15, 25, 20, ...]),   # Labels réels
}''', language='python')

st.markdown("**Métadonnées JSON**")
st.code('''{
    "model_name": "convnext_canonical",
    "split_name": "val",
    "split_signature": "cf53f8eb169b3531",
    "classes_fp": "cdfa70b13f7390e6",
    "num_samples": 10827,
    ...
}''', language='json')

st.markdown('<div class="subsection-header">4.2 Processus d\'Alignement pour la Fusion</div>', unsafe_allow_html=True)

with st.expander("📄 Voir l'exemple de code d'alignement pour la fusion"):
    st.code('''# === Chargement des prédictions des modèles Image ===
img1 = load_predictions("convnext_canonical/val.npz", verify_split_signature=sig)
img2 = load_predictions("swin_canonical/val.npz", verify_split_signature=sig)

# img1["idx"] = [1, 5, 9, 13, ...] ← Indices d'origine
# img2["idx"] = [1, 5, 9, 13, ...] ← Exactement identiques

# === Chargement des prédictions des modèles Texte ===
text1 = load_predictions("camembert_canonical/val.npz", verify_split_signature=sig)
text2 = load_predictions("xlmr_canonical/val.npz", verify_split_signature=sig)

# text1["idx"] = [1, 5, 9, 13, ...] ← Exactement identiques
# text2["idx"] = [1, 5, 9, 13, ...] ← Exactement identiques

# === Vérification de l'alignement ===
assert np.array_equal(img1["idx"], img2["idx"])
assert np.array_equal(img1["idx"], text1["idx"])
assert np.array_equal(img1["idx"], text2["idx"])

# ✅ Les idx de tous les modèles sont complètement identiques

# === Fusion des probabilités (correspondance par ligne) ===
# Puisque les idx sont complètement identiques, on peut fusionner directement par ligne
blended_probs = (
    0.3 * img1["probs"] +      # Ligne i = prédiction pour la ligne idx[i] des données brutes
    0.3 * img2["probs"] +      # Ligne i = prédiction pour la ligne idx[i] des données brutes
    0.2 * text1["probs"] +     # Ligne i = prédiction pour la ligne idx[i] des données brutes
    0.2 * text2["probs"]       # Ligne i = prédiction pour la ligne idx[i] des données brutes
)

# blended_probs[0] = prédiction fusionnée pour la ligne 1 des données brutes
# blended_probs[1] = prédiction fusionnée pour la ligne 5 des données brutes''', language='python')

# ========== Principes Fondamentaux de Garantie ==========
st.markdown('<div class="section-header">🎓 Principes Fondamentaux de Garantie</div>', unsafe_allow_html=True)

principles = [
    {
        "title": "1. Invariance des Indices d'Origine",
        "content": [
            "Après le chargement des données brutes, chaque ligne a un indice de position fixe (0 à 84915)",
            "Toutes les opérations sont basées sur cet indice d'origine"
        ]
    },
    {
        "title": "2. Caractère Absolu du Split",
        "content": [
            "splits['val_idx'] = [1, 5, 9, ...] sont les numéros de ligne absolus des données brutes",
            "Pas des indices relatifs à un sous-ensemble"
        ]
    },
    {
        "title": "3. Transmission Explicite des Indices",
        "content": [
            "À chaque étape (découpage, entraînement, prédiction, export), les indices d'origine sont tracés explicitement",
            "Sauvegarde du tableau idx lors de l'export"
        ]
    },
    {
        "title": "4. Principe de Non-Suppression des Données",
        "content": [
            "Traiter les valeurs manquantes par remplissage plutôt que par suppression",
            "Préserver la longueur du DataFrame"
        ]
    },
    {
        "title": "5. Vérification par Signature",
        "content": [
            "Utiliser la signature SHA256 pour garantir que tous les modèles utilisent exactement les mêmes splits",
            "Vérification obligatoire de la cohérence des signatures lors de la fusion"
        ]
    }
]

for i, principle in enumerate(principles):
    with st.container():
        st.markdown(f'<div class="key-point"><b>{principle["title"]}</b></div>', unsafe_allow_html=True)
        for content in principle["content"]:
            st.markdown(f"- {content}")

# ========== Résumé ==========
st.markdown('<div class="section-header">📊 Tableau Comparatif Récapitulatif</div>', unsafe_allow_html=True)

comparison_data = {
    "Opération": [
        "Prétraitement des données",
        "Génération du Split",
        "Traçabilité des indices",
        "Format d'export",
        "Vérification par signature",
        "Alignement pour fusion"
    ],
    "Code Sans Risque (Image & Text 09-12)": [
        "✅ Remplissage des valeurs vides, préservation de toutes les lignes",
        "✅ load_splits() unifié",
        "✅ Préservation constante des numéros de ligne d'origine",
        "✅ .npz + idx + probs + y_true",
        "✅ Vérification par signature SHA256",
        "✅ idx complètement identiques, fusion directe"
    ]
}

import pandas as pd
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ========== Explication Finale ==========
st.success("""
**Grâce à ce mécanisme de garantie complet**, même après un processus d'entraînement complexe,
les résultats de prédiction exportés peuvent toujours correspondre avec précision à chaque ligne
des données d'origine via les idx, réalisant un **alignement parfait** et une **fusion sécurisée**
des prédictions des modèles Image et Texte.

C'est la clé du succès de notre Ensemble multi-modal !
""")

# Footer
st.markdown("---")
st.caption("📝 Cette documentation est générée automatiquement à partir du code source du projet | Rakuten Product Classification Project")
