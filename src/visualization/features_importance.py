import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import CATEGORY_SHORT_NAMES

def plot_features_importance(
    model_coef,
    feature_blocks,
    encoder=None,
    cmap="Blues",
    by_block=True
):
    """
    Visualise l'importance des features d'un modèle linéaire par blocs de features.

    Cette fonction est conçue pour analyser des pipelines complexes (notamment
    multimodaux) où les features sont concaténées par blocs homogènes
    (HOG, BoVW, couleurs, texte, etc.).  
    L'importance est mesurée à partir de la valeur absolue des coefficients
    du modèle (ex. régression logistique).

    Deux niveaux de visualisation sont proposés :
    1. Une heatmap agrégée montrant la moyenne des coefficients absolus
       par bloc de features.
    2. (Optionnel) Une heatmap détaillée pour chaque bloc individuellement.

    Args:
        model_coef (np.ndarray):
            Matrice des coefficients du modèle de forme
            (n_classes, n_features).
        feature_blocks (dict[str, range]):
            Dictionnaire associant le nom de chaque bloc de features
            à l'intervalle des indices correspondants dans le vecteur
            de features final.
        encoder (optional):
            Encodeur de labels (ex. LabelEncoder) permettant de
            convertir les indices de classes en noms lisibles.
            Si None, les indices numériques sont utilisés.
        cmap (str, optional):
            Colormap utilisée pour les heatmaps. Par défaut "Blues".
        by_block (bool, optional):
            Si True, génère une visualisation détaillée pour chaque
            bloc de features en plus de la vue agrégée. Par défaut True.

    Returns:
        None:
            La fonction affiche directement les figures matplotlib
            et ne retourne aucun objet.
    """
    class_labels = range(model_coef.shape[0])

    if encoder:
        class_labels = encoder.inverse_transform(class_labels)
        class_labels = [CATEGORY_SHORT_NAMES[c] for c in class_labels]

    abs_coef = np.abs(model_coef)
    max_abs_coef = abs_coef.max()

    mean_abs_coef_by_block = np.array([
        abs_coef[:, feature_range].mean(axis=1)
        for feature_range in feature_blocks.values()
    ]).T

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        mean_abs_coef_by_block,
        vmin=0,
        vmax=max_abs_coef,
        cmap=cmap
    )
    plt.title("Importance moyenne des features par bloc")
    plt.xticks(
        np.arange(len(feature_blocks)) + 0.5,
        labels=feature_blocks.keys()
    )
    plt.yticks(
        np.arange(len(class_labels)) + 0.5,
        labels=class_labels
    )
    plt.show()

    if by_block:
        for block_name, feature_range in feature_blocks.items():
            if len(feature_range) > 1:
                plt.figure(figsize=(14, 6))
                sns.heatmap(
                    abs_coef[:, feature_range],
                    vmin=0,
                    vmax=max_abs_coef,
                    cmap=cmap
                )
                plt.title(f"Importance des features – bloc {block_name}")
                plt.yticks(
                    np.arange(len(class_labels)) + 0.5,
                    labels=class_labels
                )
                plt.show()