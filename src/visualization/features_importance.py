import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import CATEGORY_SHORT_NAMES

def plot_features_importance(
    model_coef,
    feature_blocks,
    classes,
    cmap="Blues",
    by_block=True,
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
    class_labels = [CATEGORY_SHORT_NAMES[c] for c in classes]

    abs_coef = np.abs(model_coef)
    max_abs_coef = abs_coef.max()

    mean_abs_coef_by_block = np.array([
        abs_coef[:, feature_range].mean(axis=1)
        for feature_range in feature_blocks.values()
    ]).T

    plt.figure(figsize=(14, 6))
    vmax = max_abs_coef if by_block else None
    sns.heatmap(
        mean_abs_coef_by_block,
        vmin=0,
        vmax=vmax,
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

def plot_features_importance(
    model_coef,
    feature_spec=None,
    classes=None,
    cmap="Blues",
    by_block=True,
):
    """
    Visualise l'importance des features d'un modèle de classification linéaire.

    Cette fonction permet d'analyser l'influence relative des features à partir
    des coefficients d'un modèle linéaire (régression logistique, SVM linéaire).
    L'importance est mesurée par la valeur absolue des coefficients.

    Le paramètre ``feature_spec`` permet de gérer différents niveaux
    d'organisation des features :

    - ``None`` : aucune information sur les features n'est affichée sur l'axe
      des abscisses.
    - ``list[str]`` : chaque feature est considérée individuellement et les
      labels fournis sont affichés.
    - ``dict[str, range]`` : les features sont regroupées par blocs homogènes
      (ex. texte, couleur, HOG, etc.) et l'importance est agrégée par bloc à
      l'aide de la moyenne des coefficients absolus.

    Args:
        model_coef (np.ndarray):
            Matrice des coefficients du modèle de forme
            ``(n_classes, n_features)``.
        feature_spec (None | list[str] | dict[str, range], optional):
            Spécification de l'organisation des features.
            - ``None`` : aucune étiquette de feature.
            - ``list[str]`` : noms des features individuelles.
            - ``dict[str, range]`` : dictionnaire associant chaque nom de bloc
              à l'intervalle d'indices correspondant dans le vecteur de features.
            Par défaut ``None``.
        classes (array-like, optional):
            Identifiants des classes correspondant aux lignes de
            ``model_coef``. S'ils sont fournis, ils sont convertis en noms de
            catégories lisibles via ``CATEGORY_SHORT_NAMES``.
            Par défaut ``None``.
        cmap (str, optional):
            Colormap utilisée pour les heatmaps. Par défaut ``"Blues"``.
        by_block (bool, optional):
            Si ``True`` et si ``feature_spec`` est un dictionnaire, génère en
            plus des visualisations détaillées pour chaque bloc de features.
            Par défaut ``True``.

    Returns:
        None:
            La fonction affiche directement les figures matplotlib et ne
            retourne aucun objet.
    """

    # === Labels des classes ===
    if classes is None:
        class_names = range(model_coef.shape[0])
    else:
        class_names = [CATEGORY_SHORT_NAMES[c] for c in classes]

    abs_coef = np.abs(model_coef)

    # === Gestion de la spécification des features ===
    if feature_spec is None:
        feature_mode = "none"
        importance_matrix = abs_coef
        x_labels = None

    elif isinstance(feature_spec, dict):
        feature_mode = "blocks"
        importance_matrix = np.array([
            abs_coef[:, feature_range].mean(axis=1)
            for feature_range in feature_spec.values()
        ]).T
        x_labels = list(feature_spec.keys())

    else:
        feature_mode = "features"
        importance_matrix = abs_coef
        x_labels = feature_spec

        if len(x_labels) != model_coef.shape[1]:
            raise ValueError(
                "La longueur de feature_spec ne correspond pas au nombre "
                "de features du modèle."
            )

    # === Heatmap principale ===
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        importance_matrix,
        cmap=cmap,
        vmin=0,
    )
    plt.title("Importance des features")

    if x_labels is not None:
        plt.xticks(
            np.arange(len(x_labels)) + 0.5,
            labels=x_labels,
            rotation=45,
            ha="right",
        )
    else:
        plt.xticks([])

    plt.yticks(
        np.arange(len(class_names)) + 0.5,
        labels=class_names,
    )
    plt.show()

    # === Visualisations détaillées par bloc ===
    if feature_mode == "blocks" and by_block:
        vmax = abs_coef.max()
        for block_name, feature_range in feature_spec.items():
            if len(feature_range) > 1:
                plt.figure(figsize=(14, 6))
                sns.heatmap(
                    abs_coef[:, feature_range],
                    vmin=0,
                    vmax=vmax,
                    cmap=cmap,
                )
                plt.title(f"Importance des features – bloc : {block_name}")
                plt.yticks(
                    np.arange(len(class_names)) + 0.5,
                    labels=class_names,
                )
                plt.show()
