import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, f1_score
from data import CATEGORY_SHORT_NAMES


def plot_classification_report(
    y_true,
    y_pred,
    encoder=None,
    k_worst_f1=5,
    k_worst_errors=5,
    cmap="Blues",
):
    """
    Génère un rapport de classification complet avec visualisations.

    Cette fonction affiche :
    - le score F1 pondéré global,
    - la matrice de confusion normalisée,
    - les classes ayant les pires scores F1,
    - les erreurs de classification les plus fréquentes.

    Elle est conçue pour l'analyse qualitative des performances
    d'un modèle de classification multi-classes.

    Args:
        y_true (array-like): Labels réels.
        y_pred (array-like): Labels prédits par le modèle.
        encoder (optional): Encodeur de labels (ex: LabelEncoder)
            permettant de convertir les indices de classes en labels.
        k_worst_f1 (int or None, optional): Nombre de classes avec
            les pires scores F1 à afficher. Si None, affiche toutes
            les classes. Par défaut 5.
        k_worst_errors (int, optional): Nombre des confusions les
            plus fréquentes à afficher. Par défaut 5.
        cmap (str, optional): Colormap utilisée pour les heatmaps.

    Returns:
        dict: Classification report complet au format dictionnaire
        (sortie de sklearn.metrics.classification_report).
    """

    # =====================
    # Score global
    # =====================
    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print(f"F1 weighted score : {f1_weighted:.3f}")

    # =====================
    # Décodage des labels
    # =====================
    if encoder is not None:
        y_true_decoded = encoder.inverse_transform(y_true)
        y_pred_decoded = encoder.inverse_transform(y_pred)
    else:
        y_true_decoded = y_true
        y_pred_decoded = y_pred

    y_true_labels = [CATEGORY_SHORT_NAMES[c] for c in y_true_decoded]
    y_pred_labels = [CATEGORY_SHORT_NAMES[c] for c in y_pred_decoded]

    # =====================
    # Classification report
    # =====================
    report_dict = classification_report(
        y_true_labels,
        y_pred_labels,
        output_dict=True,
        zero_division=0,
    )

    print(f"Accuracy : {report_dict['accuracy']:.2f}")

    report_df = (
        pd.DataFrame(report_dict)
        .T.iloc[:-3]
        .sort_values("f1-score", ascending=False)
    )

    # =====================
    # Matrice de confusion normalisée
    # =====================
    confusion_matrix_norm = pd.crosstab(
        y_true_labels,
        y_pred_labels,
        normalize=0
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        confusion_matrix_norm,
        cmap=cmap,
        vmin=0,
        vmax=1
    )
    plt.title("Matrice de confusion normalisée par ligne")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # =====================
    # Pires classes (F1)
    # =====================
    if k_worst_f1 is None:
        k_worst_f1 = len(report_df)

    display(
        report_df.iloc[-k_worst_f1:].round(3)
    )

    # =====================
    # Pires confusions
    # =====================
    cm_array = confusion_matrix_norm.to_numpy()
    np.fill_diagonal(cm_array, 0)

    top_indices = np.argsort(cm_array, axis=None)[-k_worst_errors:][::-1]
    true_idx, pred_idx = np.unravel_index(top_indices, cm_array.shape)

    error_rates = [
        round(cm_array[i, j] * 100, 1)
        for i, j in zip(true_idx, pred_idx)
    ]

    worst_confusions_df = pd.DataFrame({
        "Classe réelle": confusion_matrix_norm.index[true_idx],
        "Classe prédite": confusion_matrix_norm.columns[pred_idx],
        "% des prédictions de la classe réelle": error_rates,
    })

    display(worst_confusions_df)
