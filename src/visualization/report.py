from IPython.display import display, HTML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, f1_score
from data import CATEGORY_SHORT_NAMES, decode_labels
from .display import beautiful_print



def display_global_metrics(f1_weighted, accuracy):
    beautiful_print(f"""
        <b>Performance globale du modèle</b><br><br>
    
        • <b>F1-score pondéré</b> : <code>{f1_weighted:.3f}</code><br>
        • <b>Accuracy</b> : <code>{accuracy:.3f}</code>
    """)


def plot_classification_report(
    y_true,
    y_pred,
    k_worst_f1=5,
    k_best_f1=None,
    k_worst_errors=5,
    cmap="Blues",
    encoded=True,
):
    """
    Génère un rapport de classification complet avec visualisations.

    Cette fonction affiche :
    - le score F1 pondéré global,
    - la matrice de confusion normalisée,
    - les classes avec les pires scores F1,
    - les classes avec les meilleurs scores F1 (optionnel),
    - les confusions les plus fréquentes.

    Args:
        y_true (array-like): Labels réels.
        y_pred (array-like): Labels prédits.
        encoder (optional): Encodeur de labels.
        k_worst_f1 (int or None): Nombre de classes avec les pires F1.
        k_best_f1 (int or None): Nombre de classes avec les meilleurs F1.
        k_worst_errors (int or None): Nombre de confusions les plus fréquentes.
        cmap (str): Colormap pour les heatmaps.

    Returns:
        dict: Classification report complet (sklearn).
    """

    # =====================
    # Score global
    # =====================
    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # =====================
    # Décodage des labels
    # =====================

    # Décodage éventuel
    if encoded:
        y_true = decode_labels(y_true)
        y_pred = decode_labels(y_pred)

    # Mapping vers noms courts
    y_true = [CATEGORY_SHORT_NAMES[c] for c in y_true]
    y_pred = [CATEGORY_SHORT_NAMES[c] for c in y_pred]

    # =====================
    # Classification report
    # =====================
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report_dict)
        .T.iloc[:-3]
        .sort_values("f1-score", ascending=False)
    )

    display_global_metrics(f1_weighted, report_dict['accuracy'])

    # =====================
    # Matrice de confusion normalisée
    # =====================
    confusion_matrix_norm = pd.crosstab(
        y_true,
        y_pred,
        normalize=0
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        confusion_matrix_norm,
        cmap=cmap,
        vmin=0,
        vmax=1
    )
    plt.title("Matrice de confusion normalisée par classe réelle")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # =====================
    # Meilleures classes (F1)
    # =====================
    if k_best_f1 is not None:
        print(f"\nClasses avec les meilleurs scores F1 (top {k_best_f1})")
        display(report_df.head(k_best_f1).round(3))

    # =====================
    # Pires classes (F1)
    # =====================
    if k_worst_f1 is not None:
        print(f"\nClasses avec les plus faibles scores F1 (bottom {k_worst_f1})")
        display(report_df.tail(k_worst_f1).round(3))

    # =====================
    # Confusions les plus fréquentes
    # =====================
    if k_worst_errors is not None:
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

        print(f"\nConfusions les plus fréquentes (top {k_worst_errors})")
        display(worst_confusions_df)
