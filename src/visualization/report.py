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
    title="Matrice de confusion normalisée par classe réelle",
    save=False,
):
    """
    Génère un rapport de classification détaillé avec métriques globales
    et visualisations ciblées des erreurs.

    Cette fonction est conçue pour l'analyse approfondie des performances
    d'un classifieur multi-classes. Elle combine des métriques globales,
    une matrice de confusion normalisée et des tableaux d'analyse des classes
    et confusions les plus problématiques.

    Les éléments suivants sont affichés :
    - le F1-score pondéré global et l'accuracy,
    - une matrice de confusion normalisée par classe réelle,
      avec annotation uniquement des confusions les plus fréquentes,
    - la liste des classes avec les plus faibles scores F1,
    - la liste des classes avec les meilleurs scores F1 (optionnel),
    - un tableau récapitulatif des confusions les plus fréquentes.

    Les labels peuvent être fournis sous forme encodée (entiers) ou déjà décodée.
    Lorsque `encoded=True`, les labels sont automatiquement décodés et
    mappés vers des noms de classes courts pour l'affichage.

    Args:
        y_true (array-like):
            Labels réels des échantillons (encodés ou non).
        y_pred (array-like):
            Labels prédits par le modèle (encodés ou non).
        k_worst_f1 (int or None, default=5):
            Nombre de classes affichées avec les plus faibles scores F1.
            Si None, cette analyse est désactivée.
        k_best_f1 (int or None, default=None):
            Nombre de classes affichées avec les meilleurs scores F1.
            Si None, cette analyse est désactivée.
        k_worst_errors (int or None, default=5):
            Nombre de confusions inter-classes les plus fréquentes à analyser
            et à annoter dans la matrice de confusion.
        cmap (str, default="Blues"):
            Colormap utilisée pour la heatmap de la matrice de confusion.
        encoded (bool, default=True):
            Indique si les labels fournis sont encodés numériquement.
            Si True, un décodage automatique est appliqué.
        title (str or None, default="Matrice de confusion normalisée par classe réelle"):
            Titre affiché au-dessus de la matrice de confusion.
            Si None, aucun titre n'est affiché.
        save (bool, default=False):
            Si True, la matrice de confusion est sauvegardée au format PNG
            sous le nom `conf_matrix.png`.

    Returns:
        dict:
            Dictionnaire du classification report généré par sklearn
            (output_dict=True), contenant precision, recall, F1-score
            et support pour chaque classe.
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

    if encoded:
        y_true = decode_labels(y_true)
        y_pred = decode_labels(y_pred)

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
    
    cm_array = confusion_matrix_norm.to_numpy()
    # Identification des pires confusions (hors diagonale)
    cm_no_diag = cm_array.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    top_indices = np.argsort(cm_no_diag, axis=None)[-k_worst_errors:]
    true_idx, pred_idx = np.unravel_index(top_indices, cm_no_diag.shape)
    
    fig = plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        confusion_matrix_norm,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=True
    )
    
    # Annotation uniquement des pires confusions
    for i, j in zip(true_idx, pred_idx):
        value = cm_array[i, j]
        if value > 0:
            ax.text(
                j + 0.5,
                i + 0.5,
                # f"{value:.2f}",
                f"{value*100:.0f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",
            )

    # =====================
    # Mise en gras des labels impliqués dans les annotations
    # =====================
    
    rows_to_bold = set(true_idx)
    cols_to_bold = set(pred_idx)
    
    # Axe Y (classes réelles)
    for idx, label in enumerate(ax.get_yticklabels()):
        if idx in rows_to_bold:
            label.set_fontweight("bold")
    
    # Axe X (classes prédites)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx in cols_to_bold:
            label.set_fontweight("bold")

    # plt.title("Matrice de confusion normalisée par classe réelle")
    plt.xlabel("")
    plt.ylabel("")
    if title:
        plt.title(title)
    if save:
        fig.savefig('conf_matrix.png', bbox_inches="tight")
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