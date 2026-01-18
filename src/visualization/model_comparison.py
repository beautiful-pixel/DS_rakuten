import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from data import CATEGORY_SHORT_NAMES, decode_labels


def plot_f1_comparison_with_delta(
    y_true,
    y_pred_new,
    y_pred_ref,
    new_name="Nouveau modèle",
    ref_name="Modèle de référence",
    encoded=True,
):
    """
    Compare deux modèles de classification multi-classes à l’aide
    des F1-scores par catégorie et visualise les écarts de performance.

    Cette fonction génère un graphique horizontal comparant, pour chaque classe :
    - le F1-score du modèle de référence ;
    - le F1-score du nouveau modèle ;
    - le delta de F1-score (nouveau − référence), annoté sur le graphique.

    Les classes sont triées selon le F1-score du modèle de référence,
    ce qui permet d’analyser l’impact du nouveau modèle en priorité
    sur les catégories initialement les plus difficiles.

    En complément, les F1-scores pondérés globaux des deux modèles
    sont calculés et affichés dans le titre du graphique afin de
    fournir une vision synthétique des performances globales.

    Les labels peuvent être fournis sous forme encodée (entiers)
    ou déjà décodée. Lorsque `encoded=True`, un décodage automatique
    est appliqué et les classes sont mappées vers des noms courts
    pour améliorer la lisibilité du graphique.

    Args:
        y_true (array-like):
            Labels réels des échantillons (encodés ou non).
        y_pred_new (array-like):
            Labels prédits par le nouveau modèle.
        y_pred_ref (array-like):
            Labels prédits par le modèle de référence.
        new_name (str, default="Nouveau modèle"):
            Nom affiché pour le nouveau modèle dans la légende
            et le titre du graphique.
        ref_name (str, default="Modèle de référence"):
            Nom affiché pour le modèle de référence dans la légende
            et le titre du graphique.
        encoded (bool, default=True):
            Indique si les labels sont encodés numériquement.
            Si True, un décodage automatique est appliqué avant
            le calcul des métriques.

    Returns:
        None:
            Cette fonction ne retourne aucune valeur.
            Elle affiche directement le graphique de comparaison
            des F1-scores par catégorie.
    """


    # Décodage éventuel
    if encoded:
        y_true = decode_labels(y_true)
        y_pred_new = decode_labels(y_pred_new)
        y_pred_ref = decode_labels(y_pred_ref)

    # Mapping vers noms courts
    y_true = [CATEGORY_SHORT_NAMES[c] for c in y_true]
    y_pred_new = [CATEGORY_SHORT_NAMES[c] for c in y_pred_new]
    y_pred_ref = [CATEGORY_SHORT_NAMES[c] for c in y_pred_ref]

    # F1 globaux
    f1_weighted_new = f1_score(y_true, y_pred_new, average="weighted", zero_division=0)
    f1_weighted_ref = f1_score(y_true, y_pred_ref, average="weighted", zero_division=0)

    # F1 par classe
    f1_new = pd.DataFrame(
        classification_report(y_true, y_pred_new, output_dict=True, zero_division=0)
    ).T.iloc[:-3]["f1-score"]

    f1_ref = pd.DataFrame(
        classification_report(y_true, y_pred_ref, output_dict=True, zero_division=0)
    ).T.iloc[:-3]["f1-score"]

    df = pd.DataFrame({
        "f1_ref": f1_ref,
        "f1_new": f1_new,
    })
    df["delta_f1"] = df["f1_new"] - df["f1_ref"]

    df = df.sort_values("f1_ref")

    # Préparation du plot
    y_pos = np.arange(len(df))
    bar_height = 0.25
    plt.figure(figsize=(10, 6))

    colors = plt.get_cmap("tab20c").colors

    
    # Barres
    plt.barh(
        y_pos - bar_height / 2,
        df["f1_ref"],
        height=bar_height,
        label=ref_name,
        color=colors[2],
        alpha=0.7,
    )

    plt.barh(
        y_pos + bar_height / 2,
        df["f1_new"],
        height=bar_height,
        label=new_name,
        color=colors[1],
    )

    # Annotation du delta
    for i, (_, row) in enumerate(df.iterrows()):
        x = max(row["f1_ref"], row["f1_new"]) + 0.002
        plt.text(
            x,
            i,
            f"{row['delta_f1']:+.2f}",
            va="center",
            fontsize=9,
            # color="green" if row["delta_f1"] >= 0 else "red",
        )

    plt.yticks(y_pos, df.index)
    plt.xlabel("F1-score")
    plt.title(
        f"Comparaison des F1-scores par catégorie\n"
        f"{ref_name} → {new_name}\n"
        f"F1 pondéré : {f1_weighted_ref:.3f} → {f1_weighted_new:.3f}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()