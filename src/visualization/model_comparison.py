import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from data import CATEGORY_SHORT_NAMES


def plot_f1_comparison_with_delta(
    y_true,
    y_pred_new,
    y_pred_ref,
    new_name="Nouveau modèle",
    ref_name="Modèle de référence",
    encoder=None,
):
    """
    Compare deux modèles de classification par catégorie à l’aide des F1-scores.

    Le graphique affiche, pour chaque classe :
    - le F1-score du modèle de référence ;
    - le F1-score du nouveau modèle ;
    - le delta de F1-score (nouveau - référence), annoté sur le graphique.

    Les catégories sont triées par gain de F1 décroissant afin de mettre
    en évidence les classes les plus impactées par le nouveau modèle.
    """

    # Décodage éventuel
    if encoder is not None:
        y_true = encoder.inverse_transform(y_true)
        y_pred_new = encoder.inverse_transform(y_pred_new)
        y_pred_ref = encoder.inverse_transform(y_pred_ref)

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
