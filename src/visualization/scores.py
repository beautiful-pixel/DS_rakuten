import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data import CATEGORY_SHORT_NAMES



def plot_f1_scores(y_true, y_pred, encoder=None, save_dir=None, k_worst=None):
    # voir si faire des comparaison entre score de différents modèle peut être bien
    test_score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"f1 weighted score : {test_score:.3f}")
    if encoder is not None:
        y_pred = encoder.inverse_transform(y_pred)
        y_true = encoder.inverse_transform(y_true)

    y_true = [CATEGORY_SHORT_NAMES[code] for code in y_true]
    y_pred = [CATEGORY_SHORT_NAMES[code] for code in y_pred]

    cmap = plt.cm.Blues
    c1 = cmap(0.25)
    c2 = plt.cm.seismic(0.05)
    c3 = plt.cm.tab20b(13)
    
    full_report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)
    report = pd.DataFrame(full_report).T.iloc[:-3].sort_values('f1-score', ascending=False)


    fig2, ax = plt.subplots(1,1,figsize=(8,6))
    sns.barplot(report.iloc[-k_worst:]['f1-score'], orient='h', color=c1,)
    
    for i, (_, row) in enumerate(report.iloc[-k_worst:].iterrows()):
        p = row["precision"]
        r = row["recall"]
        ax.scatter(p, i, color=c2, marker="x", s=10, label="Précision" if i == 0 else "")
        ax.scatter(r, i, color=c3, marker="D", s=10, label="Rappel" if i == 0 else "")
        ax.plot(
            [min(p, r), max(p, r)],
            [i, i],
            color="gray",
            linestyle="--",
            linewidth=0.5
        )
    ax.set_xlabel("Score")
    ax.set_ylabel("Catégorie")
    ax.set_title("F1-score par catégorie avec précision et rappel")
    ax.legend(loc="lower right")
    ax.set_xlim(0,1)
    plt.show()
    return full_report