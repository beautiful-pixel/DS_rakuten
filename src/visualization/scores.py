import sys
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, '../src')

from data import CATEGORY_SHORT_NAMES

def print_scores(y_true, y_pred, encoder=None, save_dir=None):
    test_score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"f1 weighted test score : {test_score:.3f}")
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
    print(f"accuracy : {round(full_report['accuracy'], 2)}")
    report = pd.DataFrame(full_report).T.iloc[:-3].sort_values('f1-score', ascending=False)

    fig, ax = plt.subplots(1,1,figsize=(10,6))
    fig.suptitle('Matrice de confusion', fontsize=16)
    #ax.set_title('normalisée par ligne (la diagonale représente le recall)')
    sns.heatmap(pd.crosstab(y_true, y_pred, normalize=0), cmap=cmap, ax=ax, vmin=0, vmax=1)
    # ax2.set_title('normalisée par colonne (la diagonale représente la precision)')
    # sns.heatmap(pd.crosstab(y_true, y_pred, normalize=1), cmap=cmap, ax=ax2, vmin=0, vmax=1)


    for ax in [ax]:
        xlabels = [tick.get_text()[:13]+'..' for tick in ax.get_xticklabels()]
        ylabels = [tick.get_text()[:13]+'..' for tick in ax.get_yticklabels()]
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.show()

    fig2, ax = plt.subplots(1,1,figsize=(9,6))
    sns.barplot(report['f1-score'], orient='h', color=c1,)
    
    for i, (idx, row) in enumerate(report.iterrows()):
        # Scores
        p = row["precision"]
        r = row["recall"]
    
        # --- Marqueurs ---
        # précision = rond
        ax.scatter(p, i, color=c2, marker="x", s=10, label="Précision" if i == 0 else "")
        # rappel = losange
        ax.scatter(r, i, color=c3, marker="D", s=10, label="Rappel" if i == 0 else "")
    
        # --- Ligne fine entre les deux points ---
        ax.plot(
            [min(p, r), max(p, r)],   # x1 → x2
            [i, i],                   # y1 → y2 (même ligne)
            color="gray",
            linestyle="--",
            linewidth=0.5
        )
    ylabels = [f"{tick.get_text()} ({f1:.2f}) " for tick, f1 in zip(ax.get_yticklabels(), report['f1-score'])]
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_ylabel("Catégorie")
    ax.set_title("F1-score par catégorie avec précision et rappel")
    ax.legend(loc="lower right")
    ax.set_xlim(0,1)
    plt.show()
    if save_dir is not None:
        fig.savefig(save_dir+'/conf_2.png', transparent=True)
        fig2.savefig(save_dir+'/f1.png', transparent=True)
    return full_report