import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "../../data/raw"
SPLITS_DIR = MODULE_DIR / "../../data/splits"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def generate_splits(save=False):
    """
    Génère des splits train / validation / test fixes et stratifiés.

    Les proportions et la graine aléatoire sont volontairement figées
    afin de garantir des splits strictement identiques pour l'ensemble
    des modèles, modalités et expériences du projet.

    Args:
        save (bool): Si True, sauvegarde les indices dans data/splits/.

    Returns:
        dict: Dictionnaire contenant :
            - train_idx
            - val_idx
            - test_idx
    """
    y = pd.read_csv(DATA_DIR+'Y_train_CVw08PX.csv')['prdtypecode'].to_numpy()
    indices = np.arange(len(y))
    full_train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    train_idx, val_idx = train_test_split(
        full_train_idx, test_size=VAL_SIZE, random_state=SEED, stratify=y[full_train_idx]
    )
    splits = {
        "train_idx" : train_idx,
        "val_idx" : val_idx,
        "test_idx" : test_idx,
    }
    if save:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for name, idx in splits.items():
            np.savetxt(SPLITS_DIR / f"{name}.txt", idx, fmt="%d")
    return splits


def load_splits():
    """
    Charge les indices train / validation / test préalablement générés.

    Les splits sont supposés avoir été générés une seule fois via
    `generate_splits(save=True)` et sont ensuite réutilisés tels quels
    pour l'ensemble du projet afin de garantir la reproductibilité
    et la comparabilité des modèles.

    Returns:
        dict: Dictionnaire contenant :
            - train_idx (np.ndarray)
            - val_idx (np.ndarray)
            - test_idx (np.ndarray)

    Raises:
        FileNotFoundError: Si un ou plusieurs fichiers de split sont absents.
    """
    split_files = {
        "train_idx": SPLITS_DIR / "train_idx.txt",
        "val_idx": SPLITS_DIR / "val_idx.txt",
        "test_idx": SPLITS_DIR / "test_idx.txt",
    }

    splits = {}
    for name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Split file '{path}' not found. "
                "Run generate_splits(save=True) once to create it."
            )
        splits[name] = np.loadtxt(path, dtype=int)

    return splits