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


def generate_splits():
    """
    Génère des splits train / validation / test fixes et stratifiés.

    Les proportions et la graine aléatoire sont volontairement figées
    afin de garantir des splits strictement identiques pour l'ensemble
    des modèles, modalités et expériences du projet.

    Returns:
        dict: Dictionnaire contenant :
            - train_idx (np.ndarray)
            - val_idx (np.ndarray)
            - test_idx (np.ndarray)
    """
    y = pd.read_csv(DATA_DIR / "Y_train_CVw08PX.csv")["prdtypecode"].to_numpy()
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
    return splits