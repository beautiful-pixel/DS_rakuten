import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path

MODULE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MODULE_DIR, "../data/raw/")
IMG_DIR = os.path.join(MODULE_DIR, "../data/raw/images/image_train/")

# Constantes pour les splits unifiés du projet
SEED = 42
TEST_SIZE = 0.15  # 15% pour test
VAL_SIZE = 0.15   # 15% du reste pour validation
SPLITS_DIR = Path(MODULE_DIR) / ".." / "data" / "splits"

def split_data():
    """
    retourne les données splitées entre le jeu de test utilisé pour ce projet
    et le jeu d'entraintement / validation
    """
    X = pd.read_csv(DATA_DIR+'X_train_update.csv', index_col=0)
    y = pd.read_csv(DATA_DIR+'Y_train_CVw08PX.csv', index_col=0)['prdtypecode']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_path(df):
    """
    retourne les chemins des images depuis ce module
    
    :param df: dataframe comprenant productid et imageid
    """
    file_names = (
        "image_" + df['imageid'].astype('str') + "_product_" +
        df['productid'].astype('str') + ".jpg"
    )
    return IMG_DIR + file_names


def split_path():
    """
    retourne les chemins des images splités entre le jeu de test utilisé pour ce projet
    et le jeu d'entraintement / validation
    """
    X_train, X_test, y_train, y_test = split_data()
    X_train, X_test = get_path(X_train), get_path(X_test)
    return X_train, X_test, y_train, y_test

def split_txt():
    """
    retourne les données textuelles splitées entre le jeu de test utilisé pour ce projet
    et le jeu d'entraintement / validation
    """
    X_train, X_test, y_train, y_test = split_data()
    X_train = X_train[['designation', 'description']]
    X_test = X_test[['designation', 'description']]
    return X_train, X_test, y_train, y_test

def get_mask(tags_path, columns):
    """
    les tags doivent être du genre True pour les entrées à supprimer
    retourne True pour celles qui ont False sur toute les lignes
    
    :param tags_path: chemin vers le fichier contenant les tags
    :param columns: columns à prendre en compte
    """
    tags = pd.read_csv(tags_path)[columns]
    mask = (~tags).all(axis=1)
    return mask

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


# max_load 
# le nombre d'images retourné se fera en fonction de ce paramètre

def images_read(impath, dsize=(500,500), grayscale=False, max_load=10**9):
    """
    Docstring for images_read
    :param impath: chemin des images
    :param dsize: dimension des images souhaitée
    :param grayscale: pour charger l'image en niveau de gris
    :param max_load: représente le nombre d'octet maximum chargé en mémoire 10**9 => 1 Go
    """
    n = len(impath)
    # taille maximal en octet
    total_size = n*dsize[0]*dsize[1] if grayscale else n*dsize[0]*dsize[1]*3
    stop = int(n * (max_load/total_size)) if total_size > max_load else n
    images = []
    # si c'est une Series il faut reindexé par 0, ... ,n pour pouvoir utiliser les indices de la même manière qu'une liste
    if type(impath) == pd.Series:
        impath = impath.reset_index(drop=True)

    for i in range(stop):
        with Image.open(impath[i]) as img:
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            img = img.resize(dsize)
            images.append(np.array(img))
    return np.array(images)


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


def load_unified_splits():
    """
    Charge les indices de split unifiés du projet.

    Returns:
        dict: Contient train_idx, val_idx, test_idx
    """
    if not SPLITS_DIR.exists():
        raise FileNotFoundError(
            f"Le répertoire {SPLITS_DIR} n'existe pas. "
            f"Veuillez d'abord générer les splits avec generate_splits(save=True)"
        )

    splits = {
        "train_idx": np.loadtxt(SPLITS_DIR / "train_idx.txt", dtype=int),
        "val_idx": np.loadtxt(SPLITS_DIR / "val_idx.txt", dtype=int),
        "test_idx": np.loadtxt(SPLITS_DIR / "test_idx.txt", dtype=int)
    }

    return splits


def get_split_data_unified():
    """
    Retourne les données splitées selon les indices unifiés du projet.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Charger les données complètes
    X = pd.read_csv(DATA_DIR + 'X_train_update.csv', index_col=0)
    y = pd.read_csv(DATA_DIR + 'Y_train_CVw08PX.csv', index_col=0)['prdtypecode']

    # Charger les indices unifiés
    splits = load_unified_splits()

    # Appliquer les splits
    X_train = X.iloc[splits["train_idx"]].reset_index(drop=True)
    X_val = X.iloc[splits["val_idx"]].reset_index(drop=True)
    X_test = X.iloc[splits["test_idx"]].reset_index(drop=True)

    y_train = y.iloc[splits["train_idx"]].reset_index(drop=True)
    y_val = y.iloc[splits["val_idx"]].reset_index(drop=True)
    y_test = y.iloc[splits["test_idx"]].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test