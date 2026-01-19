import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from .splits import generate_splits
from .label_mapping import encode_labels


MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "../../data/raw"
IMG_DIR = MODULE_DIR / "../../data/raw/images/image_train"


def get_image_path(df: pd.DataFrame) -> pd.Series:
    """
    Construit les chemins vers les images à partir des identifiants produits.

    Les noms de fichiers suivent la convention :
    ``image_{imageid}_product_{productid}.jpg``.

    Args:
        df (pd.DataFrame): DataFrame contenant au minimum les colonnes
            ``imageid`` et ``productid``.

    Returns:
        pd.Series: Série de chemins vers les fichiers images.
    """
    file_names = (
        "image_" + df["imageid"].astype(str) +
        "_product_" + df["productid"].astype(str) + ".jpg"
    )

    return file_names.apply(lambda x: IMG_DIR / x)

def load_data(splitted: bool = False, encoded: bool = False):
    """
    Charge les données tabulaires du projet.

    Les données peuvent être retournées soit sous forme complète,
    soit découpées selon les splits train / validation / test
    définis dans le module ``splits``.

    Parameters
    ----------
    splitted : bool, optional
        Si ``True``, retourne les données découpées selon les splits
        train / validation / test. Sinon, retourne l'ensemble des données.
        Par défaut ``False``.
    encoded : bool, optional
        Si ``True``, les labels sont encodés via ``encode_labels``.
        Par défaut ``False``.

    Returns
    -------
    dict
        - Si ``splitted=False`` :
            - ``X`` : pandas.DataFrame
            - ``y`` : numpy.ndarray
        - Si ``splitted=True`` :
            - ``X_train``, ``y_train``
            - ``X_val``, ``y_val``
            - ``X_test``, ``y_test``
    """
    X = pd.read_csv(DATA_DIR / "X_train_update.csv")
    X['image_path'] = get_image_path(X)
    y = pd.read_csv(DATA_DIR / "Y_train_CVw08PX.csv")["prdtypecode"].values
    if encoded:
        y = encode_labels(y)

    if not splitted:
        return {"X": X, "y": y}

    splits = generate_splits()

    data = {
        "X_train": X.iloc[splits["train_idx"]],
        "X_val": X.iloc[splits["val_idx"]],
        "X_test": X.iloc[splits["test_idx"]],
        "y_train": y[splits["train_idx"]],
        "y_val": y[splits["val_idx"]],
        "y_test": y[splits["test_idx"]],
    }

    return data


def images_read(impath, dsize=(500, 500), grayscale=False, max_load=10**9):
    """
    Charge et redimensionne un ensemble d’images depuis leurs chemins.

    Les images sont chargées jusqu'à atteindre une limite approximative
    de mémoire afin d'éviter une surcharge RAM.

    Args:
        impath (list | pd.Series): Liste ou série de chemins vers les images.
        dsize (tuple[int, int], optional): Taille cible des images (H, W).
            Par défaut (500, 500).
        grayscale (bool, optional): Si True, charge les images en niveaux de gris.
            Sinon en RGB. Par défaut False.
        max_load (int, optional): Limite maximale approximative de mémoire
            utilisée (en unités proportionnelles aux pixels). Par défaut 1e9.

    Returns:
        np.ndarray: Tableau numpy contenant les images chargées
        de forme (N, H, W[, C]).
    """
    n = len(impath)

    total_size = (
        n * dsize[0] * dsize[1]
        if grayscale else
        n * dsize[0] * dsize[1] * 3
    )

    stop = int(n * (max_load / total_size)) if total_size > max_load else n
    images = []

    if isinstance(impath, pd.Series):
        impath = impath.reset_index(drop=True)

    for i in range(stop):
        with Image.open(impath[i]) as img:
            img = img.convert("L" if grayscale else "RGB")
            img = img.resize(dsize)
            images.append(np.array(img))

    return np.array(images)