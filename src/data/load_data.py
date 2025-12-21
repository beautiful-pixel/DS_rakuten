import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

MODULE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MODULE_DIR, "../../data/raw/")
IMG_DIR = os.path.join(MODULE_DIR, "../../data/images/image_train/")

def split_data():
    """
    retourne les données splitées entre le jeu de test utilisé pour ce projet
    et le jeu d'entraintement / validation
    """
    X = pd.read_csv(DATA_DIR+'X_train_update.csv')
    y = pd.read_csv(DATA_DIR+'Y_train_CVw08PX.csv')['prdtypecode']
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