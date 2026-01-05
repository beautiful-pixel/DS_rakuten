from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from .replacer import replace_numeric_expressions, get_numeric_tokens


COL_TOKENS = ["[TITRE]", "[DESC]"]

class NumericTokensTransformer(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn remplaçant les expressions numériques
    dans le texte par des tokens sémantiques.

    Ce transformateur applique une tokenisation numérique sur les champs
    textuels (titre et description) en remplaçant les valeurs quantitatives
    (dimensions, poids, dates, quantités, etc.) par des tokens discrets.
    """

    def __init__(
            self, designation_col='designation', description_col='description',
            merge=True, output_col='text'):
        """
        Initialise le transformateur de tokenisation numérique.

        Args:
            designation_col (str, optional): Nom de la colonne contenant
                le titre du produit. Par défaut 'designation'.
            description_col (str, optional): Nom de la colonne contenant
                la description du produit. Par défaut 'description'.
        """
        self.designation_col = designation_col
        self.description_col = description_col
        self.merge = merge
        # output_col utile seulement si merge = True
        self.output_col = output_col

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            NumericTokensTransformer: Instance du transformateur.
        """
        return self

    def transform(self, X):
        """
        Applique la tokenisation numérique aux champs textuels.

        Les champs titre et description sont concaténés en ajoutant
        des tokens de séparation afin de préserver la structure du texte.

        Args:
            X (pd.DataFrame): DataFrame contenant les colonnes textuelles.

        Returns:
            pd.DataFrame: DataFrame contenant le texte transformé avec tokens numériques.
        """
        title = X["designation"].fillna("").apply(replace_numeric_expressions)
        desc = X["description"].fillna("").apply(replace_numeric_expressions)
        data = {}
        if self.merge:
            data[self.output_col] = COL_TOKENS[0] + " " + title + " " + COL_TOKENS[1] + " " + desc
        else:
            data["designation"] = title
            data["description"] = desc
        return pd.DataFrame(data)
    
    def get_tokens(self):
        """
        Retourne la liste des tokens numériques utilisés par le transformateur.

        Returns:
            list[str]: Liste des tokens numériques possibles.
        """
        return COL_TOKENS + get_numeric_tokens()