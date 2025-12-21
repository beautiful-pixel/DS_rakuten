from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TextLengthTransformer(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn calculant la longueur de champs textuels.

    La longueur peut être mesurée en nombre de caractères ou en nombre
    de mots selon le paramètre `length_unit`.
    """

    _ALLOWED_UNITS = {"char", "word"}

    def __init__(self, length_unit="char", cols=[['designation', 'description']], name_prefix=None):
        if length_unit not in self._ALLOWED_UNITS:
            raise ValueError(
                f"length_unit must be one of {self._ALLOWED_UNITS}, "
                f"got '{length_unit}'."
            )
        self.length_unit = length_unit
        self.cols = cols
        self.prefix = f"{name_prefix}_" if name_prefix else ""

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            TextLengthTransformer: Instance du transformateur.
        """
        return self

    def transform(self, X):
        """
        Calcule les longueurs des champs textuels.

        Args:
            X (pd.DataFrame): DataFrame contenant les textes.

        Returns:
            pd.DataFrame: DataFrame des longueurs calculées.
        """
        X = X.fillna("")
        if self.length_unit == 'word':
            length = {self.prefix + c : X[c].str.split().str.len() for c in self.cols}
        elif self.length_unit == 'char':
            length = {self.prefix + c : X[c].str.len() for c in self.cols}
        return pd.DataFrame(length)