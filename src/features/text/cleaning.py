from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

import sys
sys.path.insert(0, '../..')
from rakuten_text import clean_text


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Transformateur scikit-learn appliquant un nettoyage de base du texte.

    Ce transformateur applique une série d'opérations de normalisation
    (encodage, HTML, Unicode) sur les colonnes textuelles spécifiées.
    """

    def __init__(self, cols=['designation', 'description'], name_prefix=None):
        """
        Initialise le transformateur de nettoyage du texte.

        Args:
            cols (list[str], optional): Liste des colonnes textuelles à nettoyer.
                Par défaut ['designation', 'description'].
            name_prefix (str | None, optional): Préfixe ajouté aux noms
                des colonnes générées. Par défaut None.
        """
        self.cols = cols
        self.name_prefix = name_prefix
        self.prefix = f"{name_prefix}_" if name_prefix else ""

    def fit(self, X, y=None):
        """
        Ajuste le transformateur (aucun apprentissage nécessaire).

        Args:
            X (pd.DataFrame): Données d'entrée.
            y (Any, optional): Variable cible ignorée.

        Returns:
            TextCleaner: Instance du transformateur.
        """
        return self

    def transform(self, X):
        """
        Applique le nettoyage du texte aux colonnes spécifiées.

        Args:
            X (pd.DataFrame): DataFrame contenant les colonnes textuelles.

        Returns:
            pd.DataFrame: DataFrame contenant les textes nettoyés.
        """
        params = {
            'fix_encoding':True, 'unescape_html':True,
            'normalize_unicode':True, 'remove_html_tags':True
            }
        cleaned_text = {
            self.prefix + c : X[c].fillna("").apply(lambda x : clean_text(x, **params))
            for c in self.cols
            }
        return pd.DataFrame(cleaned_text)


