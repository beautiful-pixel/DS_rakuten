import numpy as np
import joblib

import numpy as np
import pandas as pd

from .image import ImageFusionPipeline
from .text import TextFusionPipeline

from data.label_mapping import CANONICAL_CLASSES, decode_labels


class FinalPipeline:
    """
    Pipeline multimodal final basé sur un meta-classifieur de stacking.
    """

    def __init__(
        self,
        image_root: str,
        text_cols=["designation", "description"],
    ):
        self.text_pipeline = TextFusionPipeline()
        self.image_pipeline = ImageFusionPipeline()
        self.meta_model = joblib.load("../models/final/meta_model.joblib")
        self.image_root = image_root
        self.text_cols = text_cols
        self.classes = CANONICAL_CLASSES

    def predict_proba(self, X):
        texts, image_paths = self._prepare_inputs(X)

        P_text = self.text_pipeline.predict_proba(texts)
        P_image = self.image_pipeline.predict_proba(image_paths)

        X_meta = np.concatenate([P_text, P_image], axis=1)
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X):
        """
        Retourne les indices canoniques (0..26)
        """
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_labels(self, X):
        """
        Retourne les product_type_code originaux
        """
        return decode_labels(self.predict(X), self.classes)

    def _prepare_inputs(self, X):
        texts = X[self.text_cols]
        file_names = (
            "image_" + X["imageid"].astype(str) +
            "_product_" + X["productid"].astype(str) + ".jpg"
        )
        image_paths = file_names.apply(
            lambda x: self.image_root + "/" + x
        ).tolist()

        return texts, image_paths
    
    def predict_with_contributions(self, X):
        """
        Retourne, pour chaque produit, la classe prédite ainsi que
        les probabilités associées au texte, à l'image et au méta-modèle.

        Output columns:
        - label_pred : label canonique prédit
        - P_final    : probabilité du méta-modèle pour la classe prédite
        - P_text     : probabilité du modèle texte pour la classe prédite
        - P_image    : probabilité du modèle image pour la classe prédite
        """

        # Préparation des entrées
        texts, image_paths = self._prepare_inputs(X)

        # Probabilités par modalité
        P_text = self.text_pipeline.predict_proba(texts)    # (n, K)
        P_image = self.image_pipeline.predict_proba(image_paths)  # (n, K)

        # Fusion via le méta-modèle
        X_meta = np.concatenate([P_text, P_image], axis=1)
        P_final = self.meta_model.predict_proba(X_meta)     # (n, K)

        # Classe prédite
        y_pred = np.argmax(P_final, axis=1)

        # Extraction des probabilités pour la classe prédite
        idx = np.arange(len(X))
        p_text_pred = P_text[idx, y_pred]
        p_image_pred = P_image[idx, y_pred]
        p_final_pred = P_final[idx, y_pred]

        # Décodage des labels
        labels_pred = decode_labels(y_pred, self.classes)

        # DataFrame de sortie
        return pd.DataFrame({
            "label_pred": labels_pred,
            "P_final": p_final_pred,
            "P_text": p_text_pred,
            "P_image": p_image_pred,
        })
