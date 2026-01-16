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