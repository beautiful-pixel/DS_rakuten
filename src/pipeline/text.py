# src/pipeline/text.py

import numpy as np
import torch

import json
import joblib

from utils.calibration import calibrated_probas
from models.text.loaders import load_text_transformer
from features.text import NumericTokensTransformer, MergeTextTransformer

    
class TransformerTextPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        temperature: float,
        device: str = "cpu",
        max_length: int = 384,
        preprocess = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.T = temperature
        self.device = device
        self.max_length = max_length
        self.preprocess = preprocess

    def _predict_logits(self, texts):
        if self.preprocess:
            texts = self.preprocess.transform(texts)

        self.model.eval()
        logits_all = []

        with torch.no_grad():
            for txt in texts:
                enc = self.tokenizer(
                    txt,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**enc)
                logits_all.append(outputs.logits.cpu().numpy())

        return np.vstack(logits_all)

    def predict_proba(self, texts):
        logits = self._predict_logits(texts)
        return calibrated_probas(logits, self.T)



class TextFusionPipeline:
    def __init__(self, device="cpu"):

        # --- Paramètres fusion & calibration ---
        with open("../models/final/fusion_params.json", "r") as f:
            self.params = json.load(f)['text']

        # --- TF-IDF calibré ---
        self.tfidf = joblib.load(
            "../models/final/vectorizer_svm.joblib"
        )

        # --- CamemBERT ---
        cam_model, cam_tokenizer = load_text_transformer(
            model_dir="../models/final/camembert",
            model_name="almanach/camembert-large",
            num_labels=27,
            device=device,
        )

        self.camembert = TransformerTextPipeline(
            model=cam_model,
            tokenizer=cam_tokenizer,
            temperature=self.params['temperatures']["camembert"],
            device=device,
            preprocess=NumericTokensTransformer(strategy='light'),
        )

        # XLMR
        xlmr_model, xlmr_tokenizer = load_text_transformer(
            model_dir="../models/final/xlmr",
            model_name="xlm-roberta-base",
            num_labels=27,
            device=device,
        )

        self.xlmr = TransformerTextPipeline(
            model=xlmr_model,
            tokenizer=xlmr_tokenizer,
            temperature=self.params['temperatures']["xlmr"],
            device=device,
            preprocess=MergeTextTransformer(sep="[SEP]"),
        )


    def predict_proba(self, texts):
        P_tfidf = self.tfidf.predict_proba(texts)
        w_tfidf = self.params["weights"]["tfidf"]

        P_cam = self.camembert.predict_proba(texts)
        w_cam = self.params["weights"]["camembert"]

        P_xlmr = self.xlmr.predict_proba(texts)
        w_xlmr = self.params["weights"]["xlmr"]

        P = (
            w_tfidf * P_tfidf
            + w_cam * P_cam
            + w_xlmr * P_xlmr
        )

        return P / P.sum(axis=1, keepdims=True)