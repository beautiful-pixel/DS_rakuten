import torch
import numpy as np
from PIL import Image
import json
from models.image.loaders import load_convnext_model, load_swin_model
from utils.calibration import calibrated_probas
from .image_transforms import build_convnext_transform, build_swin_transform


class ImageModelPipeline:
    """
    Pipeline générique pour un modèle image (Swin, ConvNeXt, ViT, etc.)
    """

    def __init__(
        self,
        model,
        transform,
        temperature: float,
        device: str = "cpu",
    ):
        self.model = model
        self.transform = transform
        self.T = temperature
        self.device = device

        self.model.eval()

    def _predict_logits(self, image_paths: list[str]) -> np.ndarray:
        logits = []

        with torch.no_grad():
            for path in image_paths:
                with Image.open(path).convert("RGB") as img:
                    x = self.transform(img).unsqueeze(0).to(self.device)
                    out = self.model(x)
                    logits.append(out.cpu().numpy())

        return np.vstack(logits)

    def predict_proba(self, image_paths: list[str]) -> np.ndarray:
        logits = self._predict_logits(image_paths)
        return calibrated_probas(logits, self.T)

class ImageFusionPipeline:
    """
    Fusion (blending) des modèles image calibrés
    """

    def __init__(self, device="cpu"):
        """
        pipelines: dict[str, ImageModelPipeline]
            ex: {"swin": swin_pipeline, "convnext": convnext_pipeline}

        weights: dict[str, float]
            ex: {"swin": 0.45, "convnext": 0.55}
        """


        # --- Paramètres fusion & calibration ---
        with open("../models/final/fusion_params.json", "r") as f:
            self.params = json.load(f)['image']

        # --- ConvNeXt ---
        convnext_model = load_convnext_model(
            checkpoint_path="../models/final/convnext/model.pth",
            device=device,
        )

        self.convnext = ImageModelPipeline(
            model=convnext_model,
            transform=build_convnext_transform(img_size=384),
            temperature=self.params['temperatures']['convnext'],
            device=device,
        )

        # --- Swin ---
        swin_model = load_swin_model(
            checkpoint_path="../models/final/swin/model.pth",
            device=device,
        )

        self.swin = ImageModelPipeline(
            model=swin_model,
            transform=build_swin_transform(img_size=224),
            temperature=self.params['temperatures']['swin'],
            device=device,
        )

    def predict_proba(self, image_paths):
        P_convnext = self.convnext.predict_proba(image_paths)
        w_convnext = self.params["weights"]["convnext"]

        P_swin = self.swin.predict_proba(image_paths)
        w_swin = self.params["weights"]["swin"]

        P = (
            w_convnext * P_convnext
            + w_swin * P_swin
        )

        return P / P.sum(axis=1, keepdims=True)
