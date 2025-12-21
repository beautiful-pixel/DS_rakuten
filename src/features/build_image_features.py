from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from .image.transforms import Resizer, ImageCleaner, Flattener
from .image.color import MeanRGBTransformer, HistRGBTransformer
from .image.texture import HOGTransformer
from .image.shape import ParallelogramCounter
from .image.keypoint import CornerCounter


def build_image_pipeline():
    """
    Construit un pipeline de features visuelles combinant plusieurs familles
    de descripteurs : couleur, texture et forme.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline de feature engineering image.
    """
    preprocessing = Pipeline([
        ("resize", Resizer(dsize=(128, 128))),
        ("clean", ImageCleaner()),
    ])

    feature_union = FeatureUnion([
        ("mean_rgb", MeanRGBTransformer()),
        ("hist_rgb", HistRGBTransformer(histSize=[32])),
        ("hog", HOGTransformer()),
        ("shape", ParallelogramCounter()),
        ("corners", CornerCounter()),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("features", feature_union),
        ("scaler", StandardScaler()),
    ])

    return pipeline
