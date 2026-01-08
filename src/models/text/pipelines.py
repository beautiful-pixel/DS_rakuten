from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.features.text.vectorizers import build_tfidf

def build_text_pipeline(tfidf_params, model, calibrate=True):
    clf = (
        CalibratedClassifierCV(model, cv=2)
        if calibrate
        else model
    )

    return Pipeline([
        ("tfidf", build_tfidf(**tfidf_params)),
        ("clf", clf),
    ])
