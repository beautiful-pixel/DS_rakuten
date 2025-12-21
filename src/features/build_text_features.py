from sklearn.pipeline import Pipeline, FeatureUnion
from .text import TextCleaner, NumericTokensTransformer

def build_text_pipeline():
    """
    Construit le pipeline complet de feature engineering texte.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline texte.
    """
    text_preprocess = Pipeline(steps=[
        ('cleaner', TextCleaner),
        ('numeric_tokens', NumericTokensTransformer),
    ])

    return text_preprocess

    # return FeatureUnion([
    #     ("structured", StructuredTextFeatures()),
    #     ("cleaner", TextCleaner()),
    # ])
