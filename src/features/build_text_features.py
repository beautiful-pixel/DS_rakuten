from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from .text import (TextCleaner, NumericTokensTransformer, TokenFrequencyTransformer, 
                   LanguageDetector, FeatureWeighter, TextLengthTransformer)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from .text.numeric_tokens.replacer import get_all_numeric_tokens

from sklearn.linear_model import LogisticRegression

def build_preprocess_dl_pipeline():
    pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ("numeric_tokens", NumericTokensTransformer()),
    ])
    return pipeline


def build_ml_text_pipeline(max_features_title=50000, max_features_desc=50000, title_weight=1.8):
    """
    Construit le pipeline complet de feature engineering texte.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline texte.
    """

    html_tokens = ["<li>", "<br>", "<p>", "<ul>", "<strong>"]
    html_frequency_pipeline = Pipeline([
        ('freq', TokenFrequencyTransformer(html_tokens)),
        ('scaler', StandardScaler()),
    ])

    vec_params={
        "ngram_range" : (1, 2),
        "max_df" : 0.95,
    }

    title_tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features_title,**vec_params)),
        ('weight', FeatureWeighter(weight=title_weight)),
    ])

    desc_tfidf = TfidfVectorizer(
        max_features=max_features_desc,
        **vec_params
    )

    tfidf_pipeline = Pipeline([
        ("numeric_tokens", NumericTokensTransformer(merge=False)),
        ('tfidf', ColumnTransformer([
            ('title_tfidf', title_tfidf, 'designation'),
            ('desc_tfidf', desc_tfidf, 'description'),
        ], remainder='drop')
        )])

    nchars_pipeline = Pipeline([
        ("nchars", TextLengthTransformer(length_unit='char')),
        ('scaler', StandardScaler()),    
    ])

    nwords_pipeline = Pipeline([
        ("nwords", TextLengthTransformer(length_unit='word')),
        ('scaler', StandardScaler()),    
    ])


    cleaned_text_features_union = FeatureUnion([
        ("lang", LanguageDetector()),
        ("nchars", nchars_pipeline),
        ("nwords", nwords_pipeline),
        ('tfidf', tfidf_pipeline),
    ])

    cleaned_text_pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('cleaned_text_features_union', cleaned_text_features_union)
    ])

    features = FeatureUnion([
        ("cleaned_text_pipeline", cleaned_text_pipeline),
        ("html_frequency_pipeline", html_frequency_pipeline),
    ])

    return features