from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(**kwargs):
    default_params = dict(
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    default_params.update(kwargs)
    return TfidfVectorizer(**default_params)
