TFIDF_CONFIGS = {
    "tfidf_unigram": dict(
        ngram_range=(1,1),
    ),
    "tfidf_uni_bigram": dict(
        ngram_range=(1,2),
    ),
    "tfidf_char_3_5": dict(
        analyzer="char",
        ngram_range=(3,5),
        min_df=5
    ),
}
