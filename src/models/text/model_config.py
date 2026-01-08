from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

MODEL_CONFIGS = {
    "lr": LogisticRegression(C=2.0, max_iter=2000),
    "svm": LinearSVC(C=1.0, class_weight="balanced"),
    "nb": ComplementNB(alpha=0.5),
}
