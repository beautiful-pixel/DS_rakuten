"""
Metrics Utilities - Numpy-only metrics for model evaluation

Provides accuracy and macro F1 score calculation without heavy dependencies.
"""

import numpy as np


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Calculate accuracy from probabilities.

    Args:
        y_true: Ground truth labels (N,)
        probs: Prediction probabilities (N, num_classes)

    Returns:
        Accuracy as float
    """
    y_pred = np.argmax(probs, axis=1)
    return float(np.mean(y_pred == y_true))


def macro_f1(y_true: np.ndarray, probs: np.ndarray, num_classes: int = 27) -> float:
    """
    Calculate macro F1 score from probabilities.

    Args:
        y_true: Ground truth labels (N,)
        probs: Prediction probabilities (N, num_classes)
        num_classes: Number of classes (default 27)

    Returns:
        Macro F1 score as float
    """
    y_pred = np.argmax(probs, axis=1)

    f1_per_class = []

    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        # Precision and recall with safe division
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score with safe division
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1_per_class.append(f1)

    return float(np.mean(f1_per_class))
