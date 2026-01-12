"""
Data Loader for Streamlit App
Loads model predictions and computes metrics
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd


def load_model_prediction(model_name: str, exports_dir: str = "artifacts/exports") -> Dict:
    """
    Load prediction results for a single model.

    Args:
        model_name: Model name (e.g., "camembert_canonical")
        exports_dir: Path to exports directory

    Returns:
        Dictionary with keys: probs, y_true, idx, metadata
    """
    model_dir = Path(exports_dir) / model_name
    npz_file = model_dir / "val.npz"
    meta_file = model_dir / "val_meta.json"

    if not npz_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Model {model_name} not found in {exports_dir}")

    # Load npz data
    data = np.load(npz_file)
    probs = data['probs']
    y_true = data['y_true']
    idx = data['idx']

    # Load metadata
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    return {
        'probs': probs,
        'y_true': y_true,
        'idx': idx,
        'metadata': metadata,
        'model_name': model_name
    }


def load_all_models(exports_dir: str = "artifacts/exports") -> Dict[str, Dict]:
    """
    Load all canonical models from exports directory.

    Args:
        exports_dir: Path to exports directory

    Returns:
        Dictionary mapping model_name to prediction data (ordered by canonical order)
    """
    exports_path = Path(exports_dir)

    if not exports_path.exists():
        raise FileNotFoundError(f"Exports directory not found: {exports_dir}")

    models = {}

    # Find all canonical models
    for model_dir in sorted(exports_path.glob("*_canonical")):
        if model_dir.is_dir():
            model_name = model_dir.name
            try:
                models[model_name] = load_model_prediction(model_name, exports_dir)
            except Exception as e:
                print(f"Warning: Failed to load {model_name}: {e}")

    # Sort models by canonical order
    sorted_model_names = sort_models_by_order(list(models.keys()))
    models = {name: models[name] for name in sorted_model_names}

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with accuracy, f1_macro, f1_weighted, precision, recall
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


def get_model_metrics(model_data: Dict) -> Dict[str, float]:
    """
    Compute metrics for a model's predictions.

    Args:
        model_data: Model prediction data (from load_model_prediction)

    Returns:
        Dictionary with metrics
    """
    probs = model_data['probs']
    y_true = model_data['y_true']
    y_pred = np.argmax(probs, axis=1)

    return compute_metrics(y_true, y_pred)


def get_all_metrics(models: Dict[str, Dict], sort_by_order: bool = True) -> pd.DataFrame:
    """
    Compute metrics for all models.

    Args:
        models: Dictionary of model predictions
        sort_by_order: If True, sort by canonical order; if False, sort by accuracy

    Returns:
        DataFrame with metrics for each model
    """
    metrics_list = []

    for model_name, model_data in models.items():
        metrics = get_model_metrics(model_data)
        metrics['model'] = model_name
        metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)

    # Reorder columns
    cols = ['model', 'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    df = df[cols]

    # Sort by canonical order or by accuracy
    if sort_by_order:
        model_order = get_model_order()

        def get_sort_key(model_name):
            name_lower = model_name.lower()
            for i, pattern in enumerate(model_order):
                if pattern in name_lower:
                    return i
            return len(model_order)

        df['_sort_key'] = df['model'].apply(get_sort_key)
        df = df.sort_values('_sort_key').drop('_sort_key', axis=1).reset_index(drop=True)
    else:
        df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)

    return df


def ensemble_predictions(models: Dict[str, Dict],
                        model_names: List[str],
                        weights: Optional[List[float]] = None,
                        method: str = 'average') -> np.ndarray:
    """
    Ensemble predictions from multiple models.

    Args:
        models: Dictionary of model predictions
        model_names: List of model names to ensemble
        weights: Optional weights for each model (must sum to 1)
        method: Ensemble method ('average', 'voting')

    Returns:
        Ensembled probability predictions
    """
    if not model_names:
        raise ValueError("At least one model must be selected")

    # Collect probabilities
    probs_list = [models[name]['probs'] for name in model_names]

    if method == 'average':
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        else:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

        # Weighted average
        ensemble_probs = np.zeros_like(probs_list[0])
        for probs, weight in zip(probs_list, weights):
            ensemble_probs += probs * weight

        return ensemble_probs

    elif method == 'voting':
        # Majority voting on predicted class
        preds = [np.argmax(probs, axis=1) for probs in probs_list]
        preds_array = np.array(preds)

        # Vote for each sample
        n_samples = preds_array.shape[1]
        n_classes = probs_list[0].shape[1]
        ensemble_probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            votes = preds_array[:, i]
            # Count votes
            unique, counts = np.unique(votes, return_counts=True)
            for cls, count in zip(unique, counts):
                ensemble_probs[i, cls] = count / len(votes)

        return ensemble_probs

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def load_canonical_classes(classes_file: str = "artifacts/canonical_classes.json") -> Dict:
    """
    Load canonical class definitions.

    Args:
        classes_file: Path to canonical classes JSON

    Returns:
        Dictionary with classes, classes_fp, num_classes
    """
    with open(classes_file, 'r') as f:
        return json.load(f)


def get_class_names(classes_file: str = "artifacts/canonical_classes.json") -> Dict[int, int]:
    """
    Get mapping from class index to class ID.

    Args:
        classes_file: Path to canonical classes JSON

    Returns:
        Dictionary mapping index (0-26) to class ID
    """
    classes_data = load_canonical_classes(classes_file)
    classes = classes_data['classes']
    return {i: cls_id for i, cls_id in enumerate(classes)}


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: If True, normalize by row (true label)

    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


def get_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 27) -> pd.DataFrame:
    """
    Compute per-class metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        DataFrame with per-class precision, recall, f1-score
    """
    precision = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)

    # Count support
    support = [(y_true == i).sum() for i in range(num_classes)]

    class_names = get_class_names()

    df = pd.DataFrame({
        'class_id': [class_names[i] for i in range(num_classes)],
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })

    return df


def get_model_order() -> List[str]:
    """
    Get canonical model ordering for consistent display.

    Returns:
        List of model name patterns in preferred order
    """
    return [
        'lenet',
        'resnet50',
        'vit',
        'swin',
        'camembert',
        'xlmr',
        'mdeberta',
        'flaubert'
    ]


def sort_models_by_order(model_names: List[str]) -> List[str]:
    """
    Sort model names according to canonical order.

    Args:
        model_names: List of model names to sort

    Returns:
        Sorted list of model names
    """
    order = get_model_order()

    def get_sort_key(name):
        # Find position in order list
        name_lower = name.lower()
        for i, pattern in enumerate(order):
            if pattern in name_lower:
                return i
        return len(order)  # Unknown models at the end

    return sorted(model_names, key=get_sort_key)


def categorize_models(model_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize models into text and image models.

    Args:
        model_names: List of model names

    Returns:
        Dictionary with 'text' and 'image' keys
    """
    text_models = []
    image_models = []

    text_keywords = ['camembert', 'flaubert', 'xlmr', 'mdeberta', 'bert']

    for name in model_names:
        if any(kw in name.lower() for kw in text_keywords):
            text_models.append(name)
        else:
            image_models.append(name)

    return {
        'text': sort_models_by_order(text_models),
        'image': sort_models_by_order(image_models)
    }
