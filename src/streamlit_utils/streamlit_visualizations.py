"""
Visualization functions for Streamlit App
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List


def plot_metrics_comparison(metrics_df: pd.DataFrame, metric: str = 'accuracy') -> go.Figure:
    """
    Plot bar chart comparing models on a specific metric.

    Args:
        metrics_df: DataFrame with model metrics
        metric: Metric to plot (accuracy, f1_macro, etc.)

    Returns:
        Plotly figure
    """
    fig = px.bar(
        metrics_df,
        x='model',
        y=metric,
        title=f'Model Comparison - {metric.replace("_", " ").title()}',
        labels={'model': 'Model', metric: metric.replace("_", " ").title()},
        text=metric
    )

    # Format text to 4 decimal places
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')

    # Rotate x-axis labels
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )

    return fig


def plot_all_metrics_heatmap(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Plot heatmap of all metrics for all models.

    Args:
        metrics_df: DataFrame with model metrics

    Returns:
        Plotly figure
    """
    # Select numeric columns only
    metric_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    data = metrics_df[metric_cols].values

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[col.replace('_', ' ').title() for col in metric_cols],
        y=metrics_df['model'].values,
        colorscale='RdYlGn',
        text=np.round(data, 4),
        texttemplate='%{text:.4f}',
        textfont={"size": 10},
        colorbar=dict(title="Score")
    ))

    fig.update_layout(
        title='Model Performance Heatmap',
        xaxis_title='Metrics',
        yaxis_title='Models',
        height=600
    )

    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool = False) -> go.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: If True, show normalized values

    Returns:
        Plotly figure
    """
    if normalize:
        cm_display = np.round(cm, 3)
        colorscale = 'Blues'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm.astype(int)
        colorscale = 'Blues'
        title = 'Confusion Matrix'

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        colorscale=colorscale,
        text=cm_display,
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Count" if not normalize else "Proportion")
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        height=700,
        width=700
    )

    return fig


def plot_per_class_metrics(per_class_df: pd.DataFrame, top_n: int = None) -> go.Figure:
    """
    Plot per-class metrics (precision, recall, f1-score).

    Args:
        per_class_df: DataFrame with per-class metrics
        top_n: Show only top N classes by support

    Returns:
        Plotly figure
    """
    df = per_class_df.copy()

    if top_n:
        df = df.nlargest(top_n, 'support')

    df['class_id'] = df['class_id'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['class_id'],
        y=df['precision'],
        name='Precision',
        marker_color='lightsalmon'
    ))

    fig.add_trace(go.Bar(
        x=df['class_id'],
        y=df['recall'],
        name='Recall',
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        x=df['class_id'],
        y=df['f1_score'],
        name='F1-Score',
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title=f'Per-Class Metrics{"" if not top_n else f" (Top {top_n} by Support)"}',
        xaxis_title='Class ID',
        yaxis_title='Score',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def plot_model_agreement(models: Dict[str, Dict], model_names: List[str]) -> go.Figure:
    """
    Plot agreement matrix between models.

    Args:
        models: Dictionary of model predictions
        model_names: List of model names to compare

    Returns:
        Plotly figure
    """
    n_models = len(model_names)
    agreement_matrix = np.zeros((n_models, n_models))

    for i, model_i in enumerate(model_names):
        pred_i = np.argmax(models[model_i]['probs'], axis=1)
        for j, model_j in enumerate(model_names):
            pred_j = np.argmax(models[model_j]['probs'], axis=1)
            agreement = (pred_i == pred_j).mean()
            agreement_matrix[i, j] = agreement

    fig = go.Figure(data=go.Heatmap(
        z=agreement_matrix,
        x=model_names,
        y=model_names,
        colorscale='RdYlGn',
        text=np.round(agreement_matrix, 3),
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        colorbar=dict(title="Agreement")
    ))

    fig.update_layout(
        title='Model Agreement Matrix',
        xaxis_title='Model',
        yaxis_title='Model',
        height=600,
        xaxis_tickangle=-45
    )

    return fig


def plot_prediction_confidence(probs: np.ndarray, y_true: np.ndarray, sample_size: int = 1000) -> go.Figure:
    """
    Plot distribution of prediction confidence for correct vs incorrect predictions.

    Args:
        probs: Prediction probabilities (N, 27)
        y_true: True labels
        sample_size: Number of samples to plot

    Returns:
        Plotly figure
    """
    y_pred = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    correct = y_pred == y_true
    correct_conf = max_probs[correct]
    incorrect_conf = max_probs[~correct]

    # Sample if too large
    if len(correct_conf) > sample_size:
        correct_conf = np.random.choice(correct_conf, sample_size, replace=False)
    if len(incorrect_conf) > sample_size:
        incorrect_conf = np.random.choice(incorrect_conf, sample_size, replace=False)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=correct_conf,
        name='Correct Predictions',
        opacity=0.7,
        marker_color='green',
        nbinsx=50
    ))

    fig.add_trace(go.Histogram(
        x=incorrect_conf,
        name='Incorrect Predictions',
        opacity=0.7,
        marker_color='red',
        nbinsx=50
    ))

    fig.update_layout(
        title='Prediction Confidence Distribution',
        xaxis_title='Confidence (Max Probability)',
        yaxis_title='Count',
        barmode='overlay',
        height=400
    )

    return fig


def plot_ensemble_improvement(baseline_metrics: Dict[str, float],
                              ensemble_metrics: Dict[str, float]) -> go.Figure:
    """
    Plot comparison between baseline and ensemble metrics.

    Args:
        baseline_metrics: Metrics of best single model
        ensemble_metrics: Metrics of ensemble model

    Returns:
        Plotly figure
    """
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

    baseline_values = [baseline_metrics[m] for m in metrics]
    ensemble_values = [ensemble_metrics[m] for m in metrics]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics,
        y=baseline_values,
        name='Best Single Model',
        marker_color='lightblue',
        text=[f'{v:.4f}' for v in baseline_values],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=ensemble_values,
        name='Ensemble',
        marker_color='lightgreen',
        text=[f'{v:.4f}' for v in ensemble_values],
        textposition='auto'
    ))

    fig.update_layout(
        title='Ensemble vs Best Single Model',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def plot_sample_predictions(models: Dict[str, Dict],
                           model_names: List[str],
                           sample_idx: int,
                           class_names: Dict[int, int]) -> go.Figure:
    """
    Plot prediction probabilities for a single sample across models.

    Args:
        models: Dictionary of model predictions
        model_names: List of model names
        sample_idx: Index of sample to plot
        class_names: Mapping from class index to class ID

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for model_name in model_names:
        probs = models[model_name]['probs'][sample_idx]
        class_ids = [class_names[i] for i in range(len(probs))]

        fig.add_trace(go.Bar(
            x=class_ids,
            y=probs,
            name=model_name,
            opacity=0.7
        ))

    y_true_idx = models[model_names[0]]['y_true'][sample_idx]
    y_true_class = class_names[y_true_idx]

    fig.update_layout(
        title=f'Sample #{sample_idx} - True Class: {y_true_class}',
        xaxis_title='Class ID',
        yaxis_title='Probability',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )

    return fig
