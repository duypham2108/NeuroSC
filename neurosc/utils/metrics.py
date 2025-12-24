"""
Metrics for evaluating model performance.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted',
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str, optional (default: 'weighted')
        Averaging strategy for multi-class metrics.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    
    Examples
    --------
    >>> from neurosc.utils import compute_metrics
    >>> metrics = compute_metrics(true_labels, predicted_labels)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    >>> print(f"F1 Score: {metrics['f1']:.3f}")
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
) -> str:
    """
    Generate a classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list, optional
        List of label names.
    
    Returns
    -------
    str
        Classification report as string.
    """
    from sklearn.metrics import classification_report as sk_report
    
    return sk_report(y_true, y_pred, target_names=labels, zero_division=0)

