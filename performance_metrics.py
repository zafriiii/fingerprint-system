import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1-score.
    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted binary labels (0 or 1).
    Returns:
        dict: Dictionary with accuracy, precision, recall, and f1_score.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def measure_processing_time(func, *args, **kwargs):
    """
    Measure the processing time of a function.
    Returns:
        result: The result of the function call.
        elapsed: Time in seconds.
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

def robustness_against_spoofing(y_true, y_pred, spoof_label=1, live_label=0):
    """
    Calculate robustness against spoofing as the True Negative Rate (TNR) for spoof samples.
    Args:
        y_true (array-like): True labels (0=live, 1=spoof).
        y_pred (array-like): Predicted labels (0=live, 1=spoof).
        spoof_label (int): Label representing spoof samples.
        live_label (int): Label representing live samples.
    Returns:
        float: Robustness against spoofing (TNR for spoof class).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    spoof_mask = (y_true == spoof_label)
    if np.sum(spoof_mask) == 0:
        return None  # No spoof samples
    tn = np.sum((y_pred[spoof_mask] == spoof_label))
    tnr = tn / np.sum(spoof_mask)
    return tnr

def binary_cross_entropy(y_true, y_pred):
    """
    Calculate Binary Cross Entropy loss for binary classification.
    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).
    Returns:
        float: BCE loss value.
    """
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    bce = nn.BCELoss()
    return bce(y_pred_tensor, y_true_tensor).item()

def get_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for binary classification.
    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted binary labels (0 or 1).
    Returns:
        np.ndarray: Confusion matrix [[TN, FP], [FN, TP]].
    """
    return confusion_matrix(y_true, y_pred)
