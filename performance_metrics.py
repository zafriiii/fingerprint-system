import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

def measure_processing_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

def robustness_against_spoofing(y_true, y_pred, spoof_label=1, live_label=0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    spoof_mask = y_true == spoof_label
    if np.sum(spoof_mask) == 0:
        return None
    tn = np.sum((y_pred[spoof_mask] == spoof_label))
    tnr = tn / np.sum(spoof_mask)
    return tnr

def binary_cross_entropy(y_true, y_pred):
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
    bce = nn.BCELoss()
    return bce(y_pred_tensor, y_true_tensor).item()

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
