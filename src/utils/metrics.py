import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
N_CLASSES = len(LABELS)


def compute_metrics(y_true, y_pred):
    """Primary: weighted-F1. Secondary: macro-F1, accuracy, per-class F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": f1_score(y_true, y_pred, average=None,
                                 labels=list(range(N_CLASSES)), zero_division=0).tolist(),
    }


def bootstrap_ci(y_true, y_pred, average="weighted", n_boot=1000, seed=0):
    """95% bootstrap confidence interval for F1 on the test set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average=average, zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
