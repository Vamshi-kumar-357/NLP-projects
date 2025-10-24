
# evaluation.py
"""Model evaluation helpers (accuracy, classification report, confusion matrix)."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Any
import numpy as np

def evaluate_predictions(y_true, y_pred, target_names: List[str] = None):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "y_pred": y_pred}

def evaluate_model_clf(clf, K_test, y_test, target_names: List[str] = None):
    """
    Use clf.predict on precomputed test kernel K_test (n_test x n_train or n_test x n_features depending on classifier).
    For precomputed SVM, K_test must be shape (n_test, n_train).
    """
    y_pred = clf.predict(K_test)
    return evaluate_predictions(y_test, y_pred, target_names)
