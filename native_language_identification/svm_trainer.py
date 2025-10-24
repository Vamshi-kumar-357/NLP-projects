# svm_trainer.py
"""Train and save SVM models (supports precomputed kernels)."""

import os
from sklearn.svm import SVC
import joblib
from typing import Any
from config import MODEL_DIR, SVM_C, SVM_KERNEL, SVM_RANDOM_STATE

def train_svm_precomputed(K_train, y_train, C=SVM_C, kernel=SVM_KERNEL, random_state=SVM_RANDOM_STATE):
    """
    Train SVC expecting K_train as precomputed Gram matrix of shape (n_samples, n_samples).
    Note: scikit-learn expects the training Gram matrix; diagonal values help but are not strictly required.
    """
    clf = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
    clf.fit(K_train, y_train)
    return clf

def save_model(model: Any, filename: str = "svm_precomputed.joblib"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path

def load_model(path: str):
    return joblib.load(path)
