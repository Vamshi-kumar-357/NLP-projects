# visualization.py
"""Plotting utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import RESULTS_DIR

def plot_confusion_matrix(cm: np.ndarray, labels: list, figsize=(8,6), save: bool = True, fname: str = "confusion_matrix.png"):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, fname)
        plt.savefig(path, dpi=200)
        print(f"Saved confusion matrix to {path}")
    plt.show()
