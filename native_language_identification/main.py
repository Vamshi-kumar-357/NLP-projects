# main.py
"""
Main orchestration script: loads data, preprocesses, computes kernels, trains SVM, evaluates, and plots.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from data_loader import load_data
from preprocessing import preprocess_data_frames, get_ngrams
from kernel_utils import computeKernelMatrix_presence, computeKernelMatrix_presence_test
from config import MODEL_DIR, RESULTS_DIR

def main(train_path=None, test_path=None):
    # ------------------ Load and Preprocess Data ------------------
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)

    print("Preprocessing data...")
    train_df, test_df = preprocess_data_frames(train_df, test_df)

    # Extract texts and labels
    X_train_texts = train_df["text"]
    X_test_texts = test_df["text"]
    Y_train_raw = train_df["native_language"]
    Y_test_raw = test_df["native_language"]

    print(f"Loaded {X_train_texts[0]} training examples and {len(X_test_texts)} test examples.")

    # ------------------ Encode Labels ------------------
    le = LabelEncoder()


    Y_train = le.fit_transform(Y_train_raw)
    Y_test = le.transform(Y_test_raw)
    classes = list(le.classes_)
    print("Detected classes:", classes)

    # ------------------ Generate n-grams ------------------
    print("Generating 1-gram tokens for kernel computation...")
    V_train = X_train_texts
    V_test = X_test_texts
    for j in range(len(X_train_texts)):
        V_train[j] = get_ngrams(X_train_texts[j],1)
        # break

    print(f"Generated {V_train[0]} 1-grams for training data.")

    for j in range(len(X_test_texts)):
        V_test[j] = get_ngrams(X_test_texts[j],1)

    # ------------------ Compute Presence Kernel ------------------
    print("Computing training kernel matrix...")
    K_train = computeKernelMatrix_presence(1, 1, V_train)
    print("Computing test kernel matrix...")
    K_test = computeKernelMatrix_presence_test(1, 1, V_train, V_test)
    print("Kernel computation done.")

    # ------------------ Train SVM ------------------
    print("Training SVM...")
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, Y_train)

    # ------------------ Predict & Evaluate ------------------
    y_pred = clf.predict(K_test)
    acc = accuracy_score(Y_test, y_pred)
    print(f"The accuracy for presence kernel with 1-gram is {acc}")
    print("--" * 50)

    # ------------------ Confusion Matrix ------------------
    cm = confusion_matrix(Y_test, y_pred)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="YlGnBu")
    plt.rcParams.update({'font.size': 25})
    plt.rc('font', size=35)
    ax.set_xlabel('Predicted Labels', fontsize=35)
    ax.set_ylabel('True Labels', fontsize=35)
    ax.set_title('Confusion Matrix', fontsize=35)
    ax.xaxis.set_ticklabels(classes, rotation=90, fontsize=35)
    ax.yaxis.set_ticklabels(classes, rotation=0, fontsize=35)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_presence_1gram.png"))
    print("Confusion matrix saved.")

    # ------------------ Save Model ------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    joblib.dump(clf, os.path.join(MODEL_DIR, "svm_presence_1gram.joblib"))
    print("Model saved.")

    print("Done!")


if __name__ == "__main__":
    main()
