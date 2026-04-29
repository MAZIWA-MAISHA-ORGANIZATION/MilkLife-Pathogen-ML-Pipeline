"""Post-training evaluation suite for the MilkLife pathogen classifier.

Outputs production-grade metrics — Confusion Matrix, per-class F1-Score,
and macro-averaged ROC-AUC — to validate model quality before edge export.
"""

import os
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.config import DATA, PATHS
from src.data_loader import generate_dataset


def evaluate(
    model: Optional[keras.Model] = None,
    model_path: Optional[str] = None,
) -> dict:
    """Runs comprehensive evaluation on the held-out test set.

    Args:
        model: A trained Keras model. If None, loads from model_path.
        model_path: Path to a saved .keras model.

    Returns:
        Dict containing confusion_matrix, f1_scores, and roc_auc.
    """
    if model is None:
        model_path = model_path or PATHS.best_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run train.py first."
            )
        model = keras.models.load_model(model_path)

    _, X_test, _, y_test = generate_dataset()

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n── Confusion Matrix ──")
    _print_confusion_matrix(cm, list(DATA.class_names))

    # Classification Report (includes per-class F1)
    report = classification_report(
        y_test, y_pred, target_names=list(DATA.class_names), digits=4
    )
    print("\n── Classification Report ──")
    print(report)

    # Macro F1
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"  Macro F1-Score    : {macro_f1:.4f}")
    print(f"  Weighted F1-Score : {weighted_f1:.4f}")

    # ROC-AUC (one-vs-rest, macro)
    roc_auc = roc_auc_score(
        y_test, y_probs, multi_class="ovr", average="macro"
    )
    print(f"  Macro ROC-AUC     : {roc_auc:.4f}\n")

    return {
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
    }


def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Pretty-prints a confusion matrix to the terminal."""
    max_label = max(len(l) for l in labels)
    header = " " * (max_label + 2) + "  ".join(
        f"{l:>{max_label}}" for l in labels
    )
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>{max_label}}" for val in row)
        print(f"{labels[i]:>{max_label}}  {row_str}")


if __name__ == "__main__":
    evaluate()
