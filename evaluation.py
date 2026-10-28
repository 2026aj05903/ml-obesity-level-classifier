"""Shared helpers: column definitions and the six evaluation metrics.

Used by both train_models.py (to score the held-out split) and app.py
(to score whatever test CSV is uploaded), so the numbers always agree.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "NObeyesdad"

# Height and Weight are deliberately NOT used: the label is derived from BMI
# (weight / height^2), so including them would let any model simply recompute it.
NUMERIC_FEATURES = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_FEATURES = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 14 features

# Lightest to heaviest, used to order confusion-matrix rows/columns.
LEVEL_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def macro_ovr_auc(y_true, proba, classes):
    """Macro-averaged one-vs-rest AUC.

    Each class is scored against all the others and the scores are averaged.
    A class that is absent from y_true (or is the only class present) has no
    defined AUC, so it is skipped - this keeps the app working on small uploads.
    """
    y_true = np.asarray(y_true)
    scores = []
    for column, label in enumerate(classes):
        is_label = (y_true == label).astype(int)
        if is_label.sum() == 0 or is_label.sum() == len(is_label):
            continue
        scores.append(roc_auc_score(is_label, proba[:, column]))
    return float(np.mean(scores)) if scores else float("nan")


def compute_all_metrics(model, features, y_true):
    """Return the six metrics for a fitted pipeline on (features, y_true).

    Precision, recall and F1 are macro-averaged (every class counts equally),
    which suits this nearly balanced 7-class problem.
    """
    y_pred = model.predict(features)
    proba = model.predict_proba(features)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": macro_ovr_auc(y_true, proba, list(model.classes_)),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def ordered_confusion_matrix(y_true, y_pred):
    """Confusion matrix with rows/columns in LEVEL_ORDER (only levels present)."""
    present = set(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    labels = [level for level in LEVEL_ORDER if level in present]
    return confusion_matrix(y_true, y_pred, labels=labels), labels
