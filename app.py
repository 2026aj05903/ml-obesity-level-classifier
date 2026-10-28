"""Streamlit front-end: evaluate the five saved obesity-level classifiers on an uploaded test CSV.

The models are loaded from model/*.joblib and are NEVER retrained here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from evaluation import (
    FEATURE_COLUMNS,
    LEVEL_ORDER,
    METRIC_NAMES,
    TARGET_COLUMN,
    compute_all_metrics,
    ordered_confusion_matrix,
)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_CSV = APP_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Obesity Level Classifier", page_icon="⚖️", layout="wide")


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


def read_summary():
    path = MODEL_DIR / "model_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def draw_confusion_matrix(y_true, y_pred, as_percent):
    matrix, labels = ordered_confusion_matrix(y_true, y_pred)
    shown = matrix.astype(float)
    if as_percent:
        shown = shown / np.maximum(shown.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(shown, cmap="YlGnBu")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("Actual level")
    threshold = shown.max() / 2 if shown.max() > 0 else 0
    for row in range(len(labels)):
        for col in range(len(labels)):
            text = f"{shown[row, col]:.0f}%" if as_percent else f"{int(matrix[row, col])}"
            ax.text(col, row, text, ha="center", va="center",
                    color="white" if shown[row, col] > threshold else "black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ header
st.title("⚖️ Obesity Level Classifier")
st.caption(
    "Five classifiers trained on the UCI *Estimation of Obesity Levels* dataset "
    "(eating habits and physical condition → 7 obesity levels). "
    "Upload a test CSV to score them. Nothing is retrained in this app."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded_file = st.file_uploader(
        f"Upload a CSV with the 14 feature columns and the '{TARGET_COLUMN}' column", type=["csv"]
    )
    if SAMPLE_CSV.exists():
        st.download_button(
            "Download the sample test_data.csv",
            data=SAMPLE_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    st.header("2 · Model")
    chosen_model = st.selectbox("Choose a model", list(MODEL_FILES))
    with st.expander("Required columns"):
        st.write(", ".join(FEATURE_COLUMNS + [TARGET_COLUMN]))
        st.caption("Height and Weight are not needed: they were excluded because the label is derived from BMI.")

# ------------------------------------------------------------------ nothing uploaded yet
if uploaded_file is None:
    st.info("Upload a test CSV in the sidebar to evaluate the models. "
            "You can start with the sample file from the sidebar.")
    summary = read_summary()
    if summary:
        st.subheader("Reference results on the held-out split")
        st.caption(f"Computed during training on {summary['test_rows']} test rows.")
        reference = pd.DataFrame(
            {name: info["test_metrics"] for name, info in summary["models"].items()}
        ).T[METRIC_NAMES]
        st.dataframe(reference.round(4))
    st.stop()

# ------------------------------------------------------------------ read + validate the upload
try:
    uploaded = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read that file as a CSV: {error}")
    st.stop()

missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in uploaded.columns]
if missing:
    st.error("The CSV is missing required column(s): " + ", ".join(missing))
    st.stop()

needed = uploaded[FEATURE_COLUMNS + [TARGET_COLUMN]]
rows_before = len(needed)
needed = needed.dropna().reset_index(drop=True)
if len(needed) < rows_before:
    st.warning(f"Dropped {rows_before - len(needed)} row(s) with missing values.")

unknown_levels = sorted(set(needed[TARGET_COLUMN]) - set(LEVEL_ORDER))
if unknown_levels:
    st.error(f"'{TARGET_COLUMN}' contains unexpected value(s): {unknown_levels}. "
             f"Allowed: {LEVEL_ORDER}")
    st.stop()
if needed.empty:
    st.error("No usable rows left after removing missing values.")
    st.stop()

features, actual = needed[FEATURE_COLUMNS], needed[TARGET_COLUMN]
st.success(f"Loaded {len(needed)} rows · {len(FEATURE_COLUMNS)} features · "
           f"{actual.nunique()} obesity levels present.")

# ------------------------------------------------------------------ score every saved model (no training)
all_scores = {}
for model_name in MODEL_FILES:
    try:
        all_scores[model_name] = compute_all_metrics(load_model(model_name), features, actual)
    except Exception as error:
        st.error(f"{model_name} could not score this file: {error}")
        st.stop()

model = load_model(chosen_model)
predicted = model.predict(features)

# ------------------------------------------------------------------ results
st.header(f"Results: {chosen_model}")
tab_metrics, tab_matrix, tab_report, tab_compare, tab_rows = st.tabs(
    ["Metrics", "Confusion matrix", "Classification report", "All models", "Predictions"]
)

with tab_metrics:
    scores = all_scores[chosen_model]
    for start in (0, 3):
        for column, metric_name in zip(st.columns(3), METRIC_NAMES[start:start + 3]):
            column.metric(metric_name, f"{scores[metric_name]:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged over the 7 levels; "
               "AUC is the macro-averaged one-vs-rest AUC.")

with tab_matrix:
    as_percent = st.checkbox("Show as % of each actual level", value=False)
    figure = draw_confusion_matrix(actual, predicted, as_percent)
    st.pyplot(figure)
    plt.close(figure)

with tab_report:
    report = classification_report(actual, predicted, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

with tab_compare:
    comparison = pd.DataFrame(all_scores).T[METRIC_NAMES]
    st.dataframe(comparison.round(4))
    best_model = comparison["F1"].idxmax()
    st.success(f"Highest macro-F1 on this upload: **{best_model}** ({comparison['F1'].max():.4f})")
    st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])

with tab_rows:
    shown = needed.copy()
    shown["Predicted"] = predicted
    shown["Correct"] = shown["Predicted"] == shown[TARGET_COLUMN]
    st.dataframe(shown)
    st.download_button(
        "Download predictions (CSV)",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
