"""Train and evaluate the five classifiers on the Obesity Levels dataset.

Run:  python train_models.py
Outputs (all written next to this file):
    model/*.joblib          one fitted preprocessing+classifier pipeline per model
    model/model_summary.json  held-out metrics, best hyper-parameters, class list
    model/test_metrics.csv    the comparison table used in the README
    test_data.csv             the held-out 20 % split (14 features + target)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    METRIC_NAMES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    compute_all_metrics,
)

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_DIR = ROOT / "model"
SPLIT_SEED = 2026
TEST_FRACTION = 0.20

# display name -> file stem
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def build_preprocessor():
    """Scale numeric columns, one-hot encode categorical ones (dense output)."""
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def candidate_models():
    """(estimator, hyper-parameter grid) for each of the five models."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.1, 1, 10, 100, 1000]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SPLIT_SEED),
            {"clf__max_depth": [5, 8, 12, None], "clf__min_samples_leaf": [1, 3, 5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9, 15], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]},
        ),
        "Support Vector Machine": (
            SVC(kernel="rbf", probability=True, random_state=SPLIT_SEED),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 0.3, 1.0]},
        ),
    }


def load_clean_data():
    raw = pd.read_csv(RAW_CSV)
    n_before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)  # identical rows would leak across the split
    print(f"Rows: {n_before} -> {len(raw)} after dropping {n_before - len(raw)} exact duplicates")
    return raw[FEATURE_COLUMNS + [TARGET_COLUMN]]  # Height and Weight are left out on purpose


def main():
    data = load_clean_data()
    features, target = data[FEATURE_COLUMNS], data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_FRACTION, stratify=target, random_state=SPLIT_SEED
    )
    print(f"Train: {len(x_train)} rows | Test: {len(x_test)} rows | Features: {len(FEATURE_COLUMNS)}")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)

    rows, summary = [], {}
    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(x_train, y_train)  # tuning uses the training split only
        best = search.best_estimator_

        metrics = compute_all_metrics(best, x_test, y_test)
        joblib.dump(best, MODEL_DIR / f"{MODEL_FILES[name]}.joblib")

        rows.append({"ML Model Name": name, **metrics})
        summary[name] = {
            "file": f"{MODEL_FILES[name]}.joblib",
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1_macro": round(float(search.best_score_), 4),
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        }
        print(f"{name:<24} best={summary[name]['best_params']}  cv_f1={search.best_score_:.4f}")

    results = pd.DataFrame(rows).set_index("ML Model Name")[METRIC_NAMES].round(4)
    results.to_csv(MODEL_DIR / "test_metrics.csv")

    meta = {
        "sklearn_version": sklearn.__version__,
        "split_seed": SPLIT_SEED,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "classes": sorted(target.unique().tolist()),
        "models": summary,
    }
    (MODEL_DIR / "model_summary.json").write_text(json.dumps(meta, indent=2))

    pd.concat([x_test, y_test], axis=1).reset_index(drop=True).to_csv(ROOT / "test_data.csv", index=False)

    print("\nHeld-out test metrics:\n")
    print(results.to_string())
    try:  # to_markdown needs the optional 'tabulate' package
        print("\n(markdown)\n")
        print(results.reset_index().to_markdown(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
