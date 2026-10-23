
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

for folder in (DATA_DIR, MODEL_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load the public German Credit dataset from OpenML."""
    print("Loading public German Credit dataset from OpenML...")
    try:
        dataset = fetch_openml(
            name="credit-g",
            version=1,
            as_frame=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check your internet "
            "connection and try running the script again."
        ) from exc

    X = dataset.data.copy()
    y = pd.Series(dataset.target).astype(str).str.lower()

    # Target: good = 1 (creditworthy), bad = 0.
    y = y.map({"good": 1, "bad": 0})

    if y.isna().any():
        raise ValueError("Unexpected target labels in the dataset.")

    df = X.copy()
    df["creditworthy"] = y.astype(int)
    df.to_csv(DATA_DIR / "credit_dataset.csv", index=False)

    return X, y.astype(int)


def build_pipeline(X, classifier):
    """Preprocess numerical and categorical features safely."""
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns if col not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def main():
    X, y = load_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Creditworthy cases: {(y == 1).sum()}")
    print(f"Non-creditworthy cases: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, classifier in candidates.items():
        print(f"\nTraining {name}...")
        model = build_pipeline(X_train, classifier)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        results.append(metrics)
        trained_models[name] = model

        print(
            f"Accuracy: {metrics['accuracy']:.3f} | "
            f"F1: {metrics['f1_score']:.3f} | "
            f"ROC-AUC: {metrics['roc_auc']:.3f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by="roc_auc", ascending=False
    )
    results_df.to_csv(
        DATA_DIR / "model_comparison.csv", index=False
    )

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    joblib.dump(best_model, MODEL_DIR / "credit_scoring_model.joblib")

    report = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["Not creditworthy (bad)", "Creditworthy (good)"],
        zero_division=0,
        output_dict=True,
    )

    with open(DATA_DIR / "classification_report.json", "w") as file:
        json.dump(report, file, indent=4)

    predictions_df = X_test.copy()
    predictions_df["actual_creditworthy"] = y_test.to_numpy()
    predictions_df["predicted_creditworthy"] = best_predictions
    predictions_df["creditworthy_probability"] = best_probabilities
    predictions_df.to_csv(
        DATA_DIR / "test_predictions.csv", index=False
    )

    # Model comparison chart.
    ax = results_df.set_index("model")[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ].plot(kind="bar", figsize=(11, 6))

    ax.set_title("Credit Scoring Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Confusion matrix.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        labels=[0, 1],
        display_labels=["Bad", "Good"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # ROC curve.
    RocCurveDisplay.from_predictions(y_test, best_probabilities)
    plt.title(f"ROC Curve - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=160)
    plt.close()

    summary = {
        "project": "Credit Scoring Model",
        "dataset": "OpenML German Credit (credit-g), version 1",
        "target_definition": {
            "1": "good credit category",
            "0": "bad credit category",
        },
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "best_model_by_roc_auc": best_name,
        "best_model_metrics": results_df.iloc[0].to_dict(),
        "note": (
            "Educational benchmark only. This is not an official "
            "CIBIL score calculator or a lending decision system."
        ),
    }

    with open(DATA_DIR / "results_summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)
    print(f"Best model by ROC-AUC: {best_name}")
    print(results_df.to_string(index=False))
    print(f"\nSaved model: {MODEL_DIR / 'credit_scoring_model.joblib'}")
    print(f"Saved reports: {DATA_DIR}")
    print(f"Saved charts: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
