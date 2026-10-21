"""
Layer 6 — Leakage-Safe Customer Spending Classification
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Predicts whether a customer will be a high spender in the future outcome window.
Features are calculated only from the earlier observation window.

Observation window: through 2011-08-31
Outcome window:     2011-09-01 through 2011-12-09 (inclusive)

Target:
    HighSpender = 1 if future-period customer spend is at or above the 75th
    percentile of positive future spend among customers with future purchases;
    otherwise 0. Customers with no future purchases are class 0.

Run from project root:
    python src/customer_classification.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_purchase_lines.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

OBSERVATION_END = pd.Timestamp("2011-08-31 23:59:59")
OUTCOME_START = pd.Timestamp("2011-09-01 00:00:00")
OUTCOME_END = pd.Timestamp("2011-12-09 23:59:59")


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_high_spender": float(precision_score(y_test, pred, zero_division=0)),
        "recall_high_spender": float(recall_score(y_test, pred, zero_division=0)),
        "f1_high_spender": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, score)) if score is not None and y_test.nunique() == 2 else None,
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["Not high spender", "High spender"],
            output_dict=True, zero_division=0
        ),
    }
    return model, metrics, pred, score


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Required input not found: {INPUT_PATH}\n"
            "Run `python src/preprocessing.py` first."
        )

    data = pd.read_csv(
        INPUT_PATH,
        dtype={"CustomerID": "string", "InvoiceNo": "string"},
        parse_dates=["InvoiceDate"],
    )
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "LineValue"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    data["CustomerID"] = data["CustomerID"].str.strip()
    data["LineValue"] = pd.to_numeric(data["LineValue"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data = data.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "LineValue", "Quantity", "UnitPrice"])
    data = data[data["CustomerID"].ne("")].copy()

    observed = data[data["InvoiceDate"] <= OBSERVATION_END].copy()
    future = data[(data["InvoiceDate"] >= OUTCOME_START) & (data["InvoiceDate"] <= OUTCOME_END)].copy()
    if observed.empty or future.empty:
        raise ValueError("Observation or outcome window is empty; check date coverage.")

    # Feature engineering uses only the observation window.
    snapshot = OBSERVATION_END.normalize() + pd.Timedelta(days=1)
    features = observed.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineValue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        MeanUnitPrice=("UnitPrice", "mean"),
        DistinctProducts=("StockCode", "nunique"),
    )
    features["Recency"] = (snapshot - features["LastPurchaseDate"].dt.normalize()).dt.days
    features = features.drop(columns=["LastPurchaseDate"])

    # Outcome labels use only the later period. Customers with no later purchases get zero spend.
    future_spend = future.groupby("CustomerID")["LineValue"].sum()
    features["FutureMonetary"] = future_spend.reindex(features.index).fillna(0.0)
    positive_future_spend = features.loc[features["FutureMonetary"] > 0, "FutureMonetary"]
    if positive_future_spend.empty:
        raise ValueError("No positive future-period spend found; cannot define a high-spender threshold.")
    threshold = float(positive_future_spend.quantile(0.75))
    features["HighSpender"] = (features["FutureMonetary"] >= threshold).astype(int)

    # Target is never included in predictors.
    feature_columns = [
        "Recency", "Frequency", "Monetary", "TotalQuantity",
        "MeanUnitPrice", "DistinctProducts"
    ]
    X = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = features["HighSpender"]
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Target does not contain enough examples from both classes for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, feature_columns)])

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    predictions = pd.DataFrame({"CustomerID": X_test.index, "ActualHighSpender": y_test.values})

    for name, model in models.items():
        fitted, metrics, pred, score = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        metrics_rows.append({k: v for k, v in metrics.items() if k != "classification_report" and k != "confusion_matrix_labels_0_1"})
        predictions[name.replace(" ", "_")] = pred
        if score is not None:
            predictions[name.replace(" ", "_") + "_Probability"] = score

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "classification_model_comparison.csv", index=False, encoding="utf-8")
    predictions.to_csv(RESULTS_DIR / "classification_test_predictions.csv", index=False, encoding="utf-8")

    # Metric comparison chart; accuracy is shown alongside positive-class precision/recall/F1 and ROC-AUC.
    plot_metrics = metrics_df.set_index("model")[[
        "accuracy", "precision_high_spender", "recall_high_spender", "f1_high_spender", "roc_auc"
    ]]
    ax = plot_metrics.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Customer High-Spender Classification: Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "classification_model_comparison.png", dpi=160)
    plt.close(ax.figure)

    summary = {
        "layer": "Layer 6 — Customer Spending Classification",
        "input_file": str(INPUT_PATH),
        "observation_window": {
            "start": str(observed["InvoiceDate"].min()),
            "end": str(OBSERVATION_END),
        },
        "outcome_window": {
            "start": str(OUTCOME_START),
            "end": str(OUTCOME_END),
        },
        "customer_count_observation_window": int(len(features)),
        "future_spend_75th_percentile_threshold": threshold,
        "target_definition": "HighSpender=1 if future-window spend is >= the 75th percentile of positive future spend among customers with positive future spend; otherwise 0, including customers with no future-window purchases.",
        "features_used": feature_columns,
        "target_distribution": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        "split": {
            "method": "stratified random customer-level train/test split",
            "test_size": 0.20,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "models": metrics_rows,
        "outputs": [
            "experiments/results/classification_model_comparison.csv",
            "experiments/results/classification_test_predictions.csv",
            "reports/figures/classification_model_comparison.png"
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "caveats": [
            "All predictor features are computed only from the observation window; target spend is computed only from the later outcome window.",
            "The positive class is defined using a future-spend threshold, so this is a retrospective evaluation on this dataset, not a live deployment result.",
            "The customer-level random split evaluates generalization to held-out customers, not necessarily future calendar periods.",
            "A later-period holdout of customers and dates would be a useful extension if more historical data were available."
        ]
    }
    (RESULTS_DIR / "classification_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("LAYER 6 — CUSTOMER SPENDING CLASSIFICATION: PASSED")
    print(f"Observation rows (customers): {len(features):,}")
    print(f"Future high-spender threshold: {threshold:,.2f}")
    print(f"Target counts: {y.value_counts().sort_index().to_dict()}")
    print(f"Train/test customers: {len(X_train):,}/{len(X_test):,}")
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Metrics: {RESULTS_DIR / 'classification_model_comparison.csv'}")
    print(f"Summary: {RESULTS_DIR / 'classification_summary.json'}")


if __name__ == "__main__":
    main()
