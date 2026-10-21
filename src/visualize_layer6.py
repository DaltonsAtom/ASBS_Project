"""
Layer 6 — Classification Results Visualization

Reads the outputs created by src/customer_classification.py and creates
interpretable figures for the report. It does not retrain the classifiers.

Required inputs:
    experiments/results/classification_model_comparison.csv
    experiments/results/classification_test_predictions.csv
    experiments/results/classification_summary.json

Generated figures:
    reports/figures/layer6_model_metrics.png
    reports/figures/layer6_confusion_matrices.png
    reports/figures/layer6_roc_curves.png
    reports/figures/layer6_prediction_probability_distribution.png

Run from the project root:
    python src/visualize_layer6.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def find_project_root() -> Path:
    script_path = Path(__file__).resolve()

    # Recommended location: <project-root>/src/visualize_layer6.py
    if script_path.parent.name.lower() == "src":
        candidate = script_path.parent.parent
        if (candidate / "experiments" / "results").is_dir():
            return candidate

    # If run from elsewhere, prefer an ancestor containing the actual inputs.
    for candidate in script_path.parents:
        if (candidate / "experiments" / "results" / "classification_model_comparison.csv").is_file():
            return candidate

    # Downloaded script placed beside the project folder.
    sibling = script_path.parent / "customer-behaviour-analytics"
    if (sibling / "experiments" / "results").is_dir():
        return sibling

    raise FileNotFoundError(
        "Could not locate the project root. Save this script as "
        "<project-root>/src/visualize_layer6.py and run it from the project root."
    )


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = Path(os.environ.get(
    "ASBD_FIGURES_DIR", str(PROJECT_ROOT / "reports" / "figures")
))

COMPARISON_PATH = RESULTS_DIR / "classification_model_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "classification_test_predictions.csv"
SUMMARY_PATH = RESULTS_DIR / "classification_summary.json"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Run src/customer_classification.py first and confirm the output files exist."
        )


def load_inputs():
    for path in (COMPARISON_PATH, PREDICTIONS_PATH, SUMMARY_PATH):
        require_file(path)

    comparison = pd.read_csv(COMPARISON_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    required_metrics = {
        "model", "accuracy", "precision_high_spender",
        "recall_high_spender", "f1_high_spender", "roc_auc"
    }
    missing_metrics = required_metrics - set(comparison.columns)
    if missing_metrics:
        raise ValueError(
            f"classification_model_comparison.csv is missing columns: "
            f"{sorted(missing_metrics)}"
        )

    if "ActualHighSpender" not in predictions.columns:
        raise ValueError(
            "classification_test_predictions.csv must contain ActualHighSpender."
        )

    if comparison.empty or predictions.empty:
        raise ValueError("Classification input CSVs must not be empty.")

    if not set(predictions["ActualHighSpender"].dropna().unique()).issubset({0, 1}):
        raise ValueError("ActualHighSpender must contain binary labels 0 and 1.")

    model_columns = {
        "Logistic Regression": ("Logistic_Regression", "Logistic_Regression_Probability"),
        "Decision Tree": ("Decision_Tree", "Decision_Tree_Probability"),
        "Random Forest": ("Random_Forest", "Random_Forest_Probability"),
    }
    return comparison, predictions, summary, model_columns


def save_metric_comparison(comparison: pd.DataFrame) -> None:
    metric_columns = [
        ("accuracy", "Accuracy"),
        ("precision_high_spender", "Precision (high spender)"),
        ("recall_high_spender", "Recall (high spender)"),
        ("f1_high_spender", "F1-score (high spender)"),
        ("roc_auc", "ROC-AUC"),
    ]
    long = comparison.melt(
        id_vars="model",
        value_vars=[column for column, _ in metric_columns],
        var_name="metric",
        value_name="score",
    )
    metric_labels = dict(metric_columns)
    long["metric"] = long["metric"].map(metric_labels)
    pivot = long.pivot(index="model", columns="metric", values="score")
    ordered_labels = [label for _, label in metric_columns if label in pivot.columns]
    pivot = pivot[ordered_labels]

    ax = pivot.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), width=0.78)
    ax.set_title("Layer 6: Classification Model Performance")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "layer6_model_metrics.png", dpi=170)
    plt.close(ax.figure)


def save_confusion_matrices(predictions: pd.DataFrame, model_columns: dict) -> None:
    available = [
        (name, columns[0])
        for name, columns in model_columns.items()
        if columns[0] in predictions.columns
    ]
    if not available:
        raise ValueError("No model prediction columns found in classification_test_predictions.csv.")

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, (name, pred_col) in zip(axes, available):
        y_true = predictions["ActualHighSpender"].astype(int)
        y_pred = predictions[pred_col].astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        image = ax.imshow(matrix, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks([0, 1], labels=["Not high", "High"])
        ax.set_yticks([0, 1], labels=["Not high", "High"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
    fig.suptitle("Layer 6: Confusion Matrices on the Test Set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "layer6_confusion_matrices.png", dpi=170)
    plt.close(fig)


def save_roc_curves(predictions: pd.DataFrame, model_columns: dict) -> None:
    y_true = predictions["ActualHighSpender"].astype(int)
    if y_true.nunique() != 2:
        print("Skipping ROC curves: test labels contain only one class.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0
    for name, (_, probability_col) in model_columns.items():
        if probability_col not in predictions.columns:
            continue
        scores = pd.to_numeric(predictions[probability_col], errors="coerce")
        valid = scores.notna()
        if valid.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[valid], scores[valid])
        auc = roc_auc_score(y_true[valid], scores[valid])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plotted += 1

    if not plotted:
        plt.close(fig)
        print("Skipping ROC curves: no model probability columns were found.")
        return

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("Layer 6: ROC Curves for High-Spender Classification")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "layer6_roc_curves.png", dpi=170)
    plt.close(fig)


def save_probability_distributions(predictions: pd.DataFrame, model_columns: dict) -> None:
    y_true = predictions["ActualHighSpender"].astype(int)
    available = [
        (name, columns[1])
        for name, columns in model_columns.items()
        if columns[1] in predictions.columns
    ]
    if not available:
        print("Skipping probability distributions: no probability columns were found.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5), sharey=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (name, probability_col) in zip(axes, available):
        probabilities = pd.to_numeric(predictions[probability_col], errors="coerce")
        for label, class_name in ((0, "Not high spender"), (1, "High spender")):
            values = probabilities[y_true == label].dropna()
            if len(values):
                ax.hist(values, bins=15, alpha=0.55, label=class_name)
        ax.set_title(name)
        ax.set_xlabel("Predicted probability of high spender")
        ax.set_ylabel("Number of test customers")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle("Layer 6: Predicted Probability Distributions by Actual Class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "layer6_prediction_probability_distribution.png", dpi=170)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison, predictions, summary, model_columns = load_inputs()

    save_metric_comparison(comparison)
    save_confusion_matrices(predictions, model_columns)
    save_roc_curves(predictions, model_columns)
    save_probability_distributions(predictions, model_columns)

    print("LAYER 6 VISUALIZATION: COMPLETED")
    print(f"Models in comparison table: {len(comparison)}")
    print(f"Test predictions: {len(predictions):,}")
    print(f"Target distribution in test set: "
          f"{predictions['ActualHighSpender'].value_counts().sort_index().to_dict()}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("Generated figures:")
    for filename in (
        "layer6_model_metrics.png",
        "layer6_confusion_matrices.png",
        "layer6_roc_curves.png",
        "layer6_prediction_probability_distribution.png",
    ):
        path = FIGURES_DIR / filename
        if path.exists():
            print(f"  {filename}")


if __name__ == "__main__":
    main()
