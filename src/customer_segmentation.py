"""
Layer 5 — Customer Segmentation with RFM and K-Means
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Input:
    data/processed/customer_purchase_lines.csv

Outputs:
    experiments/results/customer_rfm_segments.csv
    experiments/results/rfm_cluster_profiles.csv
    experiments/results/rfm_clustering_metrics.json
    reports/figures/rfm_clusters.png
    reports/figures/rfm_cluster_sizes.png

RFM:
    Recency   = days since last purchase, relative to one day after latest dataset date
    Frequency = number of distinct invoices
    Monetary  = sum of LineValue

Run from project root:
    python src/customer_segmentation.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_purchase_lines.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Required input not found: {INPUT_PATH}\n"
            "Run `python src/preprocessing.py` first."
        )

    data = pd.read_csv(
        INPUT_PATH,
        dtype={"InvoiceNo": "string", "CustomerID": "string"},
        parse_dates=["InvoiceDate"],
    )
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "LineValue"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Customer dataset is missing columns: {sorted(missing)}")

    data["LineValue"] = pd.to_numeric(data["LineValue"], errors="coerce")
    data = data.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "LineValue"])
    data = data[data["CustomerID"].str.strip().ne("")].copy()
    if data.empty:
        raise ValueError("No usable customer purchase records remain.")

    # Snapshot is one day after the latest observed timestamp, not the machine's current date.
    snapshot_date = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = data.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineValue", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"].dt.normalize()).dt.days
    rfm = rfm[["Recency", "Frequency", "Monetary"]].replace([float("inf"), -float("inf")], pd.NA)
    rfm = rfm.dropna()
    # K-Means requires finite numeric values. Monetary should be non-negative by Layer 2 policy.
    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Frequency"] > 0) & (rfm["Monetary"] >= 0)].copy()
    if len(rfm) < 3:
        raise ValueError("At least three valid customers are required for clustering.")

    # Log transform reduces the influence of strongly right-skewed Frequency and Monetary.
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Frequency"] = features["Frequency"].astype(float).map(lambda x: __import__("math").log1p(x))
    features["Monetary"] = features["Monetary"].astype(float).map(lambda x: __import__("math").log1p(x))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Choose k using silhouette score among feasible candidate values.
    max_k = min(6, len(rfm) - 1)
    candidate_ks = list(range(2, max_k + 1))
    scores = []
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        # Silhouette score is undefined if all observations fall in one cluster.
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            score = float(silhouette_score(scaled, labels))
            scores.append({"k": k, "silhouette_score": score, "inertia": float(model.inertia_)})
    if not scores:
        raise RuntimeError("Could not obtain a valid silhouette score for k in [2, 6].")

    best = max(scores, key=lambda row: row["silhouette_score"])
    best_k = int(best["k"])
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = final_model.fit_predict(scaled)

    # Human-readable profiles use original, untransformed RFM values.
    profiles = rfm.groupby("Cluster").agg(
        CustomerCount=("Recency", "size"),
        MeanRecencyDays=("Recency", "mean"),
        MedianRecencyDays=("Recency", "median"),
        MeanFrequency=("Frequency", "mean"),
        MedianFrequency=("Frequency", "median"),
        MeanMonetary=("Monetary", "mean"),
        MedianMonetary=("Monetary", "median"),
    ).round(3)
    profiles["CustomerSharePercent"] = (profiles["CustomerCount"] / len(rfm) * 100).round(2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    customer_output = rfm.reset_index()
    customer_output["SnapshotDate"] = snapshot_date.strftime("%Y-%m-%d")
    customer_output.to_csv(RESULTS_DIR / "customer_rfm_segments.csv", index=False, encoding="utf-8")
    profiles.reset_index().to_csv(RESULTS_DIR / "rfm_cluster_profiles.csv", index=False, encoding="utf-8")

    # Scatter plot uses Recency and Monetary; colour denotes K-Means cluster.
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id, group in customer_output.groupby("Cluster"):
        ax.scatter(
            group["Recency"], group["Monetary"],
            s=18, alpha=0.65, label=f"Cluster {cluster_id}"
        )
    ax.set_title("Customer Segments: Recency vs Monetary Value")
    ax.set_xlabel("Recency (days; lower means more recent)")
    ax.set_ylabel("Monetary value (dataset currency)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rfm_clusters.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    profiles["CustomerCount"].sort_index().plot(kind="bar", ax=ax)
    ax.set_title(f"Customer Count by K-Means Cluster (k={best_k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rfm_cluster_sizes.png", dpi=160)
    plt.close(fig)

    summary = {
        "layer": "Layer 5 — Customer Segmentation with RFM and K-Means",
        "input_file": str(INPUT_PATH),
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "customer_count": int(len(rfm)),
        "rfm_definitions": {
            "Recency": "Days between the customer's last observed purchase and snapshot date.",
            "Frequency": "Number of distinct invoices per customer in the available observation period.",
            "Monetary": "Sum of LineValue per customer in the available observation period."
        },
        "preprocessing_for_clustering": [
            "Frequency and Monetary were transformed with log1p to reduce right skew.",
            "RFM features were standardized before K-Means.",
            "Candidate k values from 2 to 6 were compared using silhouette score.",
            "K-Means used random_state=42 and n_init=10 for reproducibility."
        ],
        "candidate_k_scores": scores,
        "selected_k": best_k,
        "selected_silhouette_score": best["silhouette_score"],
        "cluster_profiles": profiles.reset_index().to_dict(orient="records"),
        "outputs": [
            "experiments/results/customer_rfm_segments.csv",
            "experiments/results/rfm_cluster_profiles.csv",
            "reports/figures/rfm_clusters.png",
            "reports/figures/rfm_cluster_sizes.png"
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "caveats": [
            "Cluster IDs are arbitrary labels, not ordered customer-value rankings.",
            "RFM values summarize the observed dataset period and are not a future prediction.",
            "Silhouette score is a heuristic for separation and should be considered alongside interpretability.",
            "This is single-machine clustering; Spark integration is reserved for Version 2."
        ]
    }
    (RESULTS_DIR / "rfm_clustering_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("LAYER 5 — CUSTOMER SEGMENTATION: PASSED")
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Customers clustered: {len(rfm):,}")
    print(f"Selected k: {best_k}")
    print(f"Silhouette score: {best['silhouette_score']:.4f}")
    print("Cluster profiles:")
    print(profiles.to_string())
    print(f"Segments: {RESULTS_DIR / 'customer_rfm_segments.csv'}")
    print(f"Profiles: {RESULTS_DIR / 'rfm_cluster_profiles.csv'}")
    print(f"Summary: {RESULTS_DIR / 'rfm_clustering_metrics.json'}")


if __name__ == "__main__":
    main()
