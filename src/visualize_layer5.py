#!/usr/bin/env python3
"""
Layer 5 visualization: RFM customer segmentation.

Reads:
  experiments/results/customer_rfm_segments.csv
  experiments/results/rfm_cluster_profiles.csv
  experiments/results/rfm_clustering_metrics.json (optional)

Writes meaningful, data-driven plots to reports/figures/:
  layer5_segment_sizes.png
  layer5_rfm_cluster_profiles.png
  layer5_rfm_distributions.png
  layer5_recency_vs_monetary.png

Run from the project root:
  python src/visualize_layer5.py

Optional output override:
  Windows PowerShell: $env:ASBD_FIGURES_DIR="reports/figures"; python src/visualize_layer5.py
  macOS/Linux: ASBD_FIGURES_DIR=reports/figures python src/visualize_layer5.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEGMENT_CANDIDATES = [
    "Segment", "RFM_Segment", "Cluster", "Cluster_Label", "Customer_Segment",
    "KMeans_Cluster", "cluster"
]
RECENCY_CANDIDATES = ["Recency", "recency", "Recency_Days", "RecencyDays"]
FREQUENCY_CANDIDATES = ["Frequency", "frequency", "Purchase_Frequency", "Invoice_Count", "Frequency_Score"]
MONETARY_CANDIDATES = ["Monetary", "monetary", "Monetary_Value", "Total_Spend", "TotalSpend", "Customer_Spend"]


def locate_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), here.parent, here.parent.parent, here.parent.parent.parent]
    for candidate in candidates:
        if (candidate / "experiments" / "results" / "customer_rfm_segments.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find experiments/results/customer_rfm_segments.csv. "
        "Place this script in src/ and run it from your project root, or set the project layout correctly."
    )


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {str(c).lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    if required:
        raise ValueError(
            f"Could not identify a required column from {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def numeric_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    col = pick_column(df, candidates)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].notna().sum() == 0:
        raise ValueError(f"Column '{col}' has no usable numeric values for {label}.")
    return col


def main() -> int:
    root = locate_root()
    results = root / "experiments" / "results"
    out_dir = Path(os.environ.get("ASBD_FIGURES_DIR", str(root / "reports" / "figures")))
    out_dir.mkdir(parents=True, exist_ok=True)

    segments_path = results / "customer_rfm_segments.csv"
    profiles_path = results / "rfm_cluster_profiles.csv"
    metrics_path = results / "rfm_clustering_metrics.json"

    customers = pd.read_csv(segments_path)
    profiles = pd.read_csv(profiles_path) if profiles_path.exists() else None

    segment_col = pick_column(customers, SEGMENT_CANDIDATES)
    recency_col = numeric_col(customers, RECENCY_CANDIDATES, "recency")
    frequency_col = numeric_col(customers, FREQUENCY_CANDIDATES, "frequency")
    monetary_col = numeric_col(customers, MONETARY_CANDIDATES, "monetary")

    customers = customers.dropna(subset=[segment_col])
    if customers.empty:
        raise ValueError("No customers with a valid segment/cluster label were found.")

    # 1. Segment population: counts and percentages.
    counts = customers[segment_col].astype(str).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(counts.index, counts.values)
    ax.set_title("Layer 5: Customer Count by RFM Segment")
    ax.set_xlabel("Segment / cluster")
    ax.set_ylabel("Number of customers")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:,}\n({value/len(customers):.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "layer5_segment_sizes.png", dpi=180)
    plt.close(fig)

    # 2. Cluster profiles from the actual profile file when available; otherwise aggregate customers.
    profile = None
    if profiles is not None and not profiles.empty:
        profile_segment = pick_column(profiles, SEGMENT_CANDIDATES, required=False)
        if profile_segment:
            profile = profiles.set_index(profile_segment)
            # Prefer matching profile columns; map the source fields if present.
            wanted = {}
            for metric, candidates in [
                ("Recency", RECENCY_CANDIDATES),
                ("Frequency", FREQUENCY_CANDIDATES),
                ("Monetary", MONETARY_CANDIDATES),
            ]:
                col = pick_column(profiles, candidates, required=False)
                if col:
                    wanted[metric] = col
            if wanted:
                profile = profile[list(wanted.values())].rename(columns={v: k for k, v in wanted.items()})
            else:
                profile = None
    if profile is None:
        profile = customers.groupby(segment_col)[[recency_col, frequency_col, monetary_col]].mean()
        profile.columns = ["Recency", "Frequency", "Monetary"]
    profile = profile.apply(pd.to_numeric, errors="coerce")
    # Scale columns independently for readability because monetary/frequency/recency use different units.
    scaled = profile.copy()
    for col in scaled.columns:
        lo, hi = scaled[col].min(), scaled[col].max()
        scaled[col] = (scaled[col] - lo) / (hi - lo) if pd.notna(lo) and pd.notna(hi) and hi > lo else 0.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(scaled.to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Layer 5: Relative RFM Profile by Segment (min–max scaled per metric)")
    ax.set_xlabel("RFM metric")
    ax.set_ylabel("Segment / cluster")
    ax.set_xticks(range(len(scaled.columns)), scaled.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(scaled.index)), [str(x) for x in scaled.index])
    fig.colorbar(im, ax=ax, label="Relative level within each metric (0–1)")
    fig.tight_layout()
    fig.savefig(out_dir / "layer5_rfm_cluster_profiles.png", dpi=180)
    plt.close(fig)

    # 3. Distributions: log1p makes highly skewed frequency/spend easier to inspect.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, col, label in zip(
        axes, [recency_col, frequency_col, monetary_col],
        ["Recency (days)", "Frequency", "Monetary spend"]
    ):
        values = pd.to_numeric(customers[col], errors="coerce").dropna()
        values = values[values >= 0]
        ax.hist(np.log1p(values), bins=35)
        ax.set_title(label)
        ax.set_xlabel("log(1 + value)")
        ax.set_ylabel("Customers")
    fig.suptitle("Layer 5: RFM Feature Distributions (log-scaled values)")
    fig.tight_layout()
    fig.savefig(out_dir / "layer5_rfm_distributions.png", dpi=180)
    plt.close(fig)

    # 4. Recency vs monetary: log scale reduces outlier compression; sample only if very large.
    plot_df = customers[[segment_col, recency_col, monetary_col]].copy()
    plot_df[recency_col] = pd.to_numeric(plot_df[recency_col], errors="coerce")
    plot_df[monetary_col] = pd.to_numeric(plot_df[monetary_col], errors="coerce")
    plot_df = plot_df.dropna()
    plot_df = plot_df[(plot_df[recency_col] >= 0) & (plot_df[monetary_col] >= 0)]
    if len(plot_df) > 15000:
        plot_df = plot_df.sample(15000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 6))
    for segment, group in plot_df.groupby(segment_col):
        ax.scatter(group[recency_col], np.log1p(group[monetary_col]), s=15, alpha=0.55, label=str(segment))
    ax.set_title("Layer 5: Recency vs Monetary Spend by Segment")
    ax.set_xlabel("Recency (days; lower is more recent)")
    ax.set_ylabel("log(1 + monetary spend)")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "layer5_recency_vs_monetary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print("RFM metrics summary:")
            print(json.dumps(metrics, indent=2)[:2500])
        except (json.JSONDecodeError, OSError):
            print("Warning: could not parse rfm_clustering_metrics.json; plots were generated from CSV files.")

    print(f"Customers visualized: {len(customers):,}")
    print(f"Segments found: {customers[segment_col].nunique()}")
    print(f"Figures saved to: {out_dir}")
    for name in [
        "layer5_segment_sizes.png", "layer5_rfm_cluster_profiles.png",
        "layer5_rfm_distributions.png", "layer5_recency_vs_monetary.png"
    ]:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
