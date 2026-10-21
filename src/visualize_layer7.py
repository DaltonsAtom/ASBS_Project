#!/usr/bin/env python3
"""
Layer 7 visualization: frequent-pattern mining benchmark analysis.

Reads:
  experiments/results/frequent_mining_benchmark.csv
  experiments/results/frequent_mining_benchmark_summary.json (optional)

Writes:
  layer7_runtime_comparison.png
  layer7_itemset_counts.png
  layer7_runtime_speedup.png (when comparable rows are available)
  layer7_runtime_vs_itemsets.png (when usable columns are available)

Run from the project root:
  python src/visualize_layer7.py

Optional output override: ASBD_FIGURES_DIR
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


def locate_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), here.parent, here.parent.parent, here.parent.parent.parent]:
        if (candidate / "experiments" / "results" / "frequent_mining_benchmark.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find experiments/results/frequent_mining_benchmark.csv. "
        "Place this script in src/ and run it from your project root."
    )


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    # Allow minor naming variations.
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for candidate in candidates:
            ckey = candidate.lower().replace(" ", "_").replace("-", "_")
            if key == ckey:
                return col
    if required:
        raise ValueError(f"Required column not found. Tried {candidates}; available columns: {list(df.columns)}")
    return None


def main() -> int:
    root = locate_root()
    results = root / "experiments" / "results"
    source = results / "frequent_mining_benchmark.csv"
    summary_path = results / "frequent_mining_benchmark_summary.json"
    out_dir = Path(os.environ.get("ASBD_FIGURES_DIR", str(root / "reports" / "figures")))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    if df.empty:
        raise ValueError(f"{source} contains no benchmark rows.")

    algorithm_col = find_col(df, ["algorithm", "method", "miner", "model"])
    runtime_col = find_col(df, ["runtime_seconds", "runtime_sec", "runtime", "time_seconds", "time_sec", "time_s", "elapsed_seconds", "execution_time_seconds", "time"])
    count_col = find_col(
        df,
        ["itemset_count", "frequent_itemsets", "num_itemsets", "patterns_found",
         "pattern_count", "n_itemsets", "itemsets_count", "output_itemsets"],
        required=False,
    )
    support_col = find_col(df, ["min_support", "support_threshold", "support", "support_pct", "minimum_support"], required=False)
    maxlen_col = find_col(df, ["max_length", "max_itemset_length", "max_pattern_length"], required=False)

    df[runtime_col] = pd.to_numeric(df[runtime_col], errors="coerce")
    if count_col is not None:
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    else:
        # Some benchmark outputs store the number of itemsets as a path/name
        # in output_itemsets rather than a numeric count. In that case, plot
        # runtime comparisons and report pattern counts as unavailable.
        print(
            "Warning: no numeric itemset-count column was found. "
            "Runtime plots will be generated; itemset-count plots will be skipped. "
            "The 'output_itemsets' field appears to identify an output artifact, "
            "not the number of patterns."
        )
    required_drop = [algorithm_col, runtime_col]
    if count_col is not None:
        required_drop.append(count_col)
    df = df.dropna(subset=required_drop).copy()
    df = df[df[runtime_col] >= 0]
    if count_col is not None:
        df = df[df[count_col] >= 0]
    if df.empty:
        raise ValueError("No valid benchmark rows after converting runtime and itemset counts to numeric values.")

    # Make experiment condition labels from support and max length if present.
    if support_col:
        def support_label(value):
            if pd.isna(value):
                return "support unknown"
            if isinstance(value, (float, np.floating)) and 0 < value < 1:
                return f"support={value:.1%}"
            return f"support={value:g}"
        df["_condition"] = df[support_col].map(support_label)
    else:
        df["_condition"] = "all settings"
    if maxlen_col:
        df["_condition"] = df.apply(
            lambda row: f"{row['_condition']}, maxlen={row[maxlen_col]}" if pd.notna(row[maxlen_col]) else row["_condition"],
            axis=1
        )

    # 1. Runtime comparison, grouped by experimental setting.
    conditions = list(df["_condition"].drop_duplicates())
    algorithms = list(df[algorithm_col].astype(str).drop_duplicates())
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(algorithms))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, algorithm in enumerate(algorithms):
        subset = df[df[algorithm_col].astype(str) == algorithm].set_index("_condition")
        vals = [float(subset.loc[c, runtime_col]) if c in subset.index and np.isscalar(subset.loc[c, runtime_col]) else np.nan for c in conditions]
        # If duplicated algorithm-condition rows exist, use their mean.
        vals = []
        for c in conditions:
            v = pd.to_numeric(df.loc[(df[algorithm_col].astype(str) == algorithm) & (df["_condition"] == c), runtime_col], errors="coerce")
            vals.append(float(v.mean()) if not v.empty else np.nan)
        positions = x - 0.4 + width/2 + i*width
        bars = ax.bar(positions, vals, width=width, label=algorithm)
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_title("Layer 7: Runtime Comparison Across Mining Algorithms")
    ax.set_xlabel("Benchmark setting")
    ax.set_ylabel("Runtime (seconds; lower is faster)")
    ax.set_xticks(x, conditions, rotation=15, ha="right")
    ax.legend(title="Algorithm")
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "layer7_runtime_comparison.png", dpi=180)
    plt.close(fig)

    # 2. Pattern counts are only plotted when the CSV has a numeric count column.
    if count_col is not None:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for algorithm in algorithms:
            sub = df[df[algorithm_col].astype(str) == algorithm]
            ax.plot(sub["_condition"], sub[count_col], marker="o", linewidth=1.8, label=algorithm)
            for _, row in sub.iterrows():
                ax.annotate(f"{int(row[count_col])}", (row["_condition"], row[count_col]),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_title("Layer 7: Frequent Itemset Counts by Algorithm and Setting")
        ax.set_xlabel("Benchmark setting")
        ax.set_ylabel("Number of itemsets / patterns")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Algorithm")
        fig.tight_layout()
        fig.savefig(out_dir / "layer7_itemset_counts.png", dpi=180)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.axis("off")
        ax.text(0.5, 0.62, "Itemset counts unavailable in benchmark CSV",
                ha="center", va="center", fontsize=15)
        ax.text(0.5, 0.42,
                "The output_itemsets column identifies output artifacts; it is not treated as a count.\n"
                "Add a numeric itemset_count column to enable this figure.",
                ha="center", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "layer7_itemset_counts.png", dpi=180)
        plt.close(fig)

    # 3. Speedup relative to the slowest runtime in each setting; descriptive, not a correctness metric.
    pivot = df.pivot_table(index="_condition", columns=algorithm_col, values=runtime_col, aggfunc="mean")
    fastest = pivot.min(axis=1)
    speed = pivot.div(fastest, axis=0)  # runtime / fastest runtime; values >= 1
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for algorithm in speed.columns:
        ax.plot(speed.index, speed[algorithm], marker="o", label=str(algorithm))
    ax.set_title("Layer 7: Runtime Ratio Relative to the Fastest Algorithm")
    ax.set_xlabel("Benchmark setting")
    ax.set_ylabel("Runtime / fastest runtime (1 = fastest)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Algorithm")
    fig.tight_layout()
    fig.savefig(out_dir / "layer7_runtime_speedup.png", dpi=180)
    plt.close(fig)

    # 4. Runtime versus pattern count, when counts are available.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if count_col is not None:
        for algorithm in algorithms:
            sub = df[df[algorithm_col].astype(str) == algorithm]
            ax.scatter(sub[count_col], sub[runtime_col], s=65, alpha=0.8, label=algorithm)
            for _, row in sub.iterrows():
                ax.annotate(str(row["_condition"]), (row[count_col], row[runtime_col]),
                            textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_title("Layer 7: Runtime vs. Number of Patterns Found")
        ax.set_xlabel("Itemset / pattern count")
        ax.set_ylabel("Runtime (seconds)")
        ax.legend(title="Algorithm")
    else:
        ax.axis("off")
        ax.text(0.5, 0.6, "Runtime vs. itemset count cannot be plotted",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.text(0.5, 0.42, "The benchmark CSV does not contain numeric pattern counts.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "layer7_runtime_vs_itemsets.png", dpi=180)
    plt.close(fig)

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print("Benchmark summary:")
            print(json.dumps(summary, indent=2)[:2500])
        except (json.JSONDecodeError, OSError):
            print("Warning: summary JSON could not be parsed; figures were generated from the CSV.")

    print(f"Benchmark rows visualized: {len(df)}")
    if count_col is None:
        print("Itemset-count analyses were skipped because no numeric count column exists.")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Figures saved to: {out_dir}")
    for name in [
        "layer7_runtime_comparison.png", "layer7_itemset_counts.png",
        "layer7_runtime_speedup.png", "layer7_runtime_vs_itemsets.png"
    ]:
        print(f"  - {name}")
    print("Note: runtime comparisons depend on hardware and execution conditions.")
    print("Matching itemset counts alone do not establish algorithmic correctness.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
