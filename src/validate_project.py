"""
Layer 8 — Version 1 Project Output Validation

Checks that the expected pipeline outputs exist and that core CSV/JSON
artifacts can be read. This is a structural integrity check, not a substitute
for reviewing the algorithms or their scientific validity.

Run from project root:
    python src/validate_project.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/processed/customer_purchase_lines.csv",
    "data/transactions/transactions.csv",
    "experiments/results/apriori_itemsets.csv",
    "experiments/results/fp_growth_itemsets.csv",
    "experiments/results/charm_closed_itemsets.csv",
    "experiments/results/association_rules.csv",
    "experiments/results/customer_rfm_segments.csv",
    "experiments/results/rfm_cluster_profiles.csv",
    "experiments/results/rfm_clustering_metrics.json",
    "experiments/results/classification_model_comparison.csv",
    "experiments/results/classification_test_predictions.csv",
    "experiments/results/classification_summary.json",
    "experiments/results/frequent_mining_benchmark.csv",
    "experiments/results/frequent_mining_benchmark_summary.json",
    "reports/figures/frequent_mining_runtime.png",
    "reports/figures/frequent_mining_itemset_counts.png",
    "reports/figures/classification_model_comparison.png",
]


def main():
    missing = [relative for relative in REQUIRED_FILES if not (PROJECT_ROOT / relative).is_file()]
    print("LAYER 8 — VERSION 1 PROJECT OUTPUT VALIDATION")
    print(f"Required artifacts: {len(REQUIRED_FILES)}")
    print(f"Missing artifacts: {len(missing)}")
    for relative in missing:
        print(f"  MISSING: {relative}")

    csv_checks = [
        ("experiments/results/frequent_mining_benchmark.csv",
         {"algorithm", "minimum_support", "output_itemsets", "runtime_seconds"}),
        ("experiments/results/classification_model_comparison.csv",
         {"model", "accuracy", "precision_high_spender", "recall_high_spender", "f1_high_spender", "roc_auc"}),
        ("experiments/results/customer_rfm_segments.csv", {"CustomerID", "Cluster"}),
        ("experiments/results/rfm_cluster_profiles.csv", {"Cluster", "CustomerCount"}),
    ]
    failures = []
    for relative, required_columns in csv_checks:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            failures.append(f"{relative}: file missing")
            continue
        try:
            frame = pd.read_csv(path)
            absent = required_columns - set(frame.columns)
            if absent:
                failures.append(f"{relative}: missing columns {sorted(absent)}")
            elif frame.empty:
                failures.append(f"{relative}: CSV is empty")
            else:
                print(f"  CSV PASS: {relative} ({len(frame):,} rows)")
        except Exception as exc:
            failures.append(f"{relative}: cannot read CSV ({exc})")

    json_checks = [
        "experiments/results/frequent_mining_benchmark_summary.json",
        "experiments/results/classification_summary.json",
        "experiments/results/rfm_clustering_metrics.json",
    ]
    for relative in json_checks:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            failures.append(f"{relative}: file missing")
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
            print(f"  JSON PASS: {relative}")
        except Exception as exc:
            failures.append(f"{relative}: invalid JSON ({exc})")

    if missing or failures:
        print("\nVALIDATION STATUS: NEEDS ATTENTION")
        for failure in failures:
            print(f"  FAIL: {failure}")
        raise SystemExit(1)

    print("\nVALIDATION STATUS: STRUCTURAL CHECKS PASSED")
    print("Review benchmark methodology and interpretation separately before final submission.")


if __name__ == "__main__":
    main()
