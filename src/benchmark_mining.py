"""
Layer 7 — Comparative Frequent-Pattern Mining Experiments

Runs Apriori, FP-Growth, and the project's CHARM-style closed-itemset miner
on the same transactions and support thresholds. The maximum itemset length
is held constant for controlled comparison.

Outputs:
    experiments/results/frequent_mining_benchmark.csv
    experiments/results/frequent_mining_benchmark_summary.json
    reports/figures/frequent_mining_runtime.png
    reports/figures/frequent_mining_itemset_counts.png

Run from project root:
    python src/benchmark_mining.py
"""

from __future__ import annotations

import json
import time
import tracemalloc
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "transactions" / "transactions.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUPPORT_THRESHOLDS = [0.02, 0.05]
MAX_ITEMSET_LENGTH = 3


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "src" / filename
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_call(function):
    # Measure runtime without tracing overhead, then make a separate pass for
    # peak traced Python allocation. This keeps the runtime metric cleaner.
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started

    tracemalloc.start()
    function()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak_bytes / (1024 * 1024)


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Transactions file not found: {INPUT_PATH}")

    apriori_module = load_module("apriori_module", "apriori.py")
    fp_module = load_module("fp_growth_module", "fp_growth.py")
    charm_module = load_module("charm_module", "charm.py")

    transactions = apriori_module.load_transactions(INPUT_PATH)
    vertical, charm_transaction_count = charm_module.load_vertical_tidsets(INPUT_PATH)
    if charm_transaction_count != len(transactions):
        raise ValueError(
            "Input transaction counts differ between algorithm loaders; "
            "benchmark stopped to avoid an unfair comparison."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    validation = []

    for min_support in SUPPORT_THRESHOLDS:
        min_count = max(1, int(min_support * len(transactions) + 0.999999999))

        # Apriori
        (a_rows, a_meta), a_seconds, a_peak_mb = measured_call(
            lambda: apriori_module.apriori(transactions, min_support, MAX_ITEMSET_LENGTH)
        )
        a_map = {
            tuple(json.loads(row["itemset"])): int(row["support_count"])
            for row in a_rows
        }
        rows.append({
            "algorithm": "Apriori",
            "minimum_support": min_support,
            "minimum_support_count": min_count,
            "maximum_itemset_length": MAX_ITEMSET_LENGTH,
            "output_itemsets": len(a_rows),
            "runtime_seconds": round(a_seconds, 4),
            "peak_traced_memory_mb": round(a_peak_mb, 3),
            "output_type": "All frequent itemsets",
        })

        # FP-Growth
        (f_rows, f_meta), f_seconds, f_peak_mb = measured_call(
            lambda: fp_module.fp_growth(transactions, min_support, MAX_ITEMSET_LENGTH)
        )
        f_map = {
            tuple(json.loads(row["itemset"])): int(row["support_count"])
            for row in f_rows
        }
        fp_matches_apriori = a_map == f_map
        validation.append({
            "minimum_support": min_support,
            "apriori_fp_growth_identical_itemsets_and_counts": fp_matches_apriori,
            "apriori_itemset_count": len(a_map),
            "fp_growth_itemset_count": len(f_map),
        })
        rows.append({
            "algorithm": "FP-Growth",
            "minimum_support": min_support,
            "minimum_support_count": min_count,
            "maximum_itemset_length": MAX_ITEMSET_LENGTH,
            "output_itemsets": len(f_rows),
            "runtime_seconds": round(f_seconds, 4),
            "peak_traced_memory_mb": round(f_peak_mb, 3),
            "output_type": "All frequent itemsets",
        })

        # CHARM-style closed frequent itemsets: same transactions and support,
        # but its output count is not directly comparable to all frequent sets.
        def run_charm():
            frequent = charm_module.mine_frequent_itemsets(
                vertical, min_count, MAX_ITEMSET_LENGTH
            )
            return charm_module.retain_closed_itemsets(frequent), len(frequent)

        (closed_map, examined_count), c_seconds, c_peak_mb = measured_call(run_charm)
        closed_support_map = {
            tuple(sorted(itemset)): len(tids) for itemset, tids in closed_map.items()
        }
        rows.append({
            "algorithm": "CHARM-style",
            "minimum_support": min_support,
            "minimum_support_count": min_count,
            "maximum_itemset_length": MAX_ITEMSET_LENGTH,
            "output_itemsets": len(closed_support_map),
            "runtime_seconds": round(c_seconds, 4),
            "peak_traced_memory_mb": round(c_peak_mb, 3),
            "output_type": "Closed frequent itemsets",
        })
        validation[-1]["charm_closed_itemsets"] = len(closed_support_map)
        validation[-1]["charm_frequent_itemsets_examined"] = examined_count

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(RESULTS_DIR / "frequent_mining_benchmark.csv", index=False, encoding="utf-8")

    # Runtime chart
    pivot_time = benchmark.pivot(index="minimum_support", columns="algorithm", values="runtime_seconds")
    ax = pivot_time.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Frequent-Pattern Mining Runtime by Minimum Support")
    ax.set_xlabel("Minimum support")
    ax.set_ylabel("Runtime (seconds; lower is faster)")
    ax.set_xticklabels([f"{float(x):.0%}" for x in pivot_time.index], rotation=0)
    ax.legend(title="Algorithm")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "frequent_mining_runtime.png", dpi=160)
    plt.close(ax.figure)

    # Output count chart; CHARM-style is explicitly closed itemsets, not all frequent itemsets.
    pivot_count = benchmark.pivot(index="minimum_support", columns="algorithm", values="output_itemsets")
    ax = pivot_count.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Pattern Output Counts (CHARM-Style Outputs Closed Itemsets)")
    ax.set_xlabel("Minimum support")
    ax.set_ylabel("Number of output itemsets")
    ax.set_xticklabels([f"{float(x):.0%}" for x in pivot_count.index], rotation=0)
    ax.legend(title="Algorithm")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "frequent_mining_itemset_counts.png", dpi=160)
    plt.close(ax.figure)

    summary = {
        "layer": "Layer 7 — Comparative Frequent-Pattern Mining Experiments",
        "input_file": str(INPUT_PATH),
        "transaction_count": len(transactions),
        "unique_product_count": len(set().union(*transactions)),
        "support_thresholds": SUPPORT_THRESHOLDS,
        "maximum_itemset_length": MAX_ITEMSET_LENGTH,
        "benchmark_rows": rows,
        "correctness_checks": validation,
        "all_apriori_fp_growth_checks_passed": all(
            item["apriori_fp_growth_identical_itemsets_and_counts"] for item in validation
        ),
        "measurement_notes": [
            "All algorithms use the same transaction input, support thresholds, and maximum itemset length.",
            "Peak memory is measured with Python tracemalloc and represents traced Python allocations, not total process resident memory. Runtime is measured without tracing overhead; peak traced memory is measured in a separate repeat run and represents Python allocations, not total process resident memory.",
            "Runtime depends on the machine, Python version, background load, and implementation details.",
            "CHARM-style output contains closed frequent itemsets, whereas Apriori and FP-Growth output all frequent itemsets; their output counts are not directly equivalent.",
            "The CHARM implementation is a transparent vertical-TID-set educational baseline, not an optimized canonical CHARM implementation.",
            "Results are single-machine baselines; distributed Spark/Hadoop experiments belong to Version 2."
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": [
            "experiments/results/frequent_mining_benchmark.csv",
            "reports/figures/frequent_mining_runtime.png",
            "reports/figures/frequent_mining_itemset_counts.png"
        ]
    }
    (RESULTS_DIR / "frequent_mining_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("LAYER 7 — COMPARATIVE MINING BENCHMARK: COMPLETED")
    print(f"Transactions: {len(transactions):,}")
    print(f"Unique products: {summary['unique_product_count']:,}")
    print(benchmark.to_string(index=False))
    print("\nApriori vs FP-Growth correctness checks:")
    for check in validation:
        print(
            f" support={check['minimum_support']:.0%}: "
            f"{'PASS' if check['apriori_fp_growth_identical_itemsets_and_counts'] else 'FAIL'}"
        )
    print(f"Benchmark CSV: {RESULTS_DIR / 'frequent_mining_benchmark.csv'}")
    print(f"Summary JSON: {RESULTS_DIR / 'frequent_mining_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
