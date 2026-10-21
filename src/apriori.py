"""
Layer 4A — Apriori Frequent Itemset Mining
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Input:
    data/transactions/transactions.csv
    One row per invoice; `items` contains a JSON list of StockCode values.

Outputs:
    experiments/results/apriori_itemsets.csv
    experiments/results/apriori_run_summary.json

Run from project root:
    python src/apriori.py
    python src/apriori.py --min-support 0.01 --max-len 3
    python src/apriori.py --min-support 0.05 --max-len 4

Support is the fraction of transactions containing an itemset.
`--max-len` limits the largest itemset size for a controlled experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "transactions" / "transactions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"


def load_transactions(path: Path) -> list[frozenset[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Transaction input not found: {path}\n"
            "Run `python src/preprocessing.py` first."
        )

    frame = pd.read_csv(path, dtype={"InvoiceNo": "string"})
    if "items" not in frame.columns:
        raise ValueError("Expected an `items` column containing JSON-encoded product lists.")

    transactions: list[frozenset[str]] = []
    for row_number, raw_items in enumerate(frame["items"], start=2):
        try:
            parsed = json.loads(raw_items)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON item list at CSV row {row_number}.") from exc

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a list of item codes at CSV row {row_number}.")

        # A transaction is a set: repeated copies of one StockCode count once.
        items = frozenset(str(item).strip() for item in parsed if str(item).strip())
        if items:
            transactions.append(items)

    if not transactions:
        raise ValueError("No non-empty transactions were loaded.")
    return transactions


def generate_candidates(previous_frequents: set[tuple[str, ...]], k: int) -> set[tuple[str, ...]]:
    """Join frequent (k-1)-itemsets and prune candidates with infrequent subsets."""
    previous = sorted(previous_frequents)
    previous_lookup = previous_frequents
    candidates: set[tuple[str, ...]] = set()

    for i in range(len(previous)):
        for j in range(i + 1, len(previous)):
            left, right = previous[i], previous[j]
            if left[: k - 2] != right[: k - 2]:
                if k > 2:
                    # Sorted join order means later prefixes cannot match this left prefix.
                    if right[: k - 2] > left[: k - 2]:
                        break
                continue

            candidate = tuple(sorted(set(left) | set(right)))
            if len(candidate) != k:
                continue

            # Apriori property: every (k-1)-subset must itself be frequent.
            if all(tuple(subset) in previous_lookup for subset in combinations(candidate, k - 1)):
                candidates.add(candidate)

    return candidates


def count_candidate_supports(
    transactions: list[frozenset[str]],
    candidates: set[tuple[str, ...]],
    k: int,
    frequent_singletons: set[str],
) -> Counter[tuple[str, ...]]:
    """
    Count candidates by enumerating combinations present in each transaction.
    This avoids scanning every candidate against every transaction.
    """
    counts: Counter[tuple[str, ...]] = Counter()
    for transaction in transactions:
        eligible = sorted(transaction.intersection(frequent_singletons))
        if len(eligible) < k:
            continue
        for itemset in combinations(eligible, k):
            if itemset in candidates:
                counts[itemset] += 1
    return counts


def apriori(
    transactions: list[frozenset[str]],
    min_support: float,
    max_len: int | None,
) -> tuple[list[dict], dict]:
    transaction_count = len(transactions)
    min_count = max(1, int((min_support * transaction_count) + 0.999999999))
    start = time.perf_counter()

    singleton_counts: Counter[str] = Counter()
    for transaction in transactions:
        singleton_counts.update(transaction)

    frequent_singletons = {
        item for item, count in singleton_counts.items() if count >= min_count
    }
    previous_frequents: set[tuple[str, ...]] = {
        (item,) for item in frequent_singletons
    }

    rows: list[dict] = []
    for itemset in sorted(previous_frequents):
        count = singleton_counts[itemset[0]]
        rows.append({
            "itemset": json.dumps(list(itemset), ensure_ascii=False),
            "length": 1,
            "support_count": int(count),
            "support": count / transaction_count,
        })

    level_summaries = [{
        "itemset_length": 1,
        "candidate_count": len(singleton_counts),
        "frequent_itemset_count": len(previous_frequents),
    }]

    k = 2
    while previous_frequents and (max_len is None or k <= max_len):
        candidates = generate_candidates(previous_frequents, k)
        if not candidates:
            level_summaries.append({
                "itemset_length": k,
                "candidate_count": 0,
                "frequent_itemset_count": 0,
            })
            break

        counts = count_candidate_supports(
            transactions, candidates, k, frequent_singletons
        )
        current_frequents = {
            candidate for candidate, count in counts.items() if count >= min_count
        }

        level_summaries.append({
            "itemset_length": k,
            "candidate_count": len(candidates),
            "frequent_itemset_count": len(current_frequents),
        })

        for itemset in sorted(current_frequents):
            count = counts[itemset]
            rows.append({
                "itemset": json.dumps(list(itemset), ensure_ascii=False),
                "length": k,
                "support_count": int(count),
                "support": count / transaction_count,
            })

        previous_frequents = current_frequents
        k += 1

    elapsed = time.perf_counter() - start
    rows.sort(key=lambda row: (row["length"], -row["support_count"], row["itemset"]))
    metadata = {
        "algorithm": "Apriori",
        "input_file": str(DEFAULT_INPUT),
        "transaction_count": transaction_count,
        "unique_items_in_input": len(set().union(*transactions)),
        "minimum_support_fraction": min_support,
        "minimum_support_percent": round(min_support * 100, 4),
        "minimum_support_count": min_count,
        "maximum_itemset_length": max_len,
        "total_frequent_itemsets": len(rows),
        "frequent_itemsets_by_length": {
            str(level["itemset_length"]): level["frequent_itemset_count"]
            for level in level_summaries
        },
        "levels": level_summaries,
        "runtime_seconds": round(elapsed, 4),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Each invoice is treated as a set of distinct StockCode values.",
            "Support is the fraction of input transactions containing the itemset.",
            "The maximum itemset length may truncate mining when configured.",
            "This implementation is a single-machine baseline, not a distributed Spark implementation."
        ],
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Apriori frequent itemset mining.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-support", type=float, default=0.02,
                        help="Minimum support as a fraction, e.g. 0.02 means 2%% (default: 0.02).")
    parser.add_argument("--max-len", type=int, default=3,
                        help="Maximum itemset length; use 0 for no explicit limit (default: 3).")
    args = parser.parse_args()

    if not 0 < args.min_support <= 1:
        parser.error("--min-support must be greater than 0 and at most 1.")
    if args.max_len < 0:
        parser.error("--max-len must be zero (unlimited) or a positive integer.")
    max_len = None if args.max_len == 0 else args.max_len
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(args.input.expanduser().resolve())
    rows, metadata = apriori(transactions, args.min_support, max_len)
    metadata["input_file"] = str(args.input.expanduser().resolve())

    output_csv = output_dir / "apriori_itemsets.csv"
    output_json = output_dir / "apriori_run_summary.json"
    pd.DataFrame(rows, columns=["itemset", "length", "support_count", "support"]).to_csv(
        output_csv, index=False, encoding="utf-8"
    )
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("LAYER 4A — APRIORI: PASSED")
    print(f"Transactions: {metadata['transaction_count']:,}")
    print(f"Unique input products: {metadata['unique_items_in_input']:,}")
    print(f"Minimum support: {args.min_support:.2%} ({metadata['minimum_support_count']:,} transactions)")
    print(f"Maximum itemset length: {'unlimited' if max_len is None else max_len}")
    print(f"Frequent itemsets found: {metadata['total_frequent_itemsets']:,}")
    for level in metadata["levels"]:
        print(
            f"  Length {level['itemset_length']}: "
            f"{level['frequent_itemset_count']:,} frequent / "
            f"{level['candidate_count']:,} candidates"
        )
    print(f"Runtime: {metadata['runtime_seconds']:.2f} seconds")
    print(f"Itemsets: {output_csv}")
    print(f"Run summary: {output_json}")


if __name__ == "__main__":
    main()
