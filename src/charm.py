"""
Layer 4C — CHARM-Style Closed Frequent Itemset Mining
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

This educational implementation uses vertical transaction-ID (TID) sets,
depth-first intersections, and equal-support closure filtering. It is a
transparent CHARM-style baseline, not an optimized canonical CHARM implementation.

Default settings match Apriori and FP-Growth:
    minimum support = 2%
    maximum itemset length = 3

Run from project root:
    python src/charm.py
    python src/charm.py --min-support 0.01 --max-len 3
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "transactions" / "transactions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"


def load_vertical_tidsets(path: Path) -> tuple[dict[str, frozenset[int]], int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Transaction input not found: {path}\n"
            "Run `python src/preprocessing.py` first."
        )
    frame = pd.read_csv(path, dtype={"InvoiceNo": "string"})
    if "items" not in frame.columns:
        raise ValueError("Expected an `items` column containing JSON-encoded product lists.")

    vertical: dict[str, set[int]] = {}
    transaction_count = 0
    for raw_items in frame["items"]:
        try:
            items = json.loads(raw_items)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid JSON item list in transactions.csv.") from exc
        if not isinstance(items, list):
            raise ValueError("Each transaction's `items` value must be a JSON list.")
        normalized = set(str(x).strip() for x in items if str(x).strip())
        if not normalized:
            continue
        tid = transaction_count
        transaction_count += 1
        for item in normalized:
            vertical.setdefault(item, set()).add(tid)

    if transaction_count == 0:
        raise ValueError("No non-empty transactions were loaded.")
    return {item: frozenset(tids) for item, tids in vertical.items()}, transaction_count


def mine_frequent_itemsets(
    vertical: dict[str, frozenset[int]],
    min_count: int,
    max_len: int | None,
) -> dict[tuple[str, ...], frozenset[int]]:
    """Depth-first vertical mining using TID-set intersections."""
    frequent_singletons = sorted(
        item for item, tids in vertical.items() if len(tids) >= min_count
    )
    found: dict[tuple[str, ...], frozenset[int]] = {
        (item,): vertical[item] for item in frequent_singletons
    }

    def extend(prefix: tuple[str, ...], prefix_tids: frozenset[int], extensions: list[str]) -> None:
        if max_len is not None and len(prefix) >= max_len:
            return
        for index, item in enumerate(extensions):
            tids = prefix_tids.intersection(vertical[item])
            if len(tids) < min_count:
                continue
            candidate = tuple(sorted(prefix + (item,)))
            found[candidate] = frozenset(tids)
            if max_len is None or len(candidate) < max_len:
                extend(candidate, frozenset(tids), extensions[index + 1:])

    for index, item in enumerate(frequent_singletons):
        if max_len == 1:
            continue
        extend((item,), vertical[item], frequent_singletons[index + 1:])
    return found


def retain_closed_itemsets(
    frequent: dict[tuple[str, ...], frozenset[int]],
) -> dict[tuple[str, ...], frozenset[int]]:
    """
    An itemset is closed if no proper frequent superset has the same TID set.
    This closure pass is restricted to the itemsets mined under --max-len.
    """
    by_tidset: dict[frozenset[int], list[tuple[str, ...]]] = {}
    for itemset, tids in frequent.items():
        by_tidset.setdefault(tids, []).append(itemset)

    closed = {}
    for itemset, tids in frequent.items():
        same_support_sets = by_tidset[tids]
        has_same_support_proper_superset = any(
            len(other) > len(itemset) and set(itemset).issubset(other)
            for other in same_support_sets
        )
        if not has_same_support_proper_superset:
            closed[itemset] = tids
    return closed


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine closed frequent itemsets using vertical TID sets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--max-len", type=int, default=3,
                        help="Maximum itemset length; use 0 for no explicit limit (default: 3).")
    args = parser.parse_args()
    if not 0 < args.min_support <= 1:
        parser.error("--min-support must be greater than 0 and at most 1.")
    if args.max_len < 0:
        parser.error("--max-len must be zero (unlimited) or a positive integer.")
    max_len = None if args.max_len == 0 else args.max_len
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    vertical, transaction_count = load_vertical_tidsets(input_path)
    min_count = max(1, math.ceil(args.min_support * transaction_count))
    frequent = mine_frequent_itemsets(vertical, min_count, max_len)
    closed = retain_closed_itemsets(frequent)
    elapsed = time.perf_counter() - start

    rows = [{
        "itemset": json.dumps(list(itemset), ensure_ascii=False),
        "length": len(itemset),
        "support_count": len(tids),
        "support": len(tids) / transaction_count,
    } for itemset, tids in closed.items()]
    rows.sort(key=lambda row: (row["length"], -row["support_count"], row["itemset"]))

    summary = {
        "algorithm": "CHARM-style closed frequent itemset mining",
        "implementation_note": "Vertical TID-set depth-first mining plus equal-support closure filtering; educational baseline, not an optimized canonical CHARM implementation.",
        "input_file": str(input_path),
        "transaction_count": transaction_count,
        "unique_items_in_input": len(vertical),
        "minimum_support_fraction": args.min_support,
        "minimum_support_percent": round(args.min_support * 100, 4),
        "minimum_support_count": min_count,
        "maximum_itemset_length": max_len,
        "frequent_itemsets_examined_before_closure_filter": len(frequent),
        "closed_itemsets_output": len(closed),
        "closed_itemsets_by_length": {
            str(length): sum(1 for row in rows if row["length"] == length)
            for length in sorted({row["length"] for row in rows})
        },
        "runtime_seconds": round(elapsed, 4),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Support is the number of transactions containing the itemset divided by all loaded non-empty transactions.",
            "Closedness is assessed only among itemsets mined up to the configured maximum length.",
            "For an exhaustive closed-itemset result, run with --max-len 0; this may be computationally expensive.",
            "This is a single-machine implementation, not a distributed Spark implementation."
        ]
    }

    out_csv = output_dir / "charm_closed_itemsets.csv"
    out_json = output_dir / "charm_run_summary.json"
    pd.DataFrame(rows, columns=["itemset", "length", "support_count", "support"]).to_csv(
        out_csv, index=False, encoding="utf-8"
    )
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("LAYER 4C — CLOSED ITEMSET MINING: PASSED")
    print(f"Transactions: {transaction_count:,}")
    print(f"Unique products: {len(vertical):,}")
    print(f"Minimum support: {args.min_support:.2%} ({min_count:,} transactions)")
    print(f"Frequent itemsets examined: {len(frequent):,}")
    print(f"Closed itemsets output: {len(closed):,}")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Itemsets: {out_csv}")
    print(f"Run summary: {out_json}")


if __name__ == "__main__":
    import math
    main()
