"""
Layer 4B — FP-Growth Frequent Itemset Mining
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Uses an FP-tree and conditional pattern bases; it does not generate candidate
itemsets in the Apriori manner.

Default settings match Layer 4A for fair comparison:
    minimum support = 2%
    maximum itemset length = 3

Run from project root:
    python src/fp_growth.py
    python src/fp_growth.py --min-support 0.01 --max-len 3
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "transactions" / "transactions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"


class FPNode:
    """A node in an FP-tree."""

    def __init__(self, item: str | None, parent: "FPNode | None"):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[str, FPNode] = {}
        self.link: FPNode | None = None


class FPTree:
    """FP-tree plus header table containing item support and node links."""

    def __init__(self, weighted_transactions: list[tuple[list[str], int]], min_count: int):
        self.root = FPNode(None, None)
        self.header_support: Counter[str] = Counter()
        self.header_heads: dict[str, FPNode] = {}
        self.header_tails: dict[str, FPNode] = {}

        for items, weight in weighted_transactions:
            for item in set(items):
                self.header_support[item] += weight

        self.header_support = Counter({
            item: count for item, count in self.header_support.items()
            if count >= min_count
        })
        self.order = {
            item: rank
            for rank, (item, _) in enumerate(
                sorted(self.header_support.items(), key=lambda pair: (-pair[1], pair[0]))
            )
        }

        for items, weight in weighted_transactions:
            filtered = [item for item in set(items) if item in self.header_support]
            filtered.sort(key=lambda item: self.order[item])
            if not filtered:
                continue

            current = self.root
            for item in filtered:
                if item not in current.children:
                    child = FPNode(item, current)
                    current.children[item] = child
                    if item not in self.header_heads:
                        self.header_heads[item] = child
                        self.header_tails[item] = child
                    else:
                        self.header_tails[item].link = child
                        self.header_tails[item] = child
                current = current.children[item]
                current.count += weight

    def conditional_pattern_base(self, item: str) -> list[tuple[list[str], int]]:
        """Return prefix paths and their counts for a given item."""
        paths: list[tuple[list[str], int]] = []
        node = self.header_heads.get(item)
        while node is not None:
            path: list[str] = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                paths.append((path, node.count))
            node = node.link
        return paths


def load_transactions(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Transaction input not found: {path}\n"
            "Run `python src/preprocessing.py` first."
        )
    frame = pd.read_csv(path, dtype={"InvoiceNo": "string"})
    if "items" not in frame.columns:
        raise ValueError("Expected an `items` column containing JSON-encoded product lists.")

    transactions: list[list[str]] = []
    for row_number, raw_items in enumerate(frame["items"], start=2):
        try:
            parsed = json.loads(raw_items)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON item list at CSV row {row_number}.") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a list of product codes at CSV row {row_number}.")
        items = sorted(set(str(item).strip() for item in parsed if str(item).strip()))
        if items:
            transactions.append(items)
    if not transactions:
        raise ValueError("No non-empty transactions were loaded.")
    return transactions


def mine_fp_tree(
    tree: FPTree,
    suffix: tuple[str, ...],
    min_count: int,
    max_len: int | None,
    found: dict[tuple[str, ...], int],
) -> None:
    # Mine least-frequent items first, as required by the FP-growth procedure.
    for item, support_count in sorted(
        tree.header_support.items(), key=lambda pair: (pair[1], pair[0])
    ):
        pattern = tuple(sorted(suffix + (item,)))
        if max_len is not None and len(pattern) > max_len:
            continue
        found[pattern] = int(support_count)

        if max_len is not None and len(pattern) >= max_len:
            continue

        conditional_base = tree.conditional_pattern_base(item)
        if not conditional_base:
            continue

        conditional_tree = FPTree(conditional_base, min_count)
        if conditional_tree.header_support:
            mine_fp_tree(
                conditional_tree, pattern, min_count, max_len, found
            )


def fp_growth(
    transactions: list[list[str]],
    min_support: float,
    max_len: int | None,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    transaction_count = len(transactions)
    min_count = max(1, int(__import__("math").ceil(min_support * transaction_count)))

    weighted_transactions = [(items, 1) for items in transactions]
    tree = FPTree(weighted_transactions, min_count)
    found: dict[tuple[str, ...], int] = {}
    mine_fp_tree(tree, tuple(), min_count, max_len, found)

    rows = [
        {
            "itemset": json.dumps(list(itemset), ensure_ascii=False),
            "length": len(itemset),
            "support_count": count,
            "support": count / transaction_count,
        }
        for itemset, count in found.items()
    ]
    rows.sort(key=lambda row: (row["length"], -row["support_count"], row["itemset"]))
    elapsed = time.perf_counter() - start

    metadata = {
        "algorithm": "FP-Growth",
        "transaction_count": transaction_count,
        "unique_items_in_input": len(set().union(*(set(t) for t in transactions))),
        "minimum_support_fraction": min_support,
        "minimum_support_percent": round(min_support * 100, 4),
        "minimum_support_count": min_count,
        "maximum_itemset_length": max_len,
        "total_frequent_itemsets": len(rows),
        "frequent_itemsets_by_length": {
            str(length): sum(1 for row in rows if row["length"] == length)
            for length in sorted({row["length"] for row in rows})
        },
        "runtime_seconds": round(elapsed, 4),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Uses an FP-tree and conditional pattern bases rather than Apriori candidate generation.",
            "Each invoice is treated as a set of distinct StockCode values.",
            "Support is the fraction of input transactions containing the itemset.",
            "The maximum itemset length may truncate mining when configured.",
            "This implementation is a single-machine baseline, not a distributed Spark implementation."
        ],
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FP-Growth frequent itemset mining.")
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

    transactions = load_transactions(input_path)
    rows, metadata = fp_growth(transactions, args.min_support, max_len)
    metadata["input_file"] = str(input_path)

    output_csv = output_dir / "fp_growth_itemsets.csv"
    output_json = output_dir / "fp_growth_run_summary.json"
    pd.DataFrame(rows, columns=["itemset", "length", "support_count", "support"]).to_csv(
        output_csv, index=False, encoding="utf-8"
    )
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("LAYER 4B — FP-GROWTH: PASSED")
    print(f"Transactions: {metadata['transaction_count']:,}")
    print(f"Unique input products: {metadata['unique_items_in_input']:,}")
    print(f"Minimum support: {args.min_support:.2%} ({metadata['minimum_support_count']:,} transactions)")
    print(f"Maximum itemset length: {'unlimited' if max_len is None else max_len}")
    print(f"Frequent itemsets found: {metadata['total_frequent_itemsets']:,}")
    print(f"Runtime: {metadata['runtime_seconds']:.2f} seconds")
    print(f"Itemsets: {output_csv}")
    print(f"Run summary: {output_json}")


if __name__ == "__main__":
    main()
