"""
Layer 2 — Data Preprocessing and Transaction Construction
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Policy:
- Preserve the raw workbook; never write to it.
- Remove exact duplicate source rows.
- Exclude cancellation invoices (InvoiceNo begins with C) from purchase analysis.
- Exclude non-positive quantities from purchase baskets (negative quantities are treated as returns/adjustments).
- Keep rows with missing CustomerID for invoice-level pattern mining.
- For customer monetary analytics, require a known CustomerID and non-negative UnitPrice.
- Keep zero-price purchases; they can be legitimate free/promotional items.
- Basket items are distinct StockCode values per invoice, not quantities.

Run from the project root:
    python src/preprocessing.py
Optional:
    python src/preprocessing.py --input "data/raw/Online Retail.xlsx"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Online Retail data for pattern mining and customer analytics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to Online Retail.xlsx")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Raw workbook not found: {input_path}")

    raw = pd.read_excel(input_path, sheet_name="Online Retail")
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"Unexpected schema; missing columns: {missing_columns}")

    initial_rows = len(raw)
    exact_duplicates = int(raw.duplicated().sum())

    # Normalize identifier representations and types; no imputation is applied.
    data = raw.copy()
    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["StockCode"] = data["StockCode"].astype("string").str.strip()
    data["Description"] = data["Description"].astype("string").str.strip()
    data["Country"] = data["Country"].astype("string").str.strip()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["CustomerID"] = pd.to_numeric(data["CustomerID"], errors="coerce").astype("Int64")

    # Remove exact duplicates before applying the documented purchase rules.
    data = data.drop_duplicates().copy()
    rows_after_dedup = len(data)

    is_cancellation = data["InvoiceNo"].str.upper().str.startswith("C", na=False)
    cancellation_rows = int(is_cancellation.sum())
    purchases = data.loc[~is_cancellation].copy()
    rows_after_cancellations = len(purchases)

    invalid_required = purchases["InvoiceNo"].isna() | purchases["StockCode"].isna()
    invalid_required_rows = int(invalid_required.sum())
    purchases = purchases.loc[~invalid_required].copy()

    nonpositive_quantity = purchases["Quantity"].isna() | (purchases["Quantity"] <= 0)
    nonpositive_quantity_rows = int(nonpositive_quantity.sum())
    purchases = purchases.loc[~nonpositive_quantity].copy()

    # Invoice-level basket data can include unknown CustomerID and zero-price items.
    basket_lines = purchases[
        ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "CustomerID", "Country"]
    ].copy()

    # Distinct product presence per invoice; quantity is not duplicated into item occurrences.
    transaction_items = basket_lines[["InvoiceNo", "StockCode"]].drop_duplicates()
    transaction_items = transaction_items.sort_values(["InvoiceNo", "StockCode"], kind="stable")
    transactions = (
        transaction_items.groupby("InvoiceNo", sort=True)["StockCode"]
        .agg(list)
        .reset_index(name="items")
    )
    transactions["item_count"] = transactions["items"].map(len)
    transactions["items_json"] = transactions["items"].map(lambda items: json.dumps(items, ensure_ascii=False))

    # Customer analytics require a known customer ID and a usable non-negative unit price.
    known_customer = purchases["CustomerID"].notna()
    negative_price = purchases["UnitPrice"].isna() | (purchases["UnitPrice"] < 0)
    customer_lines = purchases.loc[known_customer & ~negative_price].copy()
    customer_lines["LineValue"] = customer_lines["Quantity"] * customer_lines["UnitPrice"]

    # Save line-level datasets, transaction items and one row per invoice basket.
    processed_dir = PROJECT_ROOT / "data" / "processed"
    transactions_dir = PROJECT_ROOT / "data" / "transactions"
    reports_dir = PROJECT_ROOT / "reports" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    transactions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    basket_lines.to_csv(processed_dir / "purchase_lines_for_mining.csv", index=False, encoding="utf-8")
    customer_lines.to_csv(processed_dir / "customer_purchase_lines.csv", index=False, encoding="utf-8")
    transaction_items.to_csv(transactions_dir / "transaction_items.csv", index=False, encoding="utf-8")
    transactions[["InvoiceNo", "items_json", "item_count"]].rename(
        columns={"items_json": "items"}
    ).to_csv(transactions_dir / "transactions.csv", index=False, encoding="utf-8")

    summary = {
        "project": "Comparative Frequent Pattern Mining and Customer Behaviour Analytics",
        "layer": "Layer 2 — Data Preprocessing and Transaction Construction",
        "input_file": str(input_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "exact_duplicates": "Removed exact duplicate source rows.",
            "cancellations": "Excluded invoice numbers beginning with C.",
            "quantity": "Excluded missing or non-positive quantities from purchase analysis.",
            "missing_customer_id": "Retained for invoice-level pattern mining; excluded from customer analytics.",
            "unit_price": "Zero prices retained; negative or missing prices excluded from customer monetary analytics.",
            "basket_item": "Each distinct StockCode occurs at most once per InvoiceNo."
        },
        "counts": {
            "source_rows": int(initial_rows),
            "exact_duplicate_rows_removed": exact_duplicates,
            "rows_after_deduplication": int(rows_after_dedup),
            "cancellation_rows_excluded": cancellation_rows,
            "rows_after_cancellation_filter": int(rows_after_cancellations),
            "rows_with_missing_invoice_or_stock_code_excluded": invalid_required_rows,
            "rows_with_missing_or_nonpositive_quantity_excluded": nonpositive_quantity_rows,
            "purchase_line_rows_for_mining": int(len(basket_lines)),
            "unique_mining_invoices": int(basket_lines["InvoiceNo"].nunique()),
            "unique_mining_stock_codes": int(basket_lines["StockCode"].nunique()),
            "unique_invoice_stockcode_pairs": int(len(transaction_items)),
            "customer_purchase_line_rows": int(len(customer_lines)),
            "unique_customers_for_customer_analytics": int(customer_lines["CustomerID"].nunique()),
            "unique_customer_invoices": int(customer_lines["InvoiceNo"].nunique()),
            "customer_rows_excluded_for_missing_id": int((~known_customer & ~negative_price).sum()),
            "customer_rows_excluded_for_missing_or_negative_price": int((known_customer & negative_price).sum()),
            "transactions_with_one_or_more_items": int(len(transactions)),
            "mean_items_per_transaction": round(float(transactions["item_count"].mean()), 4),
            "median_items_per_transaction": float(transactions["item_count"].median()),
            "max_items_per_transaction": int(transactions["item_count"].max())
        },
        "outputs": [
            "data/processed/purchase_lines_for_mining.csv",
            "data/processed/customer_purchase_lines.csv",
            "data/transactions/transaction_items.csv",
            "data/transactions/transactions.csv"
        ],
        "note": "This script creates analytical datasets and does not modify the raw workbook."
    }
    summary_path = reports_dir / "layer2_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary["counts"].items()]
    )
    counts.to_csv(reports_dir / "layer2_preprocessing_counts.csv", index=False, encoding="utf-8")

    print("LAYER 2 — PREPROCESSING: PASSED")
    print(f"Source rows: {initial_rows:,}")
    print(f"Exact duplicates removed: {exact_duplicates:,}")
    print(f"Purchase lines for mining: {len(basket_lines):,}")
    print(f"Distinct invoices for mining: {basket_lines['InvoiceNo'].nunique():,}")
    print(f"Distinct invoice-product pairs: {len(transaction_items):,}")
    print(f"Customer purchase lines: {len(customer_lines):,}")
    print(f"Known customers: {customer_lines['CustomerID'].nunique():,}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
