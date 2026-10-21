"""
Layer 1 — Data Acquisition
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Responsibilities:
1. Validate that the raw workbook exists and is readable.
2. Verify the expected schema.
3. Record file integrity metadata (size and SHA-256).
4. Generate a dataset profile and non-destructive quality-check reports.

This script does NOT clean or modify the raw dataset. Cleaning belongs to Layer 2.
Run from the project root:
    python src/acquire_data.py
Or provide an alternate input:
    python src/acquire_data.py --input "data/raw/Online Retail.xlsx"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
SOURCE_URL = "https://uci-ics-mlr-prod.aws.uci.edu/dataset/352/online%2Bretail"

EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 checksum without loading the entire file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert common Pandas/Numpy values into JSON-serializable Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def build_column_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact profile for every column without changing the data."""
    rows = []
    total_rows = len(data)

    for column in data.columns:
        series = data[column]
        non_null = series.dropna()
        samples = [str(value) for value in non_null.drop_duplicates().head(5).tolist()]
        rows.append(
            {
                "column": column,
                "dtype_as_loaded": str(series.dtype),
                "row_count": total_rows,
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 4),
                "unique_non_null": int(series.nunique(dropna=True)),
                "sample_values": " | ".join(samples),
            }
        )
    return pd.DataFrame(rows)


def build_quality_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Record diagnostic counts. These are observations, not cleaning actions."""
    invoice_text = data["InvoiceNo"].astype("string").str.strip()
    checks = [
        ("row_count", len(data), "Total source records."),
        ("column_count", len(data.columns), "Total source columns."),
        ("exact_duplicate_rows", int(data.duplicated().sum()),
         "Rows identical across all columns; do not remove at acquisition stage."),
        ("missing_customer_id_rows", int(data["CustomerID"].isna().sum()),
         "May still be usable for invoice-level pattern mining."),
        ("missing_description_rows", int(data["Description"].isna().sum()),
         "Review during preprocessing; StockCode may still be present."),
        ("cancelled_invoice_lines", int(invoice_text.str.startswith("C", na=False).sum()),
         "Invoice numbers beginning with C; investigate in preprocessing."),
        ("negative_quantity_lines", int((data["Quantity"] < 0).sum()),
         "May represent returns or adjustments; do not assume all are errors."),
        ("zero_quantity_lines", int((data["Quantity"] == 0).sum()),
         "Zero-quantity records, if any."),
        ("negative_unit_price_lines", int((data["UnitPrice"] < 0).sum()),
         "Requires review in preprocessing."),
        ("zero_unit_price_lines", int((data["UnitPrice"] == 0).sum()),
         "May be samples, promotional items or other special records."),
        ("unique_invoices", int(data["InvoiceNo"].nunique(dropna=True)),
         "Distinct invoice identifiers, including any cancellation invoices."),
        ("unique_stock_codes", int(data["StockCode"].nunique(dropna=True)),
         "Distinct product codes."),
        ("unique_customers_non_null", int(data["CustomerID"].nunique(dropna=True)),
         "Distinct known customer identifiers."),
        ("unique_countries", int(data["Country"].nunique(dropna=True)),
         "Distinct country values."),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and profile the raw Online Retail dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the raw Excel workbook (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for profile reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}\n"
            "Place Online Retail.xlsx in data/raw/ or pass --input with the correct path."
        )
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the workbook but never write to or alter the source file.
    workbook = pd.ExcelFile(input_path)
    if not workbook.sheet_names:
        raise ValueError("The Excel workbook contains no worksheets.")
    selected_sheet = "Online Retail" if "Online Retail" in workbook.sheet_names else workbook.sheet_names[0]
    data = pd.read_excel(workbook, sheet_name=selected_sheet)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    unexpected_columns = [column for column in data.columns if column not in EXPECTED_COLUMNS]
    if missing_columns:
        raise ValueError(
            "Dataset schema validation failed. "
            f"Missing expected columns: {missing_columns}. "
            f"Columns found: {list(data.columns)}"
        )

    column_profile = build_column_profile(data)
    quality_checks = build_quality_checks(data)

    profile = {
        "project": "Comparative Frequent Pattern Mining and Customer Behaviour Analytics",
        "layer": "Layer 1 — Data Acquisition",
        "source_name": "UCI Online Retail",
        "source_url": SOURCE_URL,
        "input_file": input_path.name,
        "input_path_at_execution": str(input_path),
        "file_size_bytes": input_path.stat().st_size,
        "file_size_mib": round(input_path.stat().st_size / (1024 ** 2), 3),
        "sha256": sha256_file(input_path),
        "acquisition_record_created_utc": datetime.now(timezone.utc).isoformat(),
        "worksheet_names": workbook.sheet_names,
        "worksheet_profiled": selected_sheet,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "columns": list(data.columns),
        "missing_expected_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "date_min": json_safe(data["InvoiceDate"].min()) if "InvoiceDate" in data else None,
        "date_max": json_safe(data["InvoiceDate"].max()) if "InvoiceDate" in data else None,
        "unique_invoices": int(data["InvoiceNo"].nunique(dropna=True)) if "InvoiceNo" in data else None,
        "unique_stock_codes": int(data["StockCode"].nunique(dropna=True)) if "StockCode" in data else None,
        "unique_customers_non_null": int(data["CustomerID"].nunique(dropna=True)) if "CustomerID" in data else None,
        "unique_countries": int(data["Country"].nunique(dropna=True)) if "Country" in data else None,
        "schema_validation": "passed",
        "raw_data_modified": False,
        "notes": [
            "This is a profiling and validation stage only; no rows were removed or transformed.",
            "Missing customer identifiers do not automatically invalidate invoice-level transactions.",
            "Cancellation and negative-quantity records require an explicit preprocessing policy in Layer 2.",
            "The SHA-256 checksum identifies the exact workbook used for this run."
        ],
    }

    profile_path = output_dir / "layer1_dataset_profile.json"
    columns_path = output_dir / "layer1_column_profile.csv"
    checks_path = output_dir / "layer1_quality_checks.csv"

    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    column_profile.to_csv(columns_path, index=False, encoding="utf-8")
    quality_checks.to_csv(checks_path, index=False, encoding="utf-8")

    print("\nLAYER 1 — DATA ACQUISITION: PASSED")
    print("-" * 54)
    print(f"Input file       : {input_path}")
    print(f"Worksheet        : {selected_sheet}")
    print(f"Dimensions       : {len(data):,} rows x {len(data.columns)} columns")
    print(f"Date range       : {data['InvoiceDate'].min()} to {data['InvoiceDate'].max()}")
    print(f"Exact duplicates : {int(data.duplicated().sum()):,}")
    print(f"Missing CustomerID: {int(data['CustomerID'].isna().sum()):,}")
    print(f"SHA-256          : {profile['sha256']}")
    print("\nReports written:")
    print(f"- {profile_path}")
    print(f"- {columns_path}")
    print(f"- {checks_path}")
    print("\nNo cleaning or modification of the raw dataset was performed.")


if __name__ == "__main__":
    main()
