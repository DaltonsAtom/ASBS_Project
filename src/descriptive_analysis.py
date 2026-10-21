"""
Layer 3 — Descriptive Statistics and Exploratory Data Analysis (EDA)
Project: Comparative Frequent Pattern Mining and Customer Behaviour Analytics

Reads Layer 2 outputs and creates descriptive tables and figures.
It does not change the raw or preprocessed data.

Run from the project root:
    python src/descriptive_analysis.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINING_PATH = PROJECT_ROOT / "data" / "processed" / "purchase_lines_for_mining.csv"
CUSTOMER_PATH = PROJECT_ROOT / "data" / "processed" / "customer_purchase_lines.csv"
ITEMS_PATH = PROJECT_ROOT / "data" / "transactions" / "transaction_items.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Layer 2 output not found: {path}\n"
            "Run `python src/preprocessing.py` before running this script."
        )


def save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path,
                   horizontal: bool = False, rotate: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax)
    else:
        series.plot(kind="bar", ax=ax)
        if rotate:
            ax.tick_params(axis="x", labelrotation=rotate)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    for path in [MINING_PATH, CUSTOMER_PATH, ITEMS_PATH]:
        require_file(path)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mining = pd.read_csv(
        MINING_PATH,
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string", "Country": "string"},
        parse_dates=["InvoiceDate"],
    )
    customer = pd.read_csv(
        CUSTOMER_PATH,
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string", "Country": "string"},
        parse_dates=["InvoiceDate"],
    )
    items = pd.read_csv(ITEMS_PATH, dtype={"InvoiceNo": "string", "StockCode": "string"})

    # Core descriptive statistics
    numeric_cols = [c for c in ["Quantity", "UnitPrice", "LineValue"] if c in customer.columns]
    numeric_summary = customer[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    numeric_summary.to_csv(TABLES_DIR / "layer3_numeric_summary.csv", encoding="utf-8")

    missing_profile = pd.DataFrame({
        "column": mining.columns,
        "missing_count": [int(mining[c].isna().sum()) for c in mining.columns],
        "missing_percent": [round(float(mining[c].isna().mean() * 100), 4) for c in mining.columns],
        "unique_non_null": [int(mining[c].nunique(dropna=True)) for c in mining.columns],
        "dtype": [str(mining[c].dtype) for c in mining.columns],
    })
    missing_profile.to_csv(TABLES_DIR / "layer3_mining_data_profile.csv", index=False, encoding="utf-8")

    # Transaction/basket size: number of distinct products per invoice
    basket_sizes = items.groupby("InvoiceNo", dropna=True)["StockCode"].nunique().rename("item_count")
    basket_sizes.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_frame().to_csv(
        TABLES_DIR / "layer3_basket_size_summary.csv", encoding="utf-8"
    )

    # Product frequency: count invoices containing each product, not raw line occurrences.
    product_invoice_counts = items.groupby("StockCode")["InvoiceNo"].nunique().sort_values(ascending=False)
    product_invoice_counts.head(30).rename("invoice_count").to_csv(
        TABLES_DIR / "layer3_top_products_by_invoice_count.csv", encoding="utf-8"
    )

    # Monthly transaction-line count and known-customer revenue.
    mining["InvoiceMonth"] = mining["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_lines = mining.groupby("InvoiceMonth").size().rename("purchase_line_count")
    monthly_lines.to_csv(TABLES_DIR / "layer3_monthly_purchase_lines.csv", encoding="utf-8")

    customer["InvoiceMonth"] = customer["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_revenue = customer.groupby("InvoiceMonth")["LineValue"].sum().rename("known_customer_revenue")
    monthly_revenue.to_csv(TABLES_DIR / "layer3_monthly_known_customer_revenue.csv", encoding="utf-8")

    country_summary = (
        customer.groupby("Country")
        .agg(
            purchase_lines=("InvoiceNo", "size"),
            unique_customers=("CustomerID", "nunique"),
            unique_invoices=("InvoiceNo", "nunique"),
            known_customer_revenue=("LineValue", "sum"),
        )
        .sort_values("known_customer_revenue", ascending=False)
    )
    country_summary.to_csv(TABLES_DIR / "layer3_country_summary.csv", encoding="utf-8")

    # Outlier diagnostics: IQR fences are descriptive flags, not automatic deletion criteria.
    outlier_rows = []
    for column in ["Quantity", "UnitPrice", "LineValue"]:
        values = pd.to_numeric(customer[column], errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rows.append({
            "variable": column,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_iqr_fence": float(lower),
            "upper_iqr_fence": float(upper),
            "below_lower_fence_count": int((values < lower).sum()),
            "above_upper_fence_count": int((values > upper).sum()),
            "interpretation": "Flag for investigation; not automatically an error or removed record."
        })
    pd.DataFrame(outlier_rows).to_csv(TABLES_DIR / "layer3_iqr_outlier_diagnostics.csv", index=False, encoding="utf-8")

    # Visualizations
    save_bar_chart(
        product_invoice_counts.head(15).sort_values(),
        "Top 15 Products by Number of Invoices",
        "Distinct product code", "Number of invoices",
        FIGURES_DIR / "layer3_top_products.png", horizontal=True
    )
    save_bar_chart(
        monthly_lines,
        "Monthly Purchase-Line Volume",
        "Invoice month", "Purchase-line records",
        FIGURES_DIR / "layer3_monthly_purchase_lines.png", rotate=45
    )
    save_bar_chart(
        monthly_revenue,
        "Monthly Revenue from Known-Customer Purchase Lines",
        "Invoice month", "Line value (dataset currency)",
        FIGURES_DIR / "layer3_monthly_known_customer_revenue.png", rotate=45
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(basket_sizes, bins=50)
    ax.set_title("Distribution of Distinct Products per Invoice")
    ax.set_xlabel("Distinct product count in invoice")
    ax.set_ylabel("Number of invoices")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "layer3_basket_size_distribution.png", dpi=160)
    plt.close(fig)

    for column, filename, title in [
        ("Quantity", "layer3_quantity_boxplot.png", "Quantity Distribution (Customer Analytics Dataset)"),
        ("UnitPrice", "layer3_unit_price_boxplot.png", "Unit Price Distribution (Customer Analytics Dataset)"),
        ("LineValue", "layer3_line_value_boxplot.png", "Line Value Distribution (Customer Analytics Dataset)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.boxplot(customer[column].dropna(), vert=False, showfliers=True)
        ax.set_title(title)
        ax.set_xlabel(column)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)

    summary = {
        "project": "Comparative Frequent Pattern Mining and Customer Behaviour Analytics",
        "layer": "Layer 3 — Descriptive Statistics and Exploratory Data Analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": {
            "purchase_lines_for_mining": int(len(mining)),
            "customer_purchase_lines": int(len(customer)),
            "unique_invoice_product_pairs": int(len(items)),
            "transactions": int(items["InvoiceNo"].nunique()),
        },
        "key_statistics": {
            "mining_date_min": str(mining["InvoiceDate"].min()),
            "mining_date_max": str(mining["InvoiceDate"].max()),
            "unique_products": int(items["StockCode"].nunique()),
            "mean_distinct_items_per_invoice": round(float(basket_sizes.mean()), 4),
            "median_distinct_items_per_invoice": float(basket_sizes.median()),
            "max_distinct_items_per_invoice": int(basket_sizes.max()),
            "known_customer_revenue_sum": round(float(customer["LineValue"].sum()), 2),
            "top_country_by_known_customer_revenue": str(country_summary.index[0]) if len(country_summary) else None,
            "top_product_code_by_invoice_count": str(product_invoice_counts.index[0]) if len(product_invoice_counts) else None,
        },
        "outputs": {
            "tables": [
                "layer3_numeric_summary.csv",
                "layer3_mining_data_profile.csv",
                "layer3_basket_size_summary.csv",
                "layer3_top_products_by_invoice_count.csv",
                "layer3_monthly_purchase_lines.csv",
                "layer3_monthly_known_customer_revenue.csv",
                "layer3_country_summary.csv",
                "layer3_iqr_outlier_diagnostics.csv",
            ],
            "figures": [
                "layer3_top_products.png",
                "layer3_monthly_purchase_lines.png",
                "layer3_monthly_known_customer_revenue.png",
                "layer3_basket_size_distribution.png",
                "layer3_quantity_boxplot.png",
                "layer3_unit_price_boxplot.png",
                "layer3_line_value_boxplot.png",
            ],
        },
        "caveats": [
            "Revenue is calculated only from customer_purchase_lines.csv, so it excludes rows without a known CustomerID and rows with missing or negative UnitPrice.",
            "The IQR method flags potential outliers; it does not prove that a record is erroneous.",
            "Monthly counts are descriptive and do not by themselves establish seasonality or causation.",
            "Product frequency counts distinct invoices containing each StockCode."
        ],
    }
    (TABLES_DIR / "layer3_eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("LAYER 3 — DESCRIPTIVE STATISTICS AND EDA: PASSED")
    print(f"Mining purchase lines: {len(mining):,}")
    print(f"Transactions: {items['InvoiceNo'].nunique():,}")
    print(f"Distinct products: {items['StockCode'].nunique():,}")
    print(f"Mean basket size: {basket_sizes.mean():.2f}")
    print(f"Median basket size: {basket_sizes.median():.0f}")
    print(f"Maximum basket size: {basket_sizes.max():,}")
    print(f"Known-customer revenue sum: {customer['LineValue'].sum():,.2f}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
