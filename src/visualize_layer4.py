"""
Layer 4A–4D — Frequent Itemset and Association Rule Visualizations

Reads existing CSV outputs; it does not rerun the mining algorithms.

Required inputs under experiments/results/:
    apriori_itemsets.csv
    fp_growth_itemsets.csv
    charm_closed_itemsets.csv
    association_rules.csv

Run from the project root:
    python src/visualize_layer4.py

Save this script as src/visualize_layer4.py inside the project.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents]
    for candidate in candidates:
        if (candidate / "experiments" / "results").is_dir():
            return candidate
    # Common case when running this downloaded script from /mnt/data:
    sibling_project = script_path.parent / "customer-behaviour-analytics"
    if (sibling_project / "experiments" / "results").is_dir():
        return sibling_project
    raise FileNotFoundError(
        "Could not locate the project root. Save this script as "
        "<project-root>/src/visualize_layer4.py and run it from the project root."
    )


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = Path(os.environ.get("ASBD_FIGURES_DIR", str(PROJECT_ROOT / "reports" / "figures")))

ITEMSET_FILES = {
    "Apriori": RESULTS_DIR / "apriori_itemsets.csv",
    "FP-Growth": RESULTS_DIR / "fp_growth_itemsets.csv",
    "CHARM-style closed itemsets": RESULTS_DIR / "charm_closed_itemsets.csv",
}
RULES_FILE = RESULTS_DIR / "association_rules.csv"


def parse_items(value: object) -> list[str]:
    if pd.isna(value):
        return []
    raw = str(value)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw)
    return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]


def itemset_label(value: object) -> str:
    return " + ".join(parse_items(value)) or "(empty itemset)"


def load_itemsets(path: Path, algorithm: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}. Run its Layer 4 script first.")
    df = pd.read_csv(path)
    required = {"itemset", "length", "support_count", "support"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path.name} is empty.")
    for col in ("length", "support_count", "support"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["length", "support_count", "support"]).copy()
    df["length"] = df["length"].astype(int)
    df["itemset_label"] = df["itemset"].map(itemset_label)
    df["algorithm"] = algorithm
    return df


def save_length_chart(df: pd.DataFrame, title: str, filename: str) -> None:
    counts = df["length"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_title(title)
    ax.set_xlabel("Itemset length (number of products)")
    ax.set_ylabel("Number of itemsets")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def save_top_itemsets_chart(df: pd.DataFrame, title: str, filename: str, top_n: int = 15) -> None:
    top = df.sort_values(["support", "support_count"], ascending=False).head(top_n)
    top = top.sort_values("support")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(top))))
    bars = ax.barh(top["itemset_label"], top["support"] * 100)
    ax.bar_label(bars, labels=[f"{x:.2f}%" for x in top["support"] * 100], padding=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Support (%)")
    ax.set_ylabel("Product itemset")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_rules() -> pd.DataFrame:
    if not RULES_FILE.is_file():
        raise FileNotFoundError(f"Missing input: {RULES_FILE}. Run src/association_rules.py first.")
    df = pd.read_csv(RULES_FILE)
    required = {"antecedent", "consequent", "support", "confidence", "lift"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{RULES_FILE.name} is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("association_rules.csv is empty.")
    for col in ("support", "confidence", "lift"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["support", "confidence", "lift"]).copy()
    df["antecedent_label"] = df["antecedent"].map(itemset_label)
    df["consequent_label"] = df["consequent"].map(itemset_label)
    df["rule_label"] = df["antecedent_label"] + "  →  " + df["consequent_label"]
    return df


def save_top_rules_chart(
    rules: pd.DataFrame, metric: str, title: str, filename: str, top_n: int = 15
) -> None:
    top = rules.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(top))))
    values = top[metric]
    bars = ax.barh(top["rule_label"], values)
    labels = [f"{v * 100:.1f}%" for v in values] if metric == "confidence" else [f"{v:.2f}" for v in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Confidence (%)" if metric == "confidence" else "Lift")
    ax.set_ylabel("Association rule")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confidence_lift_scatter(rules: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = (rules["support"].clip(lower=0) * 5000).clip(lower=12, upper=180)
    ax.scatter(rules["confidence"] * 100, rules["lift"], s=sizes, alpha=0.65)
    ax.axvline(30, linestyle="--", linewidth=1, label="30% confidence reference")
    ax.axhline(1, linestyle="--", linewidth=1, label="Lift = 1 reference")
    ax.set_title("Association Rules: Confidence vs Lift")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Lift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "layer4d_confidence_vs_lift.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filenames = []

    chart_config = {
        "Apriori": (
            "layer4a_apriori_itemset_lengths.png",
            "layer4a_apriori_top_itemsets.png",
        ),
        "FP-Growth": (
            "layer4b_fp_growth_itemset_lengths.png",
            "layer4b_fp_growth_top_itemsets.png",
        ),
        "CHARM-style closed itemsets": (
            "layer4c_charm_closed_itemset_lengths.png",
            "layer4c_charm_closed_top_itemsets.png",
        ),
    }

    for algorithm, path in ITEMSET_FILES.items():
        df = load_itemsets(path, algorithm)
        length_file, top_file = chart_config[algorithm]
        save_length_chart(
            df, f"{algorithm}: Itemsets by Length", length_file
        )
        save_top_itemsets_chart(
            df, f"{algorithm}: Top 15 Itemsets by Support", top_file
        )
        filenames.extend([length_file, top_file])
        print(f"{algorithm}: loaded {len(df):,} itemsets")

    rules = load_rules()
    save_top_rules_chart(
        rules, "lift", "Layer 4D: Top Association Rules by Lift",
        "layer4d_top_rules_by_lift.png"
    )
    save_confidence_lift_scatter(rules)
    save_top_rules_chart(
        rules, "confidence", "Layer 4D: Top Association Rules by Confidence",
        "layer4d_top_rules_by_confidence.png"
    )
    filenames.extend([
        "layer4d_top_rules_by_lift.png",
        "layer4d_confidence_vs_lift.png",
        "layer4d_top_rules_by_confidence.png",
    ])

    print(f"Association rules loaded: {len(rules):,}")
    print(f"Generated {len(filenames)} figures in: {FIGURES_DIR}")
    for filename in filenames:
        print(f"  {filename}")
    print(
        "Interpretation note: CHARM-style plots show closed itemsets. "
        "Their counts are not necessarily directly comparable with the "
        "all-frequent-itemset outputs from Apriori and FP-Growth."
    )


if __name__ == "__main__":
    main()
