# Customer Behaviour Analytics --- Quick Start

## 1. Set up the environment

Open a terminal in the project root (`customer-behaviour-analytics/`)
and create/activate a virtual environment if you use one.

Install the project dependencies from the repository's dependency file,
if present. For example:

``` bash
pip install -r requirements.txt
```

If the repository does not contain `requirements.txt`, install the
packages required by the scripts you plan to run (commonly `pandas`,
`numpy`, `matplotlib`, and `scikit-learn`).

## 2. Run the pipeline in order

Run each layer from the **project root**. Use the scripts and filenames
that exist in your checkout; names below follow the current project
structure.

1.  **Layer 1 --- Ingestion / initial exploration:** run the Layer 1
    ingestion or exploration script.
2.  **Layer 2 --- Preprocessing:** run the preprocessing /
    transaction-construction script.
3.  **Layer 3 --- Exploratory analysis:** run the EDA script to create
    exploratory figures.
4.  **Layer 4A --- Apriori:** `python src/apriori.py`
5.  **Layer 4B --- FP-Growth:** `python src/fp_growth.py`
6.  **Layer 4C --- CHARM-style mining:** `python src/charm.py`
7.  **Layer 4D --- Association rules:**
    `python src/association_rules.py`
8.  **Layer 5 --- Customer segmentation:**
    `python src/customer_segmentation.py`
9.  **Layer 6 --- Customer classification:**
    `python src/customer_classification.py`
10. **Layer 7 --- Mining benchmarks:** `python src/benchmark_mining.py`
11. **Layer 8 --- Artifact validation:**
    `python src/validate_project.py`

**Important:** Layers depend on outputs from earlier layers. Run them in
sequence, and check each script's output before continuing. Layer 1--3
script names may differ in your local checkout, so use the actual
filenames in `src/`.

## 3. Generate visualization figures

After running the corresponding analysis layer, run these visualization
scripts:

``` bash
python src/visualize_layer4.py
python src/visualize_layer5.py
python src/visualize_layer6.py
python src/visualize_layer7.py
```

These scripts read existing result CSV/JSON files; they do **not** rerun
the analysis. By default, generated figures are saved to
`reports/figures/`. If a visualization script is stored outside `src/`,
move it into `src/` first or adjust its project-root lookup as needed.

## 4. Check outputs

-   Analysis outputs are expected under `experiments/results/`.
-   Figures are expected under `reports/figures/`.
-   Run Layer 8 only after the required output artifacts have been
    generated.
-   A successful structural validation confirms that expected files and
    basic shapes exist; it does not, by itself, prove the mining
    algorithms or model results are analytically correct.

## 5. If a script fails

1.  Read the full error message and note the script name.
2.  Confirm you are running the command from the project root.
3.  Confirm the required input files exist in `experiments/results/` (or
    the path named by that script).
4.  Check that CSV column names match what the script expects.
5.  Fix the first error before running the next layer.
