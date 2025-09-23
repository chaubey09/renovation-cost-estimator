# Interior Renovation & Design Cost Estimator (DVC Scaffold)

A 4-stage DVC pipeline:
- **M1:** Tabular Preprocessing
- **M2:** Quantity Estimation (from dimensions & detections)
- **M3:** Cost Mapping (city, labor, material indices)
- **M4:** Fusion, Calibration & Reporting

## Quickstart

```bash
# 0) (Recommended) Create a virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# 1) Initialize DVC (once)
dvc init

# 2) Track raw data
dvc add data/raw/tabular.csv

# 3) Run pipeline
dvc repro

# 4) See metrics & outputs
dvc metrics show
python -c "import pandas as pd; print(pd.read_parquet('outputs/reports/final_estimates.parquet').head())"

# (Optional) Visualize simple plot produced by M3
dvc plots show outputs/plots/cost_per_sqft_by_city.csv
```

## Replace the sample
- Update `data/raw/tabular.csv` with your real dataset (keep headers).
- Optionally edit `data/external/cost_indices.csv` & `data/external/packages.csv` for your markets.
- If you have computer-vision detections, write them to `data/interim/detections.jsonl` with per-row overrides.
