# Heart Failure Survival Analysis

Dataset: [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)  
N = 299 patients, 13 features, outcome: `DEATH_EVENT`

---

## Project Structure

```
heart-failure-survival/
│
├── data/
│   ├── raw/                          # Original dataset (do not modify)
│   │   └── heart_failure_clinical_records_dataset.csv
│   └── processed/                    # Cleaned / feature-engineered data
│
├── analysis/
│   ├── cohort/                       # Part 1: Overall cohort analysis
│   │   ├── 01_eda.py                 # Exploratory data analysis & visualizations
│   │   └── 02_km_curves.py          # Kaplan-Meier survival curves
│   │
│   └── survival_model/              # Part 2: Individual survival prediction model
│       ├── 01_preprocess.py         # Feature engineering & train/test split
│       ├── 02_train_model.py        # Model training (Cox PH / RSF / DeepSurv)
│       └── 03_evaluate.py           # C-index, calibration, SHAP
│
├── demo/                            # Part 3: Interactive HTML demo
│   ├── app.py                       # Backend (FastAPI or Flask)
│   ├── index.html                   # Frontend UI
│   └── static/
│
├── outputs/
│   ├── figures/                     # Saved plots (PNG/SVG)
│   ├── models/                      # Serialized model files
│   └── tables/                      # Summary statistics tables
│
├── notebooks/                       # Jupyter notebooks for exploration
└── README.md
```

---

## Roadmap

### Part 1 — Cohort Analysis
- [ ] Descriptive statistics (age, sex, comorbidities)
- [ ] Distribution plots for continuous features
- [ ] Correlation heatmap
- [ ] Kaplan-Meier overall survival curve
- [ ] KM curves stratified by key variables (ejection fraction, serum creatinine, etc.)
- [ ] Log-rank tests

### Part 2 — Individual Survival Model
- [ ] Preprocessing & feature engineering
- [ ] Cox Proportional Hazards model (baseline)
- [ ] Random Survival Forest or gradient boosting (GBST/XGBoost-AFT)
- [ ] Model evaluation: C-index, Brier score
- [ ] SHAP feature importance
- [ ] Individual survival curve prediction

### Part 3 — HTML Demo
- [ ] Input form for patient features
- [ ] Real-time individual survival curve rendering (Plotly.js)
- [ ] Risk score display
- [ ] Feature contribution visualization
