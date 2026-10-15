"""
analysis/survival_model/01_preprocess.py
Part 2 — Feature Engineering & Data Split

Strategy:
  - 80% train / 20% hold-out test  (stratified by DEATH_EVENT)
  - Continuous features → StandardScaler (fit on train only)
  - Binary features     → pass through as-is
  - 5-fold CV indices saved alongside for model evaluation

Outputs:
  data/processed/train.csv
  data/processed/test.csv
  data/processed/full_scaled.csv       ← for final demo model (all 299)
  data/processed/scaler_params.csv     ← mean/std for each continuous feature
  outputs/tables/03_split_summary.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA  = "data/raw/heart_failure_clinical_records_dataset.csv"
PROC_DIR  = "data/processed"
TABLE_DIR = "outputs/tables"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

RANDOM_STATE = 42

CONTINUOUS = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium",
]
# Note: 'time' is the survival duration — NOT a predictor feature
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
FEATURES = CONTINUOUS + BINARY

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

X = df[FEATURES].copy()
y = df[["time", "DEATH_EVENT"]].copy()   # both needed for survival analysis

# ── Train / Test split (stratified by DEATH_EVENT) ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y["DEATH_EVENT"],
)

print(f"\nSplit (stratified by DEATH_EVENT):")
print(f"  Train : {len(X_train):3d} samples  "
      f"(deaths: {y_train['DEATH_EVENT'].sum():2d} / {len(y_train)}, "
      f"{y_train['DEATH_EVENT'].mean()*100:.1f}%)")
print(f"  Test  : {len(X_test):3d} samples  "
      f"(deaths: {y_test['DEATH_EVENT'].sum():2d} / {len(y_test)}, "
      f"{y_test['DEATH_EVENT'].mean()*100:.1f}%)")

# ── Scale continuous features (fit on train only) ─────────────────────────
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])
X_test_scaled[CONTINUOUS]  = scaler.transform(X_test[CONTINUOUS])

# Save scaler params for use in demo (backend needs to scale new inputs)
scaler_params = pd.DataFrame({
    "feature": CONTINUOUS,
    "mean":    scaler.mean_,
    "std":     scaler.scale_,
})
scaler_params.to_csv(f"{PROC_DIR}/scaler_params.csv", index=False)
print(f"\nScaler params saved → {PROC_DIR}/scaler_params.csv")
print(scaler_params.to_string(index=False))

# ── Scale full dataset (for final demo model) ──────────────────────────────
X_full_scaled = X.copy()
# Fit scaler on ALL data for the final model
scaler_full = StandardScaler()
X_full_scaled[CONTINUOUS] = scaler_full.fit_transform(X[CONTINUOUS])

scaler_full_params = pd.DataFrame({
    "feature": CONTINUOUS,
    "mean":    scaler_full.mean_,
    "std":     scaler_full.scale_,
})
scaler_full_params.to_csv(f"{PROC_DIR}/scaler_full_params.csv", index=False)

# ── Save processed datasets ────────────────────────────────────────────────
train_out = pd.concat([X_train_scaled.reset_index(drop=True),
                       y_train.reset_index(drop=True)], axis=1)
test_out  = pd.concat([X_test_scaled.reset_index(drop=True),
                       y_test.reset_index(drop=True)], axis=1)
full_out  = pd.concat([X_full_scaled.reset_index(drop=True),
                       y.reset_index(drop=True)], axis=1)

train_out.to_csv(f"{PROC_DIR}/train.csv", index=False)
test_out.to_csv(f"{PROC_DIR}/test.csv",   index=False)
full_out.to_csv(f"{PROC_DIR}/full_scaled.csv", index=False)

print(f"\nProcessed data saved:")
print(f"  {PROC_DIR}/train.csv        ({len(train_out)} rows)")
print(f"  {PROC_DIR}/test.csv         ({len(test_out)} rows)")
print(f"  {PROC_DIR}/full_scaled.csv  ({len(full_out)} rows)")

# ── Generate 5-fold CV indices (saved as fold column in train) ─────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_out["cv_fold"] = -1
for fold, (_, val_idx) in enumerate(
    skf.split(X_train_scaled, y_train["DEATH_EVENT"])
):
    train_out.loc[val_idx, "cv_fold"] = fold

train_out.to_csv(f"{PROC_DIR}/train.csv", index=False)   # re-save with fold col

print(f"\n5-fold CV fold distribution (train set):")
for fold in range(5):
    fold_df = train_out[train_out["cv_fold"] == fold]
    n_events = fold_df["DEATH_EVENT"].sum()
    print(f"  Fold {fold}: n={len(fold_df):3d}  events={n_events:2d}  "
          f"({n_events/len(fold_df)*100:.1f}%)")

# ── Summary table ──────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"Split": "Train", "N": len(train_out), "Deaths": int(y_train["DEATH_EVENT"].sum()),
     "Death Rate": f"{y_train['DEATH_EVENT'].mean()*100:.1f}%"},
    {"Split": "Test",  "N": len(test_out),  "Deaths": int(y_test["DEATH_EVENT"].sum()),
     "Death Rate": f"{y_test['DEATH_EVENT'].mean()*100:.1f}%"},
    {"Split": "Full",  "N": len(full_out),  "Deaths": int(y["DEATH_EVENT"].sum()),
     "Death Rate": f"{y['DEATH_EVENT'].mean()*100:.1f}%"},
])
summary.to_csv(f"{TABLE_DIR}/03_split_summary.csv", index=False)
print(f"\nSplit summary saved → {TABLE_DIR}/03_split_summary.csv")

print("\n✓ Preprocessing complete.")
