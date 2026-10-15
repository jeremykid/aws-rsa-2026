"""
analysis/survival_model/02_train_model.py
Part 2 — Model Training

Models:
  1. Cox Proportional Hazards (baseline, interpretable)
  2. Random Survival Forest  (non-parametric, handles non-linearity)

Evaluation strategy:
  - 5-fold CV on train set  → report mean ± std C-index
  - Hold-out test set       → final unbiased C-index
  - Full dataset            → final model saved for demo

Outputs:
  outputs/models/cox_final.pkl
  outputs/models/rsf_final.pkl
  outputs/tables/04_cv_results.csv
  outputs/tables/05_test_results.csv
  outputs/figures/07_cox_coefficients.png
  outputs/figures/08_rsf_feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ── Paths ──────────────────────────────────────────────────────────────────
PROC_DIR  = "data/processed"
MODEL_DIR = "outputs/models"
TABLE_DIR = "outputs/tables"
FIG_DIR   = "outputs/figures"
for d in [MODEL_DIR, TABLE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42

CONTINUOUS = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium",
]
BINARY   = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
FEATURES = CONTINUOUS + BINARY

LABEL_MAP = {
    "age":                       "Age",
    "creatinine_phosphokinase":  "CPK",
    "ejection_fraction":         "Ejection Fraction",
    "platelets":                 "Platelets",
    "serum_creatinine":          "Serum Creatinine",
    "serum_sodium":              "Serum Sodium",
    "anaemia":                   "Anaemia",
    "diabetes":                  "Diabetes",
    "high_blood_pressure":       "High Blood Pressure",
    "sex":                       "Sex (Male)",
    "smoking":                   "Smoking",
}

# ── Load processed data ────────────────────────────────────────────────────
train_df = pd.read_csv(f"{PROC_DIR}/train.csv")
test_df  = pd.read_csv(f"{PROC_DIR}/test.csv")
full_df  = pd.read_csv(f"{PROC_DIR}/full_scaled.csv")

def make_survival_array(df):
    """Convert time/event columns to sksurv structured array."""
    return np.array(
        [(bool(e), t) for e, t in zip(df["DEATH_EVENT"], df["time"])],
        dtype=[("event", bool), ("time", float)]
    )

X_train = train_df[FEATURES].values
y_train = make_survival_array(train_df)

X_test  = test_df[FEATURES].values
y_test  = make_survival_array(test_df)

X_full  = full_df[FEATURES].values
y_full  = make_survival_array(full_df)

cv_folds = train_df["cv_fold"].values

# ── Helper: C-index ────────────────────────────────────────────────────────
def c_index(model, X, y):
    risk = model.predict(X)
    return concordance_index_censored(y["event"], y["time"], risk)[0]

# ══════════════════════════════════════════════════════════════════════════
# 1. Cox Proportional Hazards
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Cox Proportional Hazards")
print("=" * 60)

cox = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron", n_iter=200)

# 5-fold CV
cox_cv_scores = []
for fold in range(5):
    val_mask   = cv_folds == fold
    train_mask = ~val_mask
    cox_cv = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron", n_iter=200)
    cox_cv.fit(X_train[train_mask], y_train[train_mask])
    score = c_index(cox_cv, X_train[val_mask], y_train[val_mask])
    cox_cv_scores.append(score)
    print(f"  Fold {fold}: C-index = {score:.4f}")

print(f"  CV C-index: {np.mean(cox_cv_scores):.4f} ± {np.std(cox_cv_scores):.4f}")

# Train on full train set → evaluate on test
cox.fit(X_train, y_train)
cox_test_ci = c_index(cox, X_test, y_test)
print(f"  Test C-index: {cox_test_ci:.4f}")

# Train final model on ALL data
cox_final = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron", n_iter=200)
cox_final.fit(X_full, y_full)
print(f"  Final model trained on full dataset ({len(X_full)} samples)")

# ── Cox coefficients plot ──────────────────────────────────────────────────
coef = pd.Series(cox.coef_, index=FEATURES).rename(LABEL_MAP)
coef_sorted = coef.sort_values()
colors = ["#E8604C" if v > 0 else "#4C9BE8" for v in coef_sorted]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(coef_sorted.index, coef_sorted.values, color=colors,
               height=0.6, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
for bar, val in zip(bars, coef_sorted.values):
    ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8.5)
ax.set_title("Cox PH — Log Hazard Ratio Coefficients\n"
             f"(Train C-index: {np.mean(cox_cv_scores):.3f}  Test C-index: {cox_test_ci:.3f})",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Log Hazard Ratio (scaled features)", fontsize=10)
pos_patch = mpatches.Patch(color="#E8604C", label="↑ Higher risk")
neg_patch = mpatches.Patch(color="#4C9BE8", label="↓ Lower risk")
ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.savefig(f"{FIG_DIR}/07_cox_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/07_cox_coefficients.png")

# ══════════════════════════════════════════════════════════════════════════
# 2. Random Survival Forest
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Random Survival Forest")
print("=" * 60)

rsf = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# 5-fold CV
rsf_cv_scores = []
for fold in range(5):
    val_mask   = cv_folds == fold
    train_mask = ~val_mask
    rsf_cv = RandomSurvivalForest(
        n_estimators=300, min_samples_split=10, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
    )
    rsf_cv.fit(X_train[train_mask], y_train[train_mask])
    score = c_index(rsf_cv, X_train[val_mask], y_train[val_mask])
    rsf_cv_scores.append(score)
    print(f"  Fold {fold}: C-index = {score:.4f}")

print(f"  CV C-index: {np.mean(rsf_cv_scores):.4f} ± {np.std(rsf_cv_scores):.4f}")

# Train on full train set → evaluate on test
rsf.fit(X_train, y_train)
rsf_test_ci = c_index(rsf, X_test, y_test)
print(f"  Test C-index: {rsf_test_ci:.4f}")

# Train final model on ALL data
rsf_final = RandomSurvivalForest(
    n_estimators=300, min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
)
rsf_final.fit(X_full, y_full)
print(f"  Final model trained on full dataset ({len(X_full)} samples)")

# ── RSF feature importance via permutation ─────────────────────────────────
print("  Computing permutation importances (this may take ~30s)...")
baseline_ci = c_index(rsf, X_train, y_train)
perm_importances = []
rng = np.random.default_rng(RANDOM_STATE)
for i, feat in enumerate(FEATURES):
    scores = []
    for _ in range(10):   # 10 repeats for stability
        X_perm = X_train.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        scores.append(c_index(rsf, X_perm, y_train))
    perm_importances.append(baseline_ci - np.mean(scores))

importances = pd.Series(perm_importances, index=FEATURES).rename(LABEL_MAP)
imp_sorted  = importances.sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors_imp = ["#E8604C" if v > 0 else "#AAAAAA" for v in imp_sorted]
ax.barh(imp_sorted.index, imp_sorted.values,
        color=colors_imp, height=0.6, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
for i, (name, val) in enumerate(imp_sorted.items()):
    ax.text(val + (0.001 if val >= 0 else -0.001), i,
            f"{val:.4f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8.5)
ax.set_title("Random Survival Forest — Permutation Feature Importance\n"
             f"(Train C-index: {np.mean(rsf_cv_scores):.3f}  Test C-index: {rsf_test_ci:.3f})",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Drop in C-index when feature is permuted", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.savefig(f"{FIG_DIR}/08_rsf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/08_rsf_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. Save models & results
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Saving models & results")
print("=" * 60)

with open(f"{MODEL_DIR}/cox_final.pkl", "wb") as f:
    pickle.dump({"model": cox_final, "features": FEATURES}, f)
with open(f"{MODEL_DIR}/rsf_final.pkl", "wb") as f:
    pickle.dump({"model": rsf_final, "features": FEATURES}, f)
print(f"  Models saved → {MODEL_DIR}/")

# CV results table
cv_df = pd.DataFrame({
    "Fold":        list(range(5)) + ["Mean", "Std"],
    "Cox C-index": cox_cv_scores  + [np.mean(cox_cv_scores), np.std(cox_cv_scores)],
    "RSF C-index": rsf_cv_scores  + [np.mean(rsf_cv_scores), np.std(rsf_cv_scores)],
}).round(4)
cv_df.to_csv(f"{TABLE_DIR}/04_cv_results.csv", index=False)
print(f"\n5-Fold CV Results:")
print(cv_df.to_string(index=False))

# Test results table
test_df_results = pd.DataFrame([
    {"Model": "Cox PH",                "Test C-index": round(cox_test_ci, 4),
     "CV C-index (mean)": round(np.mean(cox_cv_scores), 4),
     "CV C-index (std)":  round(np.std(cox_cv_scores), 4)},
    {"Model": "Random Survival Forest","Test C-index": round(rsf_test_ci, 4),
     "CV C-index (mean)": round(np.mean(rsf_cv_scores), 4),
     "CV C-index (std)":  round(np.std(rsf_cv_scores), 4)},
])
test_df_results.to_csv(f"{TABLE_DIR}/05_test_results.csv", index=False)
print(f"\nTest Set Results:")
print(test_df_results.to_string(index=False))

print("\n✓ Training complete.")
