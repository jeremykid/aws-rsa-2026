"""
analysis/survival_model/03_evaluate.py
Part 2 — Model Evaluation

Metrics (all on hold-out test set):
  1. C-index (Harrell)          — discrimination
  2. IBS    (IPCW-weighted)     — overall probabilistic accuracy
  3. MAE-PO (pseudo-observation)— absolute time prediction error
  4. D-calibration              — distributional calibration (chi-square)

Outputs:
  outputs/tables/06_evaluation_summary.csv
  outputs/figures/09_ibs_curve.png
  outputs/figures/10_dcal_histogram.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import warnings
import os
from scipy.stats import chi2
warnings.filterwarnings("ignore")

from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    integrated_brier_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────
PROC_DIR  = "data/processed"
MODEL_DIR = "outputs/models"
TABLE_DIR = "outputs/tables"
FIG_DIR   = "outputs/figures"
for d in [TABLE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium",
    "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
]

# ── Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv(f"{PROC_DIR}/train.csv")
test_df  = pd.read_csv(f"{PROC_DIR}/test.csv")

def make_survival_array(df):
    return np.array(
        [(bool(e), t) for e, t in zip(df["DEATH_EVENT"], df["time"])],
        dtype=[("event", bool), ("time", float)]
    )

X_train = train_df[FEATURES].values
y_train = make_survival_array(train_df)
X_test  = test_df[FEATURES].values
y_test  = make_survival_array(test_df)

# ── Load models ────────────────────────────────────────────────────────────
with open(f"{MODEL_DIR}/cox_final.pkl", "rb") as f:
    cox_bundle = pickle.load(f)
    cox = cox_bundle["model"]

with open(f"{MODEL_DIR}/rsf_final.pkl", "rb") as f:
    rsf_bundle = pickle.load(f)
    rsf = rsf_bundle["model"]

# Retrain on train set only for unbiased test evaluation
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

cox_eval = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron", n_iter=200)
cox_eval.fit(X_train, y_train)

rsf_eval = RandomSurvivalForest(
    n_estimators=300, min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt", n_jobs=-1, random_state=42,
)
rsf_eval.fit(X_train, y_train)

# ── Time grid for IBS ──────────────────────────────────────────────────────
# Must be strictly within [min(test time), max(test time))
# and also within training event time range
t_min = max(y_train["time"].min(), y_test["time"].min()) + 1
t_max = min(
    y_train["time"][y_train["event"]].max(),
    y_test["time"].max()
) - 1
times = np.linspace(t_min, t_max, 100)

# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

def get_c_index(model, X, y):
    risk = model.predict(X)
    ci, _, _, _, _ = concordance_index_censored(y["event"], y["time"], risk)
    return ci

def get_surv_probs(model, X, times):
    """Return (n_samples × n_times) matrix of S(t|x)."""
    step_fns = model.predict_survival_function(X)
    mat = np.row_stack([fn(times) for fn in step_fns])
    return mat

def get_ibs(model, X_tr, y_tr, X_te, y_te, times):
    """IPCW-weighted IBS using sksurv."""
    surv_mat = get_surv_probs(model, X_te, times)
    # sksurv expects (n_times × n_samples) transposed — actually (n_samples x n_times) since 0.22
    ibs = integrated_brier_score(y_tr, y_te, surv_mat, times)
    return ibs

def get_brier_curve(model, X_tr, y_tr, X_te, y_te, times):
    surv_mat = get_surv_probs(model, X_te, times)
    _, bs_vals = brier_score(y_tr, y_te, surv_mat, times)
    return bs_vals

# ══════════════════════════════════════════════════════════════════════════
# MAE-PO (Pseudo-observation method)
# Based on: Andersen & Pohar Perme (2010)
# For each subject i: pseudo_i = n * RMST_full - (n-1) * RMST_{-i}
# where RMST = restricted mean survival time up to tau = max follow-up
# ══════════════════════════════════════════════════════════════════════════

def kaplan_meier_rmst(times, events, tau):
    """KM-based RMST up to tau."""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(times, events)
    sf = kmf.survival_function_
    t_vals = sf.index.values
    s_vals = sf["KM_estimate"].values
    # Integrate S(t) dt from 0 to tau
    mask = t_vals <= tau
    t_clip = np.append(t_vals[mask], tau)
    s_clip = np.append(s_vals[mask], s_vals[mask][-1] if mask.any() else 1.0)
    return np.trapz(s_clip, t_clip)

def mae_pseudo_obs(model, X_tr, y_tr, X_te, y_te):
    """
    MAE via pseudo-observations of predicted median survival time.
    Predicted time = median of predicted survival curve.
    Pseudo-observed time = jackknife pseudo-obs of KM RMST on test set.
    MAE = mean |pseudo_i - predicted_i|
    """
    from lifelines import KaplanMeierFitter

    n   = len(y_te)
    tau = y_te["time"].max()   # restrict to observed follow-up

    # Full KM RMST on test set
    rmst_full = kaplan_meier_rmst(y_te["time"], y_te["event"].astype(int), tau)

    # Jackknife pseudo-observations
    pseudo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rmst_loo = kaplan_meier_rmst(
            y_te["time"][mask], y_te["event"][mask].astype(int), tau
        )
        pseudo[i] = n * rmst_full - (n - 1) * rmst_loo

    # Predicted median survival time from model
    step_fns = model.predict_survival_function(X_te)
    pred_times = []
    for fn in step_fns:
        t_pts = fn.x
        s_pts = fn(t_pts)
        # Find first t where S(t) <= 0.5
        idx = np.searchsorted(-s_pts, -0.5)
        if idx >= len(t_pts):
            pred_times.append(t_pts[-1])   # censored at last observed time
        else:
            pred_times.append(t_pts[idx])
    pred_times = np.array(pred_times)

    mae = np.mean(np.abs(pseudo - pred_times))
    return mae, pseudo, pred_times

# ══════════════════════════════════════════════════════════════════════════
# D-calibration
# Predicted S(t_i | x_i) should be ~ Uniform[0,1] for event subjects
# Chi-square test with num_bins bins
# ══════════════════════════════════════════════════════════════════════════

def d_calibration(model, X_te, y_te, num_bins=10):
    """
    For each test subject, get predicted S(t_i | x_i).
    For events: should follow Uniform(0,1) → chi-square test.
    For censored: use fractional bin contribution.
    """
    step_fns = model.predict_survival_function(X_te)
    pred_probs = np.array([
        fn(min(t, fn.domain[1]))
        for fn, t in zip(step_fns, y_te["time"])
    ])

    # Bin edges [0, 0.1, 0.2, ..., 1.0]
    bins = np.linspace(0, 1, num_bins + 1)
    counts = np.zeros(num_bins)

    for i, (p, e) in enumerate(zip(pred_probs, y_te["event"])):
        if e:  # observed event → single bin
            b = min(int(p * num_bins), num_bins - 1)
            counts[b] += 1
        else:   # censored → fractional contribution above p
            # subject survived past p, so contributes to bins >= p
            for j in range(num_bins):
                bin_lo = j / num_bins
                bin_hi = (j + 1) / num_bins
                if bin_hi <= p:
                    continue
                overlap_lo = max(bin_lo, p)
                overlap_hi = bin_hi
                if overlap_lo < overlap_hi:
                    frac = (overlap_hi - overlap_lo) / (1 - p + 1e-10)
                    counts[j] += frac

    expected = counts.sum() / num_bins
    chi2_stat = np.sum((counts - expected) ** 2 / (expected + 1e-10))
    p_val = 1 - chi2.cdf(chi2_stat, df=num_bins - 1)

    return chi2_stat, p_val, counts, pred_probs

# ══════════════════════════════════════════════════════════════════════════
# Run all evaluations
# ══════════════════════════════════════════════════════════════════════════
results = {}

for name, model in [("Cox PH", cox_eval), ("RSF", rsf_eval)]:
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    # 1. C-index
    ci = get_c_index(model, X_test, y_test)
    print(f"  C-index (Harrell):    {ci:.4f}")

    # 2. IBS
    ibs = get_ibs(model, X_train, y_train, X_test, y_test, times)
    bs_curve = get_brier_curve(model, X_train, y_train, X_test, y_test, times)
    print(f"  IBS (IPCW):           {ibs:.4f}")

    # 3. MAE-PO
    mae, pseudo_obs, pred_t = mae_pseudo_obs(model, X_train, y_train, X_test, y_test)
    print(f"  MAE-PO:               {mae:.2f} days")

    # 4. D-calibration
    chi2_stat, p_dcal, dcal_counts, surv_at_event = d_calibration(model, X_test, y_test)
    sig = "✓ calibrated" if p_dcal > 0.05 else "✗ miscalibrated"
    print(f"  D-calibration χ²:     {chi2_stat:.3f}  p={p_dcal:.4f}  {sig}")

    results[name] = {
        "C-index": ci,
        "IBS": ibs,
        "MAE-PO (days)": mae,
        "D-cal χ²": chi2_stat,
        "D-cal p-value": p_dcal,
        "D-cal bs_curve": bs_curve,
        "D-cal counts": dcal_counts,
        "surv_at_event": surv_at_event,
    }

# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — IBS Brier Score Curve (both models)
# ══════════════════════════════════════════════════════════════════════════
print("\nPlotting IBS curve...")
fig, ax = plt.subplots(figsize=(8, 4.5))

colors = {"Cox PH": "#4C9BE8", "RSF": "#E8604C"}
for name, res in results.items():
    ax.plot(times, res["D-cal bs_curve"], color=colors[name], linewidth=2,
            label=f"{name}  (IBS={res['IBS']:.3f})")

ax.fill_between(times, results["Cox PH"]["D-cal bs_curve"],
                results["RSF"]["D-cal bs_curve"],
                alpha=0.08, color="gray")
ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6,
           label="Reference (null model ≈ 0.25)")
ax.set_xlabel("Follow-up Time (days)", fontsize=11)
ax.set_ylabel("Brier Score", fontsize=11)
ax.set_title("Time-Dependent Brier Score (IPCW-weighted)\nHold-out Test Set (n=60)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.savefig(f"{FIG_DIR}/09_ibs_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/09_ibs_curve.png")

# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — D-calibration histograms (side by side)
# ══════════════════════════════════════════════════════════════════════════
print("Plotting D-calibration histograms...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("D-Calibration: Distribution of Predicted Survival Probabilities at Event Times",
             fontsize=12, fontweight="bold")

num_bins = 10
bin_labels = [f"{i/num_bins:.1f}–{(i+1)/num_bins:.1f}" for i in range(num_bins)]
x = np.arange(num_bins)
expected = sum(results["Cox PH"]["D-cal counts"]) / num_bins   # same n for both

for ax, (name, res) in zip(axes, results.items()):
    counts = res["D-cal counts"]
    bars = ax.bar(x, counts, color=colors[name], alpha=0.75, edgecolor="white", width=0.7)
    ax.axhline(expected, color="black", linestyle="--", linewidth=1.5,
               label=f"Expected (uniform) = {expected:.1f}")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{cnt:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Predicted S(t_event | x) bin", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    p_val = res["D-cal p-value"]
    sig = "calibrated ✓" if p_val > 0.05 else "miscalibrated ✗"
    ax.set_title(f"{name}\nχ²={res['D-cal χ²']:.2f}  p={p_val:.3f}  ({sig})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
plt.savefig(f"{FIG_DIR}/10_dcal_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/10_dcal_histogram.png")

# ══════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame([
    {
        "Model": name,
        "C-index ↑":         round(res["C-index"], 4),
        "IBS ↓":             round(res["IBS"], 4),
        "MAE-PO (days) ↓":   round(res["MAE-PO (days)"], 2),
        "D-cal χ²":          round(res["D-cal χ²"], 3),
        "D-cal p-value":     round(res["D-cal p-value"], 4),
        "D-cal (p>0.05?)":   "✓" if res["D-cal p-value"] > 0.05 else "✗",
    }
    for name, res in results.items()
])
summary.to_csv(f"{TABLE_DIR}/06_evaluation_summary.csv", index=False)
print(f"\n{'='*55}")
print("EVALUATION SUMMARY")
print(f"{'='*55}")
print(summary.to_string(index=False))
print(f"\n  Saved → {TABLE_DIR}/06_evaluation_summary.csv")
print("\n✓ Evaluation complete.")
