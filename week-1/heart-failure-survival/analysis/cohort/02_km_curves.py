"""
analysis/cohort/02_km_curves.py
Part 1 — Kaplan-Meier Survival Curves + Log-rank Tests

Outputs:
  outputs/figures/05_km_overall.png
  outputs/figures/06_km_stratified.png
  outputs/tables/02_logrank_tests.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA  = "data/raw/heart_failure_clinical_records_dataset.csv"
FIG_DIR   = "outputs/figures"
TABLE_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA)
T  = df["time"]           # follow-up duration (days)
E  = df["DEATH_EVENT"]    # event indicator (1 = died)

BLUE  = "#4C9BE8"
RED   = "#E8604C"
GREEN = "#2ECC71"
GRAY  = "#95A5A6"

# ── Helper ─────────────────────────────────────────────────────────────────
def add_at_risk(ax, kmf, t_ticks, y_pos=-0.12, fontsize=8, color="black"):
    """Print 'number at risk' table below the KM plot."""
    at_risk = [kmf.event_table["at_risk"].asof(t) for t in t_ticks]
    for t, n in zip(t_ticks, at_risk):
        ax.text(t, y_pos, str(int(n)), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=fontsize, color=color)

def fmt_p(p):
    if p < 0.0001:
        return "p < 0.0001"
    elif p < 0.001:
        return f"p = {p:.4f}"
    else:
        return f"p = {p:.3f}"

# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Overall KM curve
# ══════════════════════════════════════════════════════════════════════════
print("1. Plotting overall KM curve...")

fig, ax = plt.subplots(figsize=(8, 5))

kmf = KaplanMeierFitter()
kmf.fit(T, E, label="All patients (n=299)")
kmf.plot_survival_function(
    ax=ax, color=BLUE, linewidth=2.5, ci_show=True,
    ci_alpha=0.15, at_risk_counts=False
)

# Median survival line (may be inf if >50% survive at end of follow-up)
median_t = kmf.median_survival_time_
ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1, alpha=0.7)
if np.isfinite(median_t):
    ax.axvline(median_t, color=GRAY, linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Median = {int(median_t)} days",
        xy=(median_t, 0.5), xytext=(median_t + 15, 0.54),
        fontsize=9, color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1)
    )
else:
    ax.text(0.97, 0.53, "Median > follow-up period",
            transform=ax.transAxes, ha="right", fontsize=9, color=GRAY)

# At-risk table
t_ticks = [0, 50, 100, 150, 200, 250]
ax.set_xticks(t_ticks)
ax.text(-18, -0.10, "At risk:", transform=ax.get_xaxis_transform(),
        fontsize=8, ha="right", va="top", fontweight="bold")
add_at_risk(ax, kmf, t_ticks, y_pos=-0.10, color=BLUE)

ax.set_title("Overall Kaplan-Meier Survival Curve\nHeart Failure Cohort (N = 299)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Follow-up Time (days)", fontsize=11)
ax.set_ylabel("Survival Probability", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-5, 290)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper right")

fig.tight_layout()
plt.savefig(f"{FIG_DIR}/05_km_overall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {FIG_DIR}/05_km_overall.png")

# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — Stratified KM curves (2×3 grid)
# ══════════════════════════════════════════════════════════════════════════
print("2. Plotting stratified KM curves...")

# Define stratifications
# Continuous vars → median split; binary vars → 0/1
strats = [
    {
        "col": "ejection_fraction",
        "title": "Ejection Fraction",
        "type": "median",
        "labels": ["Low EF (≤38%)", "High EF (>38%)"],
        "colors": [RED, BLUE],
    },
    {
        "col": "serum_creatinine",
        "title": "Serum Creatinine",
        "type": "median",
        "labels": ["Low SCr (≤1.1)", "High SCr (>1.1)"],
        "colors": [BLUE, RED],
    },
    {
        "col": "age",
        "title": "Age",
        "type": "median",
        "labels": ["Younger (≤60)", "Older (>60)"],
        "colors": [BLUE, RED],
    },
    {
        "col": "serum_sodium",
        "title": "Serum Sodium",
        "type": "median",
        "labels": ["Low Na (≤137)", "High Na (>137)"],
        "colors": [RED, BLUE],
    },
    {
        "col": "anaemia",
        "title": "Anaemia",
        "type": "binary",
        "labels": ["No Anaemia", "Anaemia"],
        "colors": [BLUE, RED],
    },
    {
        "col": "high_blood_pressure",
        "title": "High Blood Pressure",
        "type": "binary",
        "labels": ["No HBP", "HBP"],
        "colors": [BLUE, RED],
    },
]

logrank_rows = []

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Kaplan-Meier Survival Curves — Stratified by Key Variables",
             fontsize=15, fontweight="bold", y=1.01)

for idx, s in enumerate(strats):
    ax = axes[idx // 3][idx % 3]
    col = s["col"]

    # Split groups
    if s["type"] == "median":
        median_val = df[col].median()
        g0 = df[df[col] <= median_val]
        g1 = df[df[col] >  median_val]
    else:
        g0 = df[df[col] == 0]
        g1 = df[df[col] == 1]

    groups = [(g0, s["labels"][0], s["colors"][0]),
              (g1, s["labels"][1], s["colors"][1])]

    for grp_df, label, color in groups:
        kmf_g = KaplanMeierFitter()
        n = len(grp_df)
        kmf_g.fit(grp_df["time"], grp_df["DEATH_EVENT"],
                  label=f"{label} (n={n})")
        kmf_g.plot_survival_function(
            ax=ax, color=color, linewidth=2, ci_show=True,
            ci_alpha=0.12, at_risk_counts=False
        )

    # Log-rank test
    lr = logrank_test(
        g0["time"], g1["time"],
        event_observed_A=g0["DEATH_EVENT"],
        event_observed_B=g1["DEATH_EVENT"]
    )
    p = lr.p_value
    logrank_rows.append({
        "Variable": s["title"],
        "Group 0": s["labels"][0],
        "Group 1": s["labels"][1],
        "N (Group 0)": len(g0),
        "N (Group 1)": len(g1),
        "Log-rank statistic": round(lr.test_statistic, 3),
        "p-value": round(p, 4),
        "Significant": "Yes" if p < 0.05 else "No",
    })

    ax.set_title(s["title"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Days", fontsize=9)
    ax.set_ylabel("Survival Probability", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-5, 290)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")

    # p-value annotation box
    sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text(0.04, 0.08, f"Log-rank {fmt_p(p)}  {sig_str}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.8))

fig.tight_layout()
plt.savefig(f"{FIG_DIR}/06_km_stratified.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {FIG_DIR}/06_km_stratified.png")

# ── Save log-rank results table ────────────────────────────────────────────
lr_df = pd.DataFrame(logrank_rows)
lr_df.to_csv(f"{TABLE_DIR}/02_logrank_tests.csv", index=False)

print("\nLog-rank Test Results:")
print(lr_df.to_string(index=False))
print(f"\n   Saved → {TABLE_DIR}/02_logrank_tests.csv")
print("\n✓ KM analysis complete.")
