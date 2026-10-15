"""
analysis/cohort/01_eda.py
Part 1 — Exploratory Data Analysis
Outputs:
  outputs/tables/01_descriptive_stats.csv
  outputs/figures/01_continuous_distributions.png
  outputs/figures/02_binary_prevalence.png
  outputs/figures/03_boxplots_by_outcome.png
  outputs/figures/04_correlation_heatmap.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA   = "data/raw/heart_failure_clinical_records_dataset.csv"
FIG_DIR    = "outputs/figures"
TABLE_DIR  = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA)

CONTINUOUS = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium", "time",
]
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
OUTCOME = "DEATH_EVENT"

LABEL_MAP = {
    "age":                        "Age (years)",
    "creatinine_phosphokinase":   "CPK (mcg/L)",
    "ejection_fraction":          "Ejection Fraction (%)",
    "platelets":                  "Platelets (kiloplatelets/mL)",
    "serum_creatinine":           "Serum Creatinine (mg/dL)",
    "serum_sodium":               "Serum Sodium (mEq/L)",
    "time":                       "Follow-up Time (days)",
    "anaemia":                    "Anaemia",
    "diabetes":                   "Diabetes",
    "high_blood_pressure":        "High Blood Pressure",
    "sex":                        "Sex (1=Male)",
    "smoking":                    "Smoking",
}

PALETTE = {"Survived": "#4C9BE8", "Died": "#E8604C"}

df["Outcome"] = df[OUTCOME].map({0: "Survived", 1: "Died"})

# ── 1. Descriptive statistics table ───────────────────────────────────────
print("=" * 60)
print("1. Descriptive Statistics")
print("=" * 60)

rows = []
for col in CONTINUOUS:
    for grp_label, grp_df in [("Overall", df), ("Survived", df[df[OUTCOME]==0]), ("Died", df[df[OUTCOME]==1])]:
        s = grp_df[col]
        rows.append({
            "Variable": LABEL_MAP[col],
            "Group": grp_label,
            "N": len(s),
            "Mean": round(s.mean(), 2),
            "SD": round(s.std(), 2),
            "Median": round(s.median(), 2),
            "IQR": f"{round(s.quantile(0.25),2)}–{round(s.quantile(0.75),2)}",
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
        })

desc_df = pd.DataFrame(rows)
desc_df.to_csv(f"{TABLE_DIR}/01_descriptive_stats.csv", index=False)
print(desc_df.to_string(index=False))

# Mann-Whitney U tests for continuous vars
print("\nMann-Whitney U tests (Survived vs Died):")
for col in CONTINUOUS:
    g0 = df.loc[df[OUTCOME]==0, col]
    g1 = df.loc[df[OUTCOME]==1, col]
    u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {LABEL_MAP[col]:<35}  p={p:.4f}  {sig}")

# ── 2. Continuous variable distributions ──────────────────────────────────
print("\n2. Plotting continuous distributions...")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
fig.suptitle("Distribution of Continuous Variables by Outcome", fontsize=15, fontweight="bold", y=1.01)

for i, col in enumerate(CONTINUOUS):
    ax = axes[i // 3][i % 3]
    for outcome, color in [("Survived", "#4C9BE8"), ("Died", "#E8604C")]:
        subset = df[df["Outcome"] == outcome][col]
        ax.hist(subset, bins=25, alpha=0.6, color=color, label=outcome, density=True)
        # KDE overlay
        kde_x = np.linspace(subset.min(), subset.max(), 200)
        kde = stats.gaussian_kde(subset)
        ax.plot(kde_x, kde(kde_x), color=color, linewidth=2)
    ax.set_title(LABEL_MAP[col], fontsize=10, fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide the last empty subplot
axes[2][2].set_visible(False)

fig.tight_layout()
plt.savefig(f"{FIG_DIR}/01_continuous_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/01_continuous_distributions.png")

# ── 3. Binary variable prevalence ─────────────────────────────────────────
print("\n3. Plotting binary variable prevalence...")

fig, axes = plt.subplots(1, 5, figsize=(14, 4))
fig.suptitle("Binary Feature Prevalence by Outcome", fontsize=14, fontweight="bold")

for i, col in enumerate(BINARY):
    ax = axes[i]
    prev = df.groupby("Outcome")[col].mean() * 100  # %
    bars = ax.bar(prev.index, prev.values,
                  color=[PALETTE[k] for k in prev.index],
                  width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, prev.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(LABEL_MAP[col], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Prevalence (%)" if i == 0 else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Chi-square test
    ct = pd.crosstab(df[OUTCOME], df[col])
    chi2, p, *_ = stats.chi2_contingency(ct)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.set_xlabel(f"χ² p={p:.3f} {sig}", fontsize=8)

fig.tight_layout()
plt.savefig(f"{FIG_DIR}/02_binary_prevalence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/02_binary_prevalence.png")

# ── 4. Box plots by outcome ────────────────────────────────────────────────
print("\n4. Plotting boxplots by outcome...")

fig, axes = plt.subplots(2, 4, figsize=(15, 9))
fig.suptitle("Continuous Variables by Outcome (Box + Strip)", fontsize=14, fontweight="bold")

for i, col in enumerate(CONTINUOUS):
    ax = axes[i // 4][i % 4]
    sns.boxplot(data=df, x="Outcome", y=col, hue="Outcome", palette=PALETTE,
                width=0.45, linewidth=1.5, fliersize=0, legend=False, ax=ax)
    sns.stripplot(data=df, x="Outcome", y=col, hue="Outcome", palette=PALETTE,
                  alpha=0.35, size=3, jitter=True, legend=False, ax=ax)
    ax.set_title(LABEL_MAP[col], fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate p-value
    g0 = df.loc[df[OUTCOME]==0, col]
    g1 = df.loc[df[OUTCOME]==1, col]
    _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else f"p={p:.2f}"))
    ymax = df[col].max()
    ax.text(0.5, 0.97, sig, transform=ax.transAxes,
            ha="center", va="top", fontsize=12, color="black")

axes[1][3].set_visible(False)
fig.tight_layout()
plt.savefig(f"{FIG_DIR}/03_boxplots_by_outcome.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/03_boxplots_by_outcome.png")

# ── 5. Correlation heatmap ─────────────────────────────────────────────────
print("\n5. Plotting correlation heatmap...")

corr_cols = CONTINUOUS + BINARY + [OUTCOME]
corr = df[corr_cols].rename(columns=LABEL_MAP | {"DEATH_EVENT": "Death Event"}).corr()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.5,
    annot_kws={"size": 8}, ax=ax,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
plt.xticks(rotation=40, ha="right", fontsize=9)
plt.yticks(fontsize=9)
fig.tight_layout()
plt.savefig(f"{FIG_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {FIG_DIR}/04_correlation_heatmap.png")

print("\n✓ EDA complete. All outputs saved.")
