"""
Visualize diabetic_transformed.csv following the visualize-dataset skill.

Charts produced
  01 — Target distribution (pie + bar)
  02 — Readmission by race (reconstructed from one-hot columns)
  03 — Readmission by age group
  04 — Readmission by medication & change flags
  05 — Time-in-hospital histogram by class
  06 — Lab procedures vs medications scatter
  07 — Correlation heatmap (numeric features)
  08 — Box plots of key features by class
  09 — Binary flags (diabetes_med, change)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "diabetic/data/diabetic_transformed.csv"
CHARTS_DIR = "charts"
TARGET_COL = "readmitted"
TARGET_ORDER  = ["<30", ">30", "no"]
TARGET_LABELS = {"<30": "Readmit < 30 days", ">30": "Readmit > 30 days", "no": "No Readmit"}
PALETTE = ["#DD8452", "#4C72B0", "#55A868"]

os.makedirs(CHARTS_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df[TARGET_COL] = pd.Categorical(df[TARGET_COL], categories=TARGET_ORDER, ordered=True)

# Age brackets for display
AGE_LABELS = {
    5: "[0-10)", 15: "[10-20)", 25: "[20-30)", 35: "[30-40)",
    45: "[40-50)", 55: "[50-60)", 65: "[60-70)", 75: "[70-80)",
    85: "[80-90)", 95: "[90-100)",
}
df["age_group"] = df["age"].map(AGE_LABELS)

# Reconstruct race from one-hot columns
race_cols = ["race_caucasian", "race_african_american", "race_hispanic", "race_other", "race_asian"]
race_map  = {
    "race_caucasian": "Caucasian",
    "race_african_american": "African American",
    "race_hispanic": "Hispanic",
    "race_other": "Other",
    "race_asian": "Asian",
}
df["race"] = df[race_cols].idxmax(axis=1).map(race_map)

print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Chart 01 — Target distribution ───────────────────────────────────────────
counts = df[TARGET_COL].value_counts().reindex(TARGET_ORDER)
labels = [TARGET_LABELS[k] for k in TARGET_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Readmission Distribution (diabetic_transformed.csv)",
             fontsize=14, fontweight="bold")

axes[0].pie(counts, labels=labels, colors=PALETTE, autopct="%1.1f%%",
            startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0].set_title("Proportion")

bars = axes[1].bar(labels, counts, color=PALETTE, edgecolor="white")
axes[1].bar_label(bars, fmt="{:,.0f}", padding=4)
axes[1].set_ylabel("Count")
axes[1].set_title("Count")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

save(fig, "t01_target_distribution.png")


# ── Chart 02 — Readmission by race ───────────────────────────────────────────
race_order = df["race"].value_counts().index.tolist()
grp = (df.groupby(["race", TARGET_COL], observed=True)
         .size().unstack(TARGET_COL).reindex(race_order).fillna(0))
pct = grp.div(grp.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
pct[TARGET_ORDER].plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75)
ax.set_title("Readmission Rate by Race", fontsize=13, fontweight="bold")
ax.set_xlabel("Race")
ax.set_ylabel("Percentage (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend([TARGET_LABELS[k] for k in TARGET_ORDER], title=TARGET_COL)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save(fig, "t02_by_race.png")


# ── Chart 03 — Readmission by age group ──────────────────────────────────────
age_order = [AGE_LABELS[k] for k in sorted(AGE_LABELS)]
grp = (df.groupby(["age_group", TARGET_COL], observed=True)
         .size().unstack(TARGET_COL).reindex(age_order).fillna(0))
pct = grp.div(grp.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
pct[TARGET_ORDER].plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75)
ax.set_title("Readmission Rate by Age Group", fontsize=13, fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("Percentage (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend([TARGET_LABELS[k] for k in TARGET_ORDER], title=TARGET_COL)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save(fig, "t03_by_age.png")


# ── Chart 04 — Readmission by diabetes_med × change ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Readmission Rate by Medication Flags", fontsize=13, fontweight="bold")

for ax, (col, col_label) in zip(axes, [("diabetes_med", "On Diabetes Medication"),
                                        ("change", "Medication Changed")]):
    grp = (df.groupby([col, TARGET_COL], observed=True)
             .size().unstack(TARGET_COL).fillna(0))
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    pct[TARGET_ORDER].plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.6)
    ax.set_title(col_label)
    ax.set_xlabel("0 = No, 1 = Yes")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    ax.legend([TARGET_LABELS[k] for k in TARGET_ORDER], title=TARGET_COL, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

save(fig, "t04_medication_flags.png")


# ── Chart 05 — Time-in-hospital histogram by class ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for key, color in zip(TARGET_ORDER, PALETTE):
    ax.hist(df[df[TARGET_COL] == key]["time_in_hospital"],
            bins=14, alpha=0.6, color=color,
            label=TARGET_LABELS[key], edgecolor="white")
ax.set_title("Time in Hospital by Readmission Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Days in Hospital")
ax.set_ylabel("Count")
ax.legend(title=TARGET_COL)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
save(fig, "t05_time_in_hospital.png")


# ── Chart 06 — Lab procedures vs medications scatter ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for key, color in zip(TARGET_ORDER, PALETTE):
    sub = df[df[TARGET_COL] == key]
    ax.scatter(sub["num_lab_procedures"], sub["num_medications"],
               alpha=0.15, s=6, color=color,
               label=TARGET_LABELS[key], rasterized=True)
ax.set_title("Lab Procedures vs Medications", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Lab Procedures")
ax.set_ylabel("Number of Medications")
ax.legend(title=TARGET_COL, markerscale=4)
save(fig, "t06_scatter_lab_vs_meds.png")


# ── Chart 07 — Correlation heatmap ───────────────────────────────────────────
# Exclude one-hot race columns (redundant with reconstructed 'race') and age_group
numeric_cols = [
    "age", "time_in_hospital", "num_lab_procedures", "num_medications",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "change", "diabetes_med",
]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels([c.replace("_", "\n") for c in numeric_cols], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([c.replace("_", " ") for c in numeric_cols], fontsize=8)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
save(fig, "t07_correlation_heatmap.png")


# ── Chart 08 — Box plots by class ────────────────────────────────────────────
box_features = [
    ("num_lab_procedures", "Lab Procedures"),
    ("num_medications",    "Medications"),
    ("number_inpatient",   "Inpatient Visits"),
    ("number_diagnoses",   "Diagnoses"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Key Features by Readmission Class", fontsize=14, fontweight="bold")

for ax, (col, title) in zip(axes.flatten(), box_features):
    data = [df[df[TARGET_COL] == k][col].dropna() for k in TARGET_ORDER]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_xticks(range(1, len(TARGET_ORDER) + 1))
    ax.set_xticklabels([TARGET_LABELS[k] for k in TARGET_ORDER],
                       rotation=15, ha="right")
    ax.set_ylabel("Value")

save(fig, "t08_boxplots.png")


# ── Chart 09 — Race representation (one-hot column sums) ─────────────────────
race_counts = df[race_cols].sum().rename(race_map).sort_values(ascending=False)
race_pct    = race_counts / race_counts.sum() * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(race_pct.index, race_pct.values, color="#4C72B0", edgecolor="white")
for bar, pct in zip(bars, race_pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_title("Race Group Representation", fontsize=13, fontweight="bold")
ax.set_ylabel("Share of Dataset (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticklabels(race_pct.index, rotation=20, ha="right")
save(fig, "t09_race_representation.png")

print("\nAll charts saved.")
