"""
Diabetic Readmission Data Visualization
Generates charts from diabetic_readmission.csv and saves them to charts/
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "diabetic/data"
CHARTS_DIR = "charts"
MAIN_CSV = os.path.join(DATA_DIR, "diabetic_readmission.csv")
TRANSFORMED_CSV = os.path.join(DATA_DIR, "diabetic_transformed.csv")
INFERENCE_PROBS_CSV = os.path.join(DATA_DIR, "inference_results_with_probabilities.csv")

PALETTE = ["#4C72B0", "#DD8452", "#55A868"]   # <30, >30, no
READMIT_ORDER = ["<30", ">30", "no"]
READMIT_LABELS = {"<30": "< 30 days", ">30": "> 30 days", "no": "Not readmitted"}

os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(MAIN_CSV)
df["readmitted"] = pd.Categorical(df["readmitted"], categories=READMIT_ORDER, ordered=True)

# Age is stored as decade mid-points (5, 15, 25, …, 95)
df["age_group"] = df["age"].apply(
    lambda v: f"{v - 4}–{v + 5}" if pd.notna(v) else "Unknown"
)

print(f"Loaded {len(df):,} rows from {MAIN_CSV}")


# ── Helper ───────────────────────────────────────────────────────────────────
def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Readmission distribution (pie + bar side-by-side)
# ════════════════════════════════════════════════════════════════════════════
print("\n[1] Readmission distribution")
counts = df["readmitted"].value_counts().reindex(READMIT_ORDER)
labels = [READMIT_LABELS[k] for k in READMIT_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Readmission Distribution", fontsize=15, fontweight="bold")

axes[0].pie(
    counts,
    labels=labels,
    colors=PALETTE,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[0].set_title("Proportion")

bars = axes[1].bar(labels, counts, color=PALETTE, edgecolor="white")
axes[1].bar_label(bars, fmt="{:,.0f}", padding=4)
axes[1].set_ylabel("Number of patients")
axes[1].set_title("Count")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

save(fig, "01_readmission_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# 2. Readmission by race
# ════════════════════════════════════════════════════════════════════════════
print("[2] Readmission by race")
race_order = df["race"].value_counts().index.tolist()
race_counts = (
    df.groupby(["race", "readmitted"], observed=True)
    .size()
    .unstack("readmitted")
    .reindex(race_order)
    .fillna(0)
)
race_pct = race_counts.div(race_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
race_pct[READMIT_ORDER].plot(
    kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75
)
ax.set_title("Readmission Rate by Race", fontsize=13, fontweight="bold")
ax.set_xlabel("Race")
ax.set_ylabel("Percentage (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend([READMIT_LABELS[k] for k in READMIT_ORDER], title="Readmission")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

save(fig, "02_readmission_by_race.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. Readmission by gender
# ════════════════════════════════════════════════════════════════════════════
print("[3] Readmission by gender")
gender_counts = (
    df.groupby(["gender", "readmitted"], observed=True)
    .size()
    .unstack("readmitted")
    .fillna(0)
)
gender_pct = gender_counts.div(gender_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(7, 5))
gender_pct[READMIT_ORDER].plot(
    kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.6
)
ax.set_title("Readmission Rate by Gender", fontsize=13, fontweight="bold")
ax.set_xlabel("Gender")
ax.set_ylabel("Percentage (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend([READMIT_LABELS[k] for k in READMIT_ORDER], title="Readmission")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

save(fig, "03_readmission_by_gender.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. Readmission by age group
# ════════════════════════════════════════════════════════════════════════════
print("[4] Readmission by age group")
age_order = sorted(df["age_group"].unique(), key=lambda s: int(s.split("–")[0]))
age_counts = (
    df.groupby(["age_group", "readmitted"], observed=True)
    .size()
    .unstack("readmitted")
    .reindex(age_order)
    .fillna(0)
)
age_pct = age_counts.div(age_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
age_pct[READMIT_ORDER].plot(
    kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75
)
ax.set_title("Readmission Rate by Age Group", fontsize=13, fontweight="bold")
ax.set_xlabel("Age group (years)")
ax.set_ylabel("Percentage (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend([READMIT_LABELS[k] for k in READMIT_ORDER], title="Readmission")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

save(fig, "04_readmission_by_age.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. Distribution of time in hospital
# ════════════════════════════════════════════════════════════════════════════
print("[5] Time in hospital distribution")
fig, ax = plt.subplots(figsize=(10, 5))
for label, color, key in zip(labels, PALETTE, READMIT_ORDER):
    subset = df[df["readmitted"] == key]["time_in_hospital"]
    ax.hist(subset, bins=14, alpha=0.6, label=label, color=color, edgecolor="white")

ax.set_title("Time in Hospital by Readmission Outcome", fontsize=13, fontweight="bold")
ax.set_xlabel("Days in hospital")
ax.set_ylabel("Number of patients")
ax.legend(title="Readmission")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

save(fig, "05_time_in_hospital.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. Number of lab procedures vs medications (scatter)
# ════════════════════════════════════════════════════════════════════════════
print("[6] Lab procedures vs medications scatter")
fig, ax = plt.subplots(figsize=(9, 6))
for color, key in zip(PALETTE, READMIT_ORDER):
    sub = df[df["readmitted"] == key]
    ax.scatter(
        sub["num_lab_procedures"],
        sub["num_medications"],
        alpha=0.25,
        s=8,
        color=color,
        label=READMIT_LABELS[key],
        rasterized=True,
    )

ax.set_title("Lab Procedures vs. Medications", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of lab procedures")
ax.set_ylabel("Number of medications")
ax.legend(title="Readmission", markerscale=3)

save(fig, "06_lab_procedures_vs_medications.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. Numeric feature correlation heatmap
# ════════════════════════════════════════════════════════════════════════════
print("[7] Correlation heatmap")
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
short = [c.replace("number_", "n_").replace("num_", "n_") for c in numeric_cols]
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short, fontsize=9)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")

save(fig, "07_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. Box plots: key numeric features by readmission
# ════════════════════════════════════════════════════════════════════════════
print("[8] Box plots of key features by readmission")
box_features = [
    ("num_lab_procedures", "Lab procedures"),
    ("num_medications", "Medications"),
    ("number_inpatient", "Inpatient visits"),
    ("number_diagnoses", "Diagnoses"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Key Features by Readmission Outcome", fontsize=14, fontweight="bold")

for ax, (col, title) in zip(axes.flatten(), box_features):
    data = [df[df["readmitted"] == k][col].dropna() for k in READMIT_ORDER]
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([READMIT_LABELS[k] for k in READMIT_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Count")

save(fig, "08_boxplots_by_readmission.png")


# ════════════════════════════════════════════════════════════════════════════
# 9. Diabetes medication and change flags
# ════════════════════════════════════════════════════════════════════════════
print("[9] Medication & change flags")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Medication Flags by Readmission", fontsize=14, fontweight="bold")

for ax, (col, col_label) in zip(
    axes, [("diabetes_med", "On Diabetes Medication"), ("change", "Medication Changed")]
):
    grp = (
        df.groupby([col, "readmitted"], observed=True)
        .size()
        .unstack("readmitted")
        .fillna(0)
    )
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    pct[READMIT_ORDER].plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.6)
    ax.set_title(col_label)
    ax.set_xlabel("Flag (0 = No, 1 = Yes)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend([READMIT_LABELS[k] for k in READMIT_ORDER], title="Readmission")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

save(fig, "09_medication_flags.png")


# ════════════════════════════════════════════════════════════════════════════
# 10. Inference probability distributions (from probabilities CSV)
# ════════════════════════════════════════════════════════════════════════════
print("[10] Inference probability distributions")
try:
    inf_df = pd.read_csv(INFERENCE_PROBS_CSV, header=None, names=["probabilities", "classes"])
    probs = inf_df["probabilities"].apply(ast.literal_eval).tolist()
    prob_df = pd.DataFrame(probs, columns=["<30", ">30", "no"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle("Predicted Probability Distributions (Inference Sample)",
                 fontsize=13, fontweight="bold")

    for ax, (col, color) in zip(axes, zip(READMIT_ORDER, PALETTE)):
        ax.hist(prob_df[col], bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(READMIT_LABELS[col])
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")

    save(fig, "10_inference_probabilities.png")
except Exception as e:
    print(f"  Skipped (could not parse probabilities): {e}")


# ════════════════════════════════════════════════════════════════════════════
print(f"\nDone! All charts saved to ./{CHARTS_DIR}/")
