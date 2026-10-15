---
name: visualize-dataset
description: >
  Use this skill whenever a user provides a CSV file and asks to explore,
  understand, or visualize the data. It covers the full workflow from
  inspecting raw structure through generating a suite of publication-ready
  charts saved as numbered PNGs. Works for any tabular CSV with a mix of
  numeric and categorical columns and a single target/label column.
---

# Skill: Visualize a CSV Dataset

## Goal
Produce a comprehensive set of exploratory visualizations for a CSV dataset
and save every chart as a numbered PNG inside a `charts/` folder.

---

## Steps

### 1. Locate and unzip the data (if needed)

If the dataset is compressed, unzip it first.

```bash
# Find any zip files in the working directory
ls *.zip

# Unzip
unzip <filename>.zip
```

Then confirm the extracted contents:

```bash
ls -lh <extracted_folder>/
```

---

### 2. Inspect the raw CSV structure

Before writing any code, understand what you are working with.

```bash
# Column names and first few rows
head -5 <path/to/data.csv>

# Row count (subtract 1 for the header)
wc -l <path/to/data.csv>
```

Key questions to answer:
- What is the **target / label column**? (What are we trying to predict?)
- Which columns are **numeric** vs **categorical**?
- Are there **binary flag** columns (0/1)?
- Are there any **derived time or group columns** (e.g. age buckets)?
- Is there a separate **inference results** file with predicted probabilities?

---

### 3. Set up the Python script

Create `visualize.py` in the working directory. Install dependencies if needed:

```bash
pip3 install numpy pandas matplotlib --quiet
```

Start the script with:

```python
import os, ast, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH   = "<path/to/data.csv>"
CHARTS_DIR = "charts"
TARGET_COL = "<target_column_name>"          # e.g. "readmitted", "label", "churn"
TARGET_ORDER  = ["<class_a>", "<class_b>"]   # desired display order
TARGET_LABELS = {"<class_a>": "Label A", …}  # human-readable names
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]  # one colour per class

os.makedirs(CHARTS_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df[TARGET_COL] = pd.Categorical(df[TARGET_COL],
                                categories=TARGET_ORDER, ordered=True)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
```

---

### 4. Chart 01 — Target variable distribution (pie + bar)

Shows class balance at a glance. Always generate this first.

```python
counts = df[TARGET_COL].value_counts().reindex(TARGET_ORDER)
labels = [TARGET_LABELS[k] for k in TARGET_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"{TARGET_COL} Distribution", fontsize=15, fontweight="bold")

axes[0].pie(counts, labels=labels, colors=PALETTE, autopct="%1.1f%%",
            startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0].set_title("Proportion")

bars = axes[1].bar(labels, counts, color=PALETTE, edgecolor="white")
axes[1].bar_label(bars, fmt="{:,.0f}", padding=4)
axes[1].set_ylabel("Count")
axes[1].set_title("Count")
axes[1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

save(fig, "01_target_distribution.png")
```

---

### 5. Charts 02–04 — Target breakdown by categorical features

For each categorical column of interest (e.g. race, gender, age group),
produce a grouped % bar chart.

```python
def cat_breakdown(col, title, filename, rotation=30):
    order = df[col].value_counts().index.tolist()
    grp = (df.groupby([col, TARGET_COL], observed=True)
             .size().unstack(TARGET_COL)
             .reindex(order).fillna(0))
    pct = grp.div(grp.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    pct[TARGET_ORDER].plot(kind="bar", ax=ax, color=PALETTE,
                           edgecolor="white", width=0.75)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right")
    ax.legend([TARGET_LABELS[k] for k in TARGET_ORDER], title=TARGET_COL)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    save(fig, filename)

cat_breakdown("race",   "Target Rate by Race",   "02_by_race.png")
cat_breakdown("gender", "Target Rate by Gender", "03_by_gender.png", rotation=0)
cat_breakdown("age_group", "Target Rate by Age", "04_by_age.png", rotation=45)
```

> **Note:** If your CSV has different categorical columns, substitute the
> column names accordingly. Create one chart per meaningful categorical.

---

### 6. Chart 05 — Numeric feature histograms (overlaid by target class)

Useful for any continuous column (e.g. days in hospital, test counts).

```python
NUM_COL = "time_in_hospital"   # replace with your column

fig, ax = plt.subplots(figsize=(10, 5))
for key, color in zip(TARGET_ORDER, PALETTE):
    ax.hist(df[df[TARGET_COL] == key][NUM_COL],
            bins=20, alpha=0.6, color=color,
            label=TARGET_LABELS[key], edgecolor="white")
ax.set_title(f"{NUM_COL} by {TARGET_COL}", fontsize=13, fontweight="bold")
ax.set_xlabel(NUM_COL.replace("_", " ").title())
ax.set_ylabel("Count")
ax.legend(title=TARGET_COL)
save(fig, "05_numeric_histogram.png")
```

Repeat for other numeric columns of interest, incrementing the chart number.

---

### 7. Chart 06 — Scatter plot of two numeric features

Reveals clusters or outliers across two dimensions.

```python
X_COL, Y_COL = "num_lab_procedures", "num_medications"  # replace as needed

fig, ax = plt.subplots(figsize=(9, 6))
for key, color in zip(TARGET_ORDER, PALETTE):
    sub = df[df[TARGET_COL] == key]
    ax.scatter(sub[X_COL], sub[Y_COL],
               alpha=0.25, s=8, color=color,
               label=TARGET_LABELS[key], rasterized=True)
ax.set_title(f"{X_COL} vs {Y_COL}", fontsize=13, fontweight="bold")
ax.set_xlabel(X_COL.replace("_", " ").title())
ax.set_ylabel(Y_COL.replace("_", " ").title())
ax.legend(title=TARGET_COL, markerscale=3)
save(fig, "06_scatter.png")
```

---

### 8. Chart 07 — Correlation heatmap of numeric features

Shows linear relationships between all numeric columns.

```python
numeric_cols = df.select_dtypes(include="number").columns.tolist()
# Remove the target if it was encoded numerically
# numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(max(7, len(numeric_cols)),
                                max(6, len(numeric_cols) - 1)))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(numeric_cols, fontsize=8)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
save(fig, "07_correlation_heatmap.png")
```

---

### 9. Chart 08 — Box plots of key features by target class

Highlights median and spread differences across classes.

```python
# List up to 4 numeric columns most relevant to the target
box_features = [
    ("num_lab_procedures", "Lab Procedures"),
    ("num_medications",    "Medications"),
    ("number_inpatient",   "Inpatient Visits"),
    ("number_diagnoses",   "Diagnoses"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(f"Key Features by {TARGET_COL}", fontsize=14, fontweight="bold")

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
save(fig, "08_boxplots.png")
```

---

### 10. Chart 09 — Binary flag breakdown (optional)

If the dataset has 0/1 flag columns, show how each flag correlates with the target.

```python
binary_flags = [
    ("diabetes_med", "On Medication"),
    ("change",       "Medication Changed"),
]

fig, axes = plt.subplots(1, len(binary_flags),
                         figsize=(6 * len(binary_flags), 5))
fig.suptitle("Binary Flags by Target", fontsize=14, fontweight="bold")

for ax, (col, col_label) in zip(axes, binary_flags):
    grp = (df.groupby([col, TARGET_COL], observed=True)
             .size().unstack(TARGET_COL).fillna(0))
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    pct[TARGET_ORDER].plot(kind="bar", ax=ax, color=PALETTE,
                           edgecolor="white", width=0.6)
    ax.set_title(col_label)
    ax.set_xlabel("0 = No, 1 = Yes")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend([TARGET_LABELS[k] for k in TARGET_ORDER], title=TARGET_COL)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save(fig, "09_binary_flags.png")
```

---

### 11. Chart 10 — Inference probability histograms (optional)

If a model inference results file exists with predicted class probabilities,
visualise the score distributions per class.

```python
# Expected CSV format: one column of probability lists, one column of class names
# e.g.  "[0.18, 0.33, 0.48]","['<30', '>30', 'no']"

INFERENCE_CSV = "<path/to/inference_results_with_probabilities.csv>"

try:
    inf_df = pd.read_csv(INFERENCE_CSV, header=None,
                         names=["probabilities", "classes"])
    probs = inf_df["probabilities"].apply(ast.literal_eval).tolist()
    prob_df = pd.DataFrame(probs, columns=TARGET_ORDER)

    fig, axes = plt.subplots(1, len(TARGET_ORDER),
                             figsize=(5 * len(TARGET_ORDER), 4))
    fig.suptitle("Predicted Probability Distributions",
                 fontsize=13, fontweight="bold")
    for ax, (col, color) in zip(axes, zip(TARGET_ORDER, PALETTE)):
        ax.hist(prob_df[col], bins=20, color=color,
                edgecolor="white", alpha=0.85)
        ax.set_title(TARGET_LABELS[col])
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
    save(fig, "10_inference_probabilities.png")
except Exception as e:
    print(f"Skipped inference chart: {e}")
```

---

### 12. Run and verify

```bash
python3 visualize.py
```

Expected output — one `Saved →` line per chart. Then confirm:

```bash
ls -lh charts/
```

All charts should be present as `01_…png` through `10_…png` (skipping any
optional steps that were not applicable).

---

## Checklist

- [ ] Raw CSV inspected (`head`, `wc -l`) before writing code
- [ ] Target column identified and class order/labels defined
- [ ] Chart 01 (class balance) generated first to check for imbalance
- [ ] One breakdown chart per meaningful categorical column
- [ ] Histogram for every key numeric feature
- [ ] Correlation heatmap covers all numeric columns
- [ ] Box plots compare distributions across target classes
- [ ] Binary flag charts included if 0/1 columns are present
- [ ] Inference probability charts included if a scores file exists
- [ ] All PNGs saved to `charts/` with sequential numeric prefixes
- [ ] Script runs cleanly with no unhandled exceptions

---

## Notes

- **Class imbalance:** If Chart 01 shows a heavily skewed target, flag this
  to the user — it affects model choice and evaluation metrics.
- **Label quality:** `no` / negative labels in administrative datasets often
  represent censoring, not true absence of the event. Mention this caveat.
- **Categorical explosion:** If a categorical column has > 15 unique values,
  group the tail into an "Other" bucket before plotting.
- **Large files:** For CSVs with > 500k rows, add `rasterized=True` to
  scatter plots and consider sampling for histograms.
