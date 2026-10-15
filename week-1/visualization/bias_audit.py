"""
Bias audit for diabetic_readmission.csv
Steps: Representation, Outcomes, Missing Data, Charts
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "diabetic/data/diabetic_readmission.csv"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

DEMO_COLS  = ["race", "gender", "age"]
OUTCOME    = "readmitted"
SMALL_N    = 1000          # flag groups with fewer records than this
DEVIATION  = 5.0           # flag outcome rates that deviate by more than 5 pp

# Clinical measurement columns to check for missingness
CLINICAL_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "max_glu_serum",
    "a1c_result", "change", "diabetes_med",
]

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Map numeric age midpoints → readable brackets
AGE_LABELS = {
    5: "[0-10)", 15: "[10-20)", 25: "[20-30)", 35: "[30-40)",
    45: "[40-50)", 55: "[50-60)", 65: "[60-70)", 75: "[70-80)",
    85: "[80-90)", 95: "[90-100)",
}
df["age_group"] = df["age"].map(AGE_LABELS)

# Binary outcome: any readmission vs none
df["is_readmitted"] = df[OUTCOME].isin(["<30", ">30"])

# For missing-data analysis, treat '?' as NaN in object columns
df_miss = df.copy()
for col in df_miss.select_dtypes("object").columns:
    df_miss[col] = df_miss[col].replace("?", np.nan)

overall_rate = df["is_readmitted"].mean() * 100

print(f"Dataset: {len(df):,} rows | Overall readmission rate: {overall_rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════
rep_results = {}   # col → DataFrame(group, count, pct, flagged)

for col in ["race", "gender", "age_group"]:
    src_col = col
    counts = df[src_col].value_counts(dropna=False).rename_axis("group").reset_index(name="count")
    counts["pct"]     = counts["count"] / counts["count"].sum() * 100
    counts["flagged"] = counts["count"] < SMALL_N
    rep_results[col]  = counts
    flagged = counts.loc[counts["flagged"], "group"].tolist()
    print(f"\n[Representation] {col}")
    print(counts.to_string(index=False))
    if flagged:
        print(f"  ⚠ Small groups (< {SMALL_N}): {flagged}")

# ── Chart 1: Representation ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Group Representation in Dataset", fontsize=14, fontweight="bold")

for ax, (col, data) in zip(axes, rep_results.items()):
    colors = ["#d62728" if f else "#1f77b4" for f in data["flagged"]]
    bars = ax.barh(data["group"], data["pct"], color=colors)
    for bar, pct in zip(bars, data["pct"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Share of dataset (%)")
    ax.set_title(col.replace("_", " ").title())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

axes[0].annotate("Red = flagged (< 1 000 records)", xy=(0, -0.18),
                 xycoords="axes fraction", fontsize=8, color="#d62728")
plt.tight_layout()
chart1 = os.path.join(CHARTS_DIR, "bias_01_representation.png")
fig.savefig(chart1, dpi=150)
plt.close(fig)
print(f"\nSaved {chart1}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – OUTCOMES
# ═══════════════════════════════════════════════════════════════════════════════
outcome_results = {}   # col → DataFrame

for col in ["race", "gender", "age_group"]:
    grp = df.groupby(col)["is_readmitted"].agg(rate="mean", count="count").reset_index()
    grp.columns = ["group", "rate", "count"]
    grp["rate_pct"]  = grp["rate"] * 100
    grp["deviation"] = grp["rate_pct"] - overall_rate
    grp["flagged"]   = grp["deviation"].abs() > DEVIATION
    outcome_results[col] = grp
    flagged = grp.loc[grp["flagged"], "group"].tolist()
    print(f"\n[Outcomes] {col} | overall = {overall_rate:.1f}%")
    print(grp[["group", "count", "rate_pct", "deviation", "flagged"]].to_string(index=False))
    if flagged:
        print(f"  ⚠ Flagged (|deviation| > {DEVIATION} pp): {flagged}")

# ── Chart 2: Outcomes ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Readmission Rate by Group  (overall = {overall_rate:.1f}%)",
             fontsize=14, fontweight="bold")

for ax, (col, data) in zip(axes, outcome_results.items()):
    # Sort age_group naturally
    if col == "age_group":
        order = [AGE_LABELS[k] for k in sorted(AGE_LABELS)]
        data = data.set_index("group").reindex(order).reset_index()

    colors = ["#d62728" if f else "#1f77b4" for f in data["flagged"]]
    bars = ax.barh(data["group"], data["rate_pct"], color=colors)
    ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1, label=f"Overall {overall_rate:.1f}%")
    for bar, pct in zip(bars, data["rate_pct"]):
        if pd.notna(pct):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Readmission rate (%)")
    ax.set_title(col.replace("_", " ").title())
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8)

axes[0].annotate("Red = flagged (deviation > 5 pp)", xy=(0, -0.18),
                 xycoords="axes fraction", fontsize=8, color="#d62728")
plt.tight_layout()
chart2 = os.path.join(CHARTS_DIR, "bias_02_outcomes.png")
fig.savefig(chart2, dpi=150)
plt.close(fig)
print(f"\nSaved {chart2}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – MISSING DATA
# ═══════════════════════════════════════════════════════════════════════════════
# Identify columns with any missingness (NaN or '?')
miss_cols = [c for c in df_miss.columns if df_miss[c].isnull().any()]
print(f"\n[Missing data] Columns with missing values: {miss_cols}")

# Build per-group missingness rates for the two most informative clinical cols
# (max_glu_serum and a1c_result are coded categorically and likely most sparse)
miss_results = {}  # demo_col → DataFrame(group, col, miss_pct)

for demo in ["race", "gender", "age_group"]:
    rows = []
    for mcol in miss_cols:
        grp = df_miss.groupby(demo)[mcol].apply(lambda s: s.isnull().mean() * 100).reset_index()
        grp.columns = ["group", "miss_pct"]
        grp["measure"] = mcol
        rows.append(grp)
    if rows:
        miss_results[demo] = pd.concat(rows, ignore_index=True)

# ── Chart 3: Missing data heatmap per demo group ──────────────────────────────
if miss_results:
    fig, axes = plt.subplots(1, len(miss_results), figsize=(6 * len(miss_results), 5))
    if len(miss_results) == 1:
        axes = [axes]
    fig.suptitle("Missing Data Rate by Group (%)", fontsize=14, fontweight="bold")

    for ax, (demo, data) in zip(axes, miss_results.items()):
        pivot = data.pivot(index="measure", columns="group", values="miss_pct")
        # Sort age columns naturally if present
        if demo == "age_group":
            order = [AGE_LABELS[k] for k in sorted(AGE_LABELS) if AGE_LABELS[k] in pivot.columns]
            pivot = pivot[order]

        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(demo.replace("_", " ").title())
        fig.colorbar(im, ax=ax, label="% missing")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=7, color="black" if val < 60 else "white")

    plt.tight_layout()
    chart3 = os.path.join(CHARTS_DIR, "bias_03_missing_data.png")
    fig.savefig(chart3, dpi=150)
    plt.close(fig)
    print(f"Saved {chart3}")
else:
    chart3 = None
    print("No missing data found — skipping chart 3.")

# ═══════════════════════════════════════════════════════════════════════════════
# COLLECT FINDINGS FOR REPORT
# ═══════════════════════════════════════════════════════════════════════════════

# Representation flags
rep_flags = {}
for col, data in rep_results.items():
    flagged = data.loc[data["flagged"], ["group", "count", "pct"]].copy()
    rep_flags[col] = flagged

# Outcome flags
outcome_flags = {}
for col, data in outcome_results.items():
    flagged = data.loc[data["flagged"], ["group", "count", "rate_pct", "deviation"]].copy()
    outcome_flags[col] = flagged

print("\n=== AUDIT COMPLETE ===")
print(f"Charts saved: {chart1}, {chart2}, {chart3}")
