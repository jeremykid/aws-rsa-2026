"""
Plot readmission rate by age group from the diabetic readmission dataset.
Saves output to charts/readmission_rate_by_age_group.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_PATH = "diabetic/data/diabetic_readmission.csv"
OUT_PATH  = "charts/readmission_rate_by_age_group.png"

# --- Load data ---
df = pd.read_csv(DATA_PATH)

# --- Map numeric decade midpoints to human-readable brackets ---
age_labels = {
    5:  "[0-10)",
    15: "[10-20)",
    25: "[20-30)",
    35: "[30-40)",
    45: "[40-50)",
    55: "[50-60)",
    65: "[60-70)",
    75: "[70-80)",
    85: "[80-90)",
    95: "[90-100)",
}
df["age_group"] = df["age"].map(age_labels)

# --- Define readmitted as any value other than 'no' ---
df["is_readmitted"] = df["readmitted"].isin(["<30", ">30"])

# --- Compute rate per age group (keep natural order via age midpoint) ---
summary = (
    df.groupby(["age", "age_group"])["is_readmitted"]
    .agg(readmit_rate="mean", count="count")
    .reset_index()
    .sort_values("age")
)
summary["readmit_pct"] = summary["readmit_rate"] * 100

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    summary["age_group"],
    summary["readmit_pct"],
    color="steelblue",
    edgecolor="white",
    linewidth=0.6,
)

# Annotate each bar with its percentage
for bar, pct in zip(bars, summary["readmit_pct"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xlabel("Age Group", fontsize=12)
ax.set_ylabel("Readmission Rate (%)", fontsize=12)
ax.set_title("Hospital Readmission Rate by Age Group", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(0, summary["readmit_pct"].max() * 1.15)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()

fig.savefig(OUT_PATH, dpi=150)
print(f"Chart saved to {OUT_PATH}")
