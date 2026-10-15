"""
demo/app.py
Heart Failure Individual Survival Prediction — Streamlit Demo

Run:
    cd /workshop/heart-failure-survival
    streamlit run demo/app.py --server.port 8081
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Survival Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
FEATURES = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium",
    "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking",
]

LABEL_MAP = {
    "age":                       "Age (years)",
    "creatinine_phosphokinase":  "CPK — Creatinine Phosphokinase (mcg/L)",
    "ejection_fraction":         "Ejection Fraction (%)",
    "platelets":                 "Platelets (kiloplatelets/mL)",
    "serum_creatinine":          "Serum Creatinine (mg/dL)",
    "serum_sodium":              "Serum Sodium (mEq/L)",
    "anaemia":                   "Anaemia",
    "diabetes":                  "Diabetes",
    "high_blood_pressure":       "High Blood Pressure",
    "sex":                       "Sex",
    "smoking":                   "Smoking",
}

SHORT_LABEL = {
    "age":                       "Age",
    "creatinine_phosphokinase":  "CPK",
    "ejection_fraction":         "EF%",
    "platelets":                 "Platelets",
    "serum_creatinine":          "SCr",
    "serum_sodium":              "Na⁺",
    "anaemia":                   "Anaemia",
    "diabetes":                  "Diabetes",
    "high_blood_pressure":       "HBP",
    "sex":                       "Sex",
    "smoking":                   "Smoking",
}

MODEL_DIR  = "outputs/models"
DATA_PATH  = "data/processed/full_scaled.csv"
SCALER_PATH = "data/processed/scaler_full_params.csv"

# ── Load model & background data (cached) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    with open(f"{MODEL_DIR}/rsf_final.pkl", "rb") as f:
        bundle = pickle.load(f)
    rsf = bundle["model"]

    full_df = pd.read_csv(DATA_PATH)
    X_full  = full_df[FEATURES].values

    scaler_df = pd.read_csv(SCALER_PATH)
    scaler    = {row["feature"]: (row["mean"], row["std"])
                 for _, row in scaler_df.iterrows()}

    # Pre-compute SHAP background (50 samples for speed)
    background = shap.sample(X_full, 50, random_state=42)
    explainer  = shap.KernelExplainer(rsf.predict, background, silent=True)

    # Pre-compute population survival curves for reference band
    step_fns    = rsf.predict_survival_function(X_full)
    times_ref   = step_fns[0].x
    surv_matrix = np.row_stack([fn(times_ref) for fn in step_fns])

    return rsf, scaler, explainer, X_full, times_ref, surv_matrix

rsf, scaler, explainer, X_full, times_ref, surv_matrix = load_resources()

# ── Helper: scale a raw input dict → numpy array ──────────────────────────
def scale_input(raw: dict) -> np.ndarray:
    vec = []
    for feat in FEATURES:
        val = raw[feat]
        if feat in scaler:
            mean, std = scaler[feat]
            val = (val - mean) / (std + 1e-10)
        vec.append(val)
    return np.array(vec).reshape(1, -1)

# ── Helper: risk tier ──────────────────────────────────────────────────────
def risk_tier(risk_score: float, all_scores: np.ndarray):
    pct = (all_scores < risk_score).mean() * 100
    if pct >= 75:
        return "🔴 High Risk",   "#E8604C", pct
    elif pct >= 40:
        return "🟡 Medium Risk", "#F0A500", pct
    else:
        return "🟢 Low Risk",    "#2ECC71", pct

# ══════════════════════════════════════════════════════════════════════════
# Sidebar — Patient Input Form
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🩺 Patient Parameters")
    st.caption("Adjust values to match the patient's clinical profile.")
    st.divider()

    st.subheader("📊 Lab Results")
    age   = st.slider("Age (years)",                   40, 95, 60)
    ef    = st.slider("Ejection Fraction (%)",          14, 80, 38,
                      help="Normal: 55–70%")
    scr   = st.slider("Serum Creatinine (mg/dL)",      0.5, 9.4, 1.1, step=0.1,
                      help="Normal: 0.7–1.2 mg/dL")
    na    = st.slider("Serum Sodium (mEq/L)",          113, 148, 137,
                      help="Normal: 135–145 mEq/L")
    cpk   = st.number_input("CPK (mcg/L)",             min_value=23, max_value=7861,
                            value=250, step=10,
                            help="Creatinine phosphokinase. Normal: <200 mcg/L")
    plt_  = st.number_input("Platelets (kiloplatelets/mL)", min_value=25100,
                            max_value=850000, value=262000, step=1000)

    st.subheader("🏥 Comorbidities")
    anaemia = st.toggle("Anaemia",              value=False)
    dm      = st.toggle("Diabetes",             value=False)
    hbp     = st.toggle("High Blood Pressure",  value=False)

    st.subheader("👤 Demographics")
    sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True)
    smoking   = st.toggle("Smoking",            value=False)

    st.divider()
    predict_btn = st.button("🔮 Predict Survival", type="primary",
                            use_container_width=True)

# ── Build raw input ────────────────────────────────────────────────────────
raw_input = {
    "age":                       age,
    "creatinine_phosphokinase":  cpk,
    "ejection_fraction":         ef,
    "platelets":                 plt_,
    "serum_creatinine":          scr,
    "serum_sodium":              na,
    "anaemia":                   int(anaemia),
    "diabetes":                  int(dm),
    "high_blood_pressure":       int(hbp),
    "sex":                       1 if sex_label == "Male" else 0,
    "smoking":                   int(smoking),
}

# ══════════════════════════════════════════════════════════════════════════
# Main panel
# ══════════════════════════════════════════════════════════════════════════
st.title("🫀 Heart Failure Survival Predictor")
st.markdown(
    "Individual survival curve prediction using a **Random Survival Forest** "
    "trained on the [UCI Heart Failure dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) "
    "(N=299). Adjust patient parameters in the sidebar and click **Predict Survival**."
)
st.divider()

if not predict_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    col1.metric("Training samples", "299")
    col2.metric("Model", "Random Survival Forest")
    col3.metric("Test C-index", "0.753")
    st.info("👈 Set patient parameters in the sidebar and click **Predict Survival** to generate an individual survival curve.")
    st.stop()

# ── Run prediction ─────────────────────────────────────────────────────────
with st.spinner("Computing survival curve…"):
    X_scaled = scale_input(raw_input)

    # Survival curve
    surv_fn   = rsf.predict_survival_function(X_scaled)[0]
    t_patient = surv_fn.x
    s_patient = surv_fn(t_patient)

    # Risk score (cumulative hazard)
    risk_score    = rsf.predict(X_scaled)[0]
    all_risks     = rsf.predict(X_full)
    tier, t_color, percentile = risk_tier(risk_score, all_risks)

    # SHAP values (silent=True suppresses tqdm stdout which breaks in Streamlit)
    shap_vals = explainer.shap_values(X_scaled, nsamples=150, silent=True)[0]

    # 6-month / 1-year survival probabilities
    def survival_at(t_target, t_arr, s_arr):
        idx = np.searchsorted(t_arr, t_target)
        if idx == 0: return s_arr[0]
        if idx >= len(t_arr): return s_arr[-1]
        return s_arr[idx - 1]

    s6m  = survival_at(180, t_patient, s_patient)
    s1y  = survival_at(285, t_patient, s_patient)

# ══════════════════════════════════════════════════════════════════════════
# Top metrics row
# ══════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Tier",              tier)
c2.metric("Risk Percentile",        f"{percentile:.0f}th")
c3.metric("6-Month Survival",       f"{s6m*100:.1f}%")
c4.metric("1-Year Survival",        f"{s1y*100:.1f}%")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# Row 1: Survival Curve + SHAP waterfall
# ══════════════════════════════════════════════════════════════════════════
col_surv, col_shap = st.columns([3, 2])

# ── Survival curve ─────────────────────────────────────────────────────────
with col_surv:
    st.subheader("Individual Survival Curve")

    # Population percentile band (10th–90th)
    p10 = np.percentile(surv_matrix, 10, axis=0)
    p90 = np.percentile(surv_matrix, 90, axis=0)
    p50 = np.percentile(surv_matrix, 50, axis=0)

    fig = go.Figure()

    # Population band
    fig.add_trace(go.Scatter(
        x=np.concatenate([times_ref, times_ref[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill="toself", fillcolor="rgba(150,150,150,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Population 10–90th %ile", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=times_ref, y=p50,
        mode="lines", line=dict(color="#AAAAAA", width=1.5, dash="dot"),
        name="Population median",
    ))

    # Patient curve
    fig.add_trace(go.Scatter(
        x=t_patient, y=s_patient,
        mode="lines", line=dict(color=t_color, width=3),
        name="This patient",
        hovertemplate="Day %{x:.0f}<br>S(t) = %{y:.3f}<extra></extra>",
    ))

    # Annotation markers
    for t_mark, s_mark, label in [(180, s6m, "6 mo"), (285, s1y, "1 yr")]:
        fig.add_trace(go.Scatter(
            x=[t_mark], y=[s_mark],
            mode="markers+text",
            marker=dict(color=t_color, size=10, symbol="circle"),
            text=[f"  {label}: {s_mark*100:.1f}%"],
            textposition="middle right",
            showlegend=False,
            hovertemplate=f"{label}: S(t)=%{{y:.3f}}<extra></extra>",
        ))
        fig.add_vline(x=t_mark, line_dash="dash",
                      line_color="rgba(100,100,100,0.3)", line_width=1)

    fig.update_layout(
        xaxis_title="Follow-up Time (days)",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        xaxis=dict(range=[0, 290]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")

    st.plotly_chart(fig, use_container_width=True)

# ── SHAP feature contributions ─────────────────────────────────────────────
with col_shap:
    st.subheader("Feature Contributions")
    st.caption("How each feature pushes the risk score above/below average.")

    feat_labels = [SHORT_LABEL[f] for f in FEATURES]
    sorted_idx  = np.argsort(np.abs(shap_vals))[::-1]

    shap_sorted  = shap_vals[sorted_idx]
    label_sorted = [feat_labels[i] for i in sorted_idx]

    bar_colors = ["#E8604C" if v > 0 else "#4C9BE8" for v in shap_sorted]

    fig2 = go.Figure(go.Bar(
        x=shap_sorted,
        y=label_sorted,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig2.add_vline(x=0, line_color="black", line_width=1)
    fig2.update_layout(
        xaxis_title="SHAP value (impact on risk score)",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=20, t=30, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=[
            dict(x=max(shap_sorted)*0.85, y=0.02, xref="x", yref="paper",
                 text="↑ Higher risk", showarrow=False,
                 font=dict(color="#E8604C", size=11)),
            dict(x=min(shap_sorted)*0.85, y=0.02, xref="x", yref="paper",
                 text="↓ Lower risk", showarrow=False,
                 font=dict(color="#4C9BE8", size=11)),
        ]
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#F0F0F0")

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# Row 2: Risk gauge + patient summary table
# ══════════════════════════════════════════════════════════════════════════
col_gauge, col_table = st.columns([1, 2])

with col_gauge:
    st.subheader("Risk Score")
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(percentile, 1),
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#E8604C"},
               "decreasing": {"color": "#2ECC71"}},
        title={"text": "Risk Percentile<br><span style='font-size:0.8em;color:gray'>vs. cohort</span>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": t_color},
            "steps": [
                {"range": [0,  40], "color": "#E8F8EF"},
                {"range": [40, 75], "color": "#FFF8E1"},
                {"range": [75, 100],"color": "#FDE8E8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": percentile,
            },
        },
        number={"suffix": "th %ile", "valueformat": ".1f"},
    ))
    fig3.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

with col_table:
    st.subheader("Patient Summary")
    display_rows = []
    for feat in FEATURES:
        raw_val = raw_input[feat]
        if feat == "sex":
            display_val = "Male" if raw_val == 1 else "Female"
        elif feat in ("anaemia", "diabetes", "high_blood_pressure", "smoking"):
            display_val = "Yes" if raw_val == 1 else "No"
        else:
            display_val = raw_val
        shap_v = shap_vals[FEATURES.index(feat)]
        direction = "↑ risk" if shap_v > 0 else "↓ risk"
        display_rows.append({
            "Feature":         LABEL_MAP[feat],
            "Value":           display_val,
            "SHAP":            f"{shap_v:+.4f}",
            "Direction":       direction,
        })

    summary_df = pd.DataFrame(display_rows)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=280,
        column_config={
            "Direction": st.column_config.TextColumn("Direction"),
            "SHAP": st.column_config.TextColumn("SHAP"),
        },
    )

st.divider()
st.caption(
    "⚠️ **Disclaimer**: This tool is for research and educational purposes only. "
    "It is not intended for clinical decision-making. "
    "Model trained on UCI Heart Failure dataset (N=299). "
    "Test C-index: 0.753 | IBS: 0.156."
)
