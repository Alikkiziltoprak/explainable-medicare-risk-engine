"""
Explainable Medicare Risk Adjustment & Revenue Leakage Analytics
Streamlit Dashboard
-----------------------------------------------------------------
Sections:
  1. Population Overview
  2. Risk Score Distribution
  3. Revenue Leakage Alerts
  4. Member Lookup & SHAP Explanation
  5. Provider Coding Accuracy
  6. Model Governance Summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import sys
from sklearn.preprocessing import LabelEncoder

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare Risk Adjustment Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = "data/synthetic/medicare_members.csv"
MODEL_PATH = "outputs/models/xgboost_raf_model.pkl"
SHAP_DIR   = "outputs/shap"

# ── HCC Metadata ──────────────────────────────────────────────────────────
HCC_META = {
    85:  {"name": "Congestive Heart Failure",    "icd": "I50.xx", "raf": 0.331},
    18:  {"name": "Diabetes w/ Chronic Comp.",   "icd": "E11.6x", "raf": 0.318},
    17:  {"name": "Diabetes w/ Acute Comp.",     "icd": "E11.0x", "raf": 0.318},
    19:  {"name": "Diabetes w/o Complication",   "icd": "E11.9",  "raf": 0.105},
    111: {"name": "COPD",                        "icd": "J44.x",  "raf": 0.328},
    138: {"name": "CKD Stage 3",                 "icd": "N18.3",  "raf": 0.069},
    137: {"name": "CKD Stage 4",                 "icd": "N18.4",  "raf": 0.289},
    136: {"name": "CKD Stage 5",                 "icd": "N18.5",  "raf": 0.289},
    96:  {"name": "Heart Arrhythmias",           "icd": "I48.xx", "raf": 0.179},
    108: {"name": "Vascular Disease w/ Comp.",   "icd": "I70.xx", "raf": 0.539},
    57:  {"name": "Schizophrenia",               "icd": "F20.x",  "raf": 0.421},
    58:  {"name": "Major Depression/Bipolar",    "icd": "F31-33", "raf": 0.212},
}

FEATURES = [
    "age", "age_band", "gender_enc", "dual_eligible", "region_enc",
    "hcc_count", "has_diabetes", "has_chf", "has_ckd", "has_copd",
    "has_cancer", "has_mental_health", "diabetes_ckd", "chf_ckd", "diabetes_chf"
]

FEATURE_LABELS = {
    "age": "Age", "age_band": "Age Band", "gender_enc": "Gender",
    "dual_eligible": "Dual Eligible", "region_enc": "Region",
    "hcc_count": "HCC Count", "has_diabetes": "Diabetes",
    "has_chf": "Congestive Heart Failure", "has_ckd": "Chronic Kidney Disease",
    "has_copd": "COPD", "has_cancer": "Cancer",
    "has_mental_health": "Mental Health Condition",
    "diabetes_ckd": "Diabetes × CKD", "chf_ckd": "CHF × CKD",
    "diabetes_chf": "Diabetes × CHF",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1B4F72;
        border-bottom: 3px solid #2E86C1; padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #2E86C1;
        border-left: 4px solid #2E86C1; padding-left: 0.6rem;
        margin: 1.5rem 0 1rem 0;
    }
    .kpi-box {
        background: linear-gradient(135deg, #EBF5FB, #D6EAF8);
        border: 1px solid #AED6F1; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1B4F72; }
    .kpi-label { font-size: 0.85rem; color: #5D6D7E; margin-top: 0.2rem; }
    .alert-high   { background:#FDEDEC; border-left:4px solid #E74C3C;
                    padding:0.6rem 1rem; border-radius:5px; margin:0.3rem 0; }
    .alert-medium { background:#FEF9E7; border-left:4px solid #F39C12;
                    padding:0.6rem 1rem; border-radius:5px; margin:0.3rem 0; }
    .alert-low    { background:#EAF2FF; border-left:4px solid #3498DB;
                    padding:0.6rem 1rem; border-radius:5px; margin:0.3rem 0; }
    .governance-pass { color:#27AE60; font-weight:700; }
    .governance-warn { color:#E67E22; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def engineer_features(df):
    d = df.copy()
    le_g = LabelEncoder(); le_r = LabelEncoder()
    d["gender_enc"] = le_g.fit_transform(d["gender"])
    d["region_enc"] = le_r.fit_transform(d["region"])
    d["hcc_count"]         = d["current_hcc_list"].apply(
        lambda x: len(str(x).split("|")) if pd.notna(x) and x != "" else 0)
    d["has_diabetes"]      = d["current_icd_codes"].str.contains("E11", na=False).astype(int)
    d["has_chf"]           = d["current_icd_codes"].str.contains("I50", na=False).astype(int)
    d["has_ckd"]           = d["current_icd_codes"].str.contains("N18", na=False).astype(int)
    d["has_copd"]          = d["current_icd_codes"].str.contains("J44", na=False).astype(int)
    d["has_cancer"]        = d["current_icd_codes"].str.contains("C[0-9]", na=False, regex=True).astype(int)
    d["has_mental_health"] = d["current_icd_codes"].str.contains("F[0-9]", na=False, regex=True).astype(int)
    d["diabetes_ckd"]      = d["has_diabetes"] * d["has_ckd"]
    d["chf_ckd"]           = d["has_chf"] * d["has_ckd"]
    d["diabetes_chf"]      = d["has_diabetes"] * d["has_chf"]
    d["age_band"]          = pd.cut(d["age"], bins=[64,69,74,79,84,99],
                                    labels=[0,1,2,3,4]).astype(int)
    return d

@st.cache_data
def build_leakage_df(df):
    rows = []
    for _, row in df.iterrows():
        prior, current = set(), set()
        try:
            if pd.notna(row["prior_hcc_list"]) and str(row["prior_hcc_list"]):
                prior = {int(x) for x in str(row["prior_hcc_list"]).split("|") if x.strip()}
        except: pass
        try:
            if pd.notna(row["current_hcc_list"]) and str(row["current_hcc_list"]):
                current = {int(x) for x in str(row["current_hcc_list"]).split("|") if x.strip()}
        except: pass

        chronic = set(HCC_META.keys())
        missing = (prior - current) & chronic
        raf_delta = sum(HCC_META.get(h, {}).get("raf", 0) for h in missing)
        pmpm_loss = round(raf_delta * 850, 2)
        annual    = round(pmpm_loss * 12, 2)

        if pmpm_loss >= 200:   priority = "High"
        elif pmpm_loss >= 75:  priority = "Medium"
        elif pmpm_loss > 0:    priority = "Low"
        else:                  priority = "None"

        alerts = []
        for h in sorted(missing):
            m = HCC_META.get(h, {})
            alerts.append(
                f"HCC {h} — {m.get('name','?')} | "
                f"Suggest: {m.get('icd','?')} | "
                f"RAF: +{m.get('raf',0):.3f} | "
                f"PMPM: +${m.get('raf',0)*850:.2f}"
            )
        rows.append({
            "member_id":           row["member_id"],
            "age":                 row["age"],
            "gender":              row["gender"],
            "region":              row["region"],
            "disease_profile":     row["disease_profile"],
            "prior_raf":           row["prior_raf_score"],
            "current_raf":         row["current_raf_score"],
            "raf_delta":           round(row["prior_raf_score"] - row["current_raf_score"], 4),
            "missing_hccs":        "|".join(map(str, sorted(missing))),
            "estimated_pmpm_loss": pmpm_loss,
            "estimated_annual_loss": annual,
            "priority_tier":       priority,
            "action_required":     len(missing) > 0,
            "coding_alerts":       " || ".join(alerts) if alerts else "No gaps",
        })
    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.image(
        "https://img.icons8.com/color/96/caduceus.png", width=60)
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("", [
        "🏠 Population Overview",
        "📊 Risk Score Analysis",
        "💰 Revenue Leakage Alerts",
        "🔍 Member Lookup",
        "🏥 Provider Coding Accuracy",
        "🛡️ Model Governance",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")
    regions   = ["All"] + sorted(df["region"].unique().tolist())
    sel_region = st.sidebar.selectbox("Region", regions)
    age_range  = st.sidebar.slider("Age Range", 65, 94, (65, 94))
    dual_only  = st.sidebar.checkbox("Dual Eligible Only", False)

    return page, sel_region, age_range, dual_only


def apply_filters(df, region, age_range, dual_only):
    d = df.copy()
    if region != "All":
        d = d[d["region"] == region]
    d = d[(d["age"] >= age_range[0]) & (d["age"] <= age_range[1])]
    if dual_only:
        d = d[d["dual_eligible"] == 1]
    return d


# ── Pages ──────────────────────────────────────────────────────────────────
def page_overview(df):
    st.markdown('<div class="main-header">🏥 Medicare Risk Adjustment Engine</div>',
                unsafe_allow_html=True)
    st.caption("Explainable HCC Risk Scoring · Revenue Leakage Detection · SR 11-7 Governance")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (f"{len(df):,}",                      "Total Members"),
        (f"{df['current_raf_score'].mean():.3f}", "Avg RAF Score"),
        (f"${df['current_pmpm'].mean():,.0f}", "Avg PMPM"),
        (f"{df['dual_eligible'].mean()*100:.1f}%", "Dual Eligible"),
        (f"{df['has_leakage'].mean()*100:.1f}%",   "Leakage Rate"),
    ]
    for col, (val, lbl) in zip([col1,col2,col3,col4,col5], kpis):
        col.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Disease profile + RAF distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Disease Profile Distribution</div>',
                    unsafe_allow_html=True)
        profile_counts = df["disease_profile"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(profile_counts)))
        bars = ax.barh(profile_counts.index, profile_counts.values,
                       color=colors, edgecolor="white")
        for bar, val in zip(bars, profile_counts.values):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=8)
        ax.set_xlabel("Member Count")
        ax.set_title("Members by Disease Profile", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">RAF Score by Disease Profile</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        profile_raf = df.groupby("disease_profile")["current_raf_score"].mean().sort_values()
        colors2 = ["#C44E52" if v > 1.0 else "#4C72B0" for v in profile_raf.values]
        ax.barh(profile_raf.index, profile_raf.values, color=colors2, edgecolor="white")
        ax.axvline(1.0, color="orange", linestyle="--", linewidth=1.5, label="RAF = 1.0")
        ax.set_xlabel("Average RAF Score")
        ax.set_title("Avg RAF Score by Profile", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Age & gender breakdown
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">Age Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df["age"], bins=20, color="#2E86C1", edgecolor="white")
        ax.axvline(df["age"].mean(), color="#C44E52", linestyle="--",
                   linewidth=2, label=f"Mean: {df['age'].mean():.1f}")
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.set_title("Member Age Distribution", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown('<div class="section-header">RAF Score Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df["current_raf_score"], bins=30, color="#55A868", edgecolor="white")
        ax.axvline(df["current_raf_score"].mean(), color="#C44E52", linestyle="--",
                   linewidth=2, label=f"Mean: {df['current_raf_score'].mean():.3f}")
        ax.set_xlabel("RAF Score"); ax.set_ylabel("Count")
        ax.set_title("Current RAF Score Distribution", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()


def page_risk_analysis(df):
    st.markdown('<div class="main-header">📊 Risk Score Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">RAF Score by Region</div>',
                    unsafe_allow_html=True)
        region_raf = df.groupby("region")["current_raf_score"].agg(["mean","std"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(region_raf["region"], region_raf["mean"],
                      yerr=region_raf["std"], color="#2E86C1",
                      edgecolor="white", capsize=5)
        ax.set_ylabel("Avg RAF Score")
        ax.set_title("RAF Score by Region (mean ± std)", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">RAF Score by Age Group</div>',
                    unsafe_allow_html=True)
        df2 = df.copy()
        df2["age_group"] = pd.cut(df2["age"], bins=[64,74,84,99],
                                   labels=["65-74","75-84","85+"])
        age_raf = df2.groupby("age_group")["current_raf_score"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(age_raf.index.astype(str), age_raf.values,
               color=["#4C72B0","#DD8452","#C44E52"], edgecolor="white")
        ax.set_ylabel("Avg RAF Score")
        ax.set_title("RAF Score by Age Group", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Top 10 High-Risk Members</div>',
                unsafe_allow_html=True)
    top10 = df.nlargest(10, "current_raf_score")[
        ["member_id","age","gender","region","disease_profile",
         "current_raf_score","current_pmpm","dual_eligible"]
    ].rename(columns={
        "member_id":"Member ID","age":"Age","gender":"Gender",
        "region":"Region","disease_profile":"Profile",
        "current_raf_score":"RAF Score","current_pmpm":"PMPM ($)",
        "dual_eligible":"Dual"
    })
    st.dataframe(top10.style.background_gradient(subset=["RAF Score"], cmap="Reds"),
                 use_container_width=True)

    # Saved SHAP plots
    st.markdown('<div class="section-header">SHAP Explainability — Feature Importance</div>',
                unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    shap_path = f"{SHAP_DIR}/shap_summary.png"
    feat_path = f"{SHAP_DIR}/feature_importance.png"
    if os.path.exists(shap_path):
        col_s1.image(shap_path, caption="SHAP Summary Plot", use_container_width=True)
    if os.path.exists(feat_path):
        col_s2.image(feat_path, caption="XGBoost Feature Importance", use_container_width=True)

    wf_path = f"{SHAP_DIR}/shap_waterfall_sample.png"
    if os.path.exists(wf_path):
        st.markdown('<div class="section-header">SHAP Waterfall — Sample Member</div>',
                    unsafe_allow_html=True)
        st.image(wf_path, caption="Individual Member RAF Score Breakdown",
                 use_container_width=True)


def page_leakage(leakage_df):
    st.markdown('<div class="main-header">💰 Revenue Leakage Alerts</div>',
                unsafe_allow_html=True)

    flagged = leakage_df[leakage_df["action_required"]]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""<div class="kpi-box">
        <div class="kpi-value">{len(flagged):,}</div>
        <div class="kpi-label">Members Flagged</div></div>""",
        unsafe_allow_html=True)
    col2.markdown(f"""<div class="kpi-box">
        <div class="kpi-value">${flagged['estimated_pmpm_loss'].sum():,.0f}</div>
        <div class="kpi-label">Monthly Leakage</div></div>""",
        unsafe_allow_html=True)
    col3.markdown(f"""<div class="kpi-box">
        <div class="kpi-value">${flagged['estimated_annual_loss'].sum():,.0f}</div>
        <div class="kpi-label">Annual Leakage Est.</div></div>""",
        unsafe_allow_html=True)
    col4.markdown(f"""<div class="kpi-box">
        <div class="kpi-value">{len(flagged[flagged['priority_tier']=='High']):,}</div>
        <div class="kpi-label">High Priority</div></div>""",
        unsafe_allow_html=True)

    st.markdown("---")

    # Dashboard image
    dash_path = f"{SHAP_DIR}/leakage_dashboard.png"
    if os.path.exists(dash_path):
        st.image(dash_path, use_container_width=True)

    # Priority filter
    st.markdown('<div class="section-header">Flagged Member Alerts</div>',
                unsafe_allow_html=True)
    tier_filter = st.selectbox("Filter by Priority", ["All","High","Medium","Low"])
    show_df = flagged if tier_filter == "All" else flagged[flagged["priority_tier"] == tier_filter]

    for _, row in show_df.head(20).iterrows():
        tier  = row["priority_tier"]
        css   = f"alert-{tier.lower()}"
        icon  = "🔴" if tier=="High" else "🟡" if tier=="Medium" else "🔵"
        st.markdown(f"""
        <div class="{css}">
            <b>{icon} {row['member_id']}</b> | Age: {row['age']} | {row['region']} |
            RAF: {row['prior_raf']:.3f} → {row['current_raf']:.3f}
            (Δ {row['raf_delta']:.3f}) |
            <b>PMPM Loss: ${row['estimated_pmpm_loss']:,.2f}</b> |
            Annual: ${row['estimated_annual_loss']:,.2f}<br>
            <small>{row['coding_alerts'][:150]}...</small>
        </div>""", unsafe_allow_html=True)


def page_member_lookup(df, leakage_df, model):
    st.markdown('<div class="main-header">🔍 Member Lookup</div>',
                unsafe_allow_html=True)

    member_ids = df["member_id"].tolist()
    selected   = st.selectbox("Select Member ID", member_ids)

    row = df[df["member_id"] == selected].iloc[0]
    leak_row = leakage_df[leakage_df["member_id"] == selected].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current RAF Score", f"{row['current_raf_score']:.3f}",
                delta=f"{leak_row['raf_delta']:.3f} vs prior year")
    col2.metric("Current PMPM", f"${row['current_pmpm']:,.2f}")
    col3.metric("Priority Tier", leak_row["priority_tier"])

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Demographics**")
        st.write(f"- Age: {row['age']} | Gender: {row['gender']}")
        st.write(f"- Region: {row['region']}")
        st.write(f"- Dual Eligible: {'Yes' if row['dual_eligible'] else 'No'}")
        st.write(f"- Disease Profile: {row['disease_profile']}")

    with col_b:
        st.markdown("**Diagnosis Codes**")
        st.write(f"- Current ICD Codes: `{row['current_icd_codes']}`")
        st.write(f"- Prior ICD Codes: `{row['prior_icd_codes']}`")
        st.write(f"- Current HCCs: `{row['current_hcc_list']}`")
        st.write(f"- Prior HCCs: `{row['prior_hcc_list']}`")

    # Coding alerts
    if leak_row["action_required"]:
        st.markdown("---")
        st.markdown("**⚠️ Coding Gap Alerts**")
        for alert in leak_row["coding_alerts"].split(" || "):
            st.warning(alert)
    else:
        st.success("✅ No coding gaps detected for this member.")


def page_provider(leakage_df):
    st.markdown('<div class="main-header">🏥 Provider Coding Accuracy</div>',
                unsafe_allow_html=True)

    region_stats = leakage_df.groupby("region").agg(
        total_members    =("member_id","count"),
        members_with_gap =("action_required","sum"),
        total_pmpm_loss  =("estimated_pmpm_loss","sum"),
        avg_raf_delta    =("raf_delta","mean"),
    ).reset_index()
    region_stats["gap_rate_pct"]          = round(region_stats["members_with_gap"] / region_stats["total_members"] * 100, 1)
    region_stats["coding_accuracy_score"] = round(100 - region_stats["gap_rate_pct"], 1)
    region_stats["annual_leakage_est"]    = round(region_stats["total_pmpm_loss"] * 12, 2)
    region_stats = region_stats.sort_values("coding_accuracy_score")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Coding Accuracy Score</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#C44E52" if s < 97 else "#55A868"
                  for s in region_stats["coding_accuracy_score"]]
        ax.barh(region_stats["region"], region_stats["coding_accuracy_score"],
                color=colors, edgecolor="white")
        ax.axvline(97, color="orange", linestyle="--", linewidth=1.5,
                   label="97% Threshold")
        ax.set_xlim(90, 102)
        for i, (_, row) in enumerate(region_stats.iterrows()):
            ax.text(row["coding_accuracy_score"] + 0.1, i,
                    f"{row['coding_accuracy_score']:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Coding Accuracy (%)")
        ax.set_title("Coding Accuracy by Region", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Annual Leakage by Region</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(region_stats["region"], region_stats["annual_leakage_est"],
                color="#C44E52", edgecolor="white")
        for i, (_, row) in enumerate(region_stats.iterrows()):
            ax.text(row["annual_leakage_est"] + 100, i,
                    f"${row['annual_leakage_est']:,.0f}", va="center", fontsize=9)
        ax.set_xlabel("Estimated Annual Leakage ($)")
        ax.set_title("Annual Revenue Leakage by Region", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Region Summary Table</div>',
                unsafe_allow_html=True)
    display_df = region_stats.rename(columns={
        "region":"Region","total_members":"Members",
        "members_with_gap":"Gaps Found","gap_rate_pct":"Gap Rate (%)",
        "coding_accuracy_score":"Accuracy Score","annual_leakage_est":"Annual Leakage ($)"
    })[["Region","Members","Gaps Found","Gap Rate (%)","Accuracy Score","Annual Leakage ($)"]]
    st.dataframe(display_df.style.background_gradient(subset=["Accuracy Score"], cmap="RdYlGn"),
                 use_container_width=True)


def page_governance():
    st.markdown('<div class="main-header">🛡️ Model Governance</div>',
                unsafe_allow_html=True)
    st.caption("SR 11-7 Aligned Model Risk Documentation")

    gov_path = f"{SHAP_DIR}/governance_report.png"
    if os.path.exists(gov_path):
        st.image(gov_path, use_container_width=True)

    st.markdown('<div class="section-header">Governance Checklist</div>',
                unsafe_allow_html=True)
    checks = [
        ("Model Purpose",        "Medicare RAF Score Prediction & Revenue Leakage Detection", True),
        ("Regulatory Framework", "SR 11-7 / CMS Risk Adjustment Guidelines",                 True),
        ("CV Stability",         "STABLE — 5-Fold R² = 0.708 ± 0.040",                       True),
        ("Bias / Fairness",      "No disparate impact — Gender, Age, Dual status checked",    True),
        ("Explainability",       "SHAP values — Full feature attribution available",           True),
        ("Residual Analysis",    "Non-normal residuals — documented (expected in RAF models)", False),
        ("Overfit Monitoring",   "Gap = 0.20 — Noted; linear model competitive",              False),
        ("Audit Readiness",      "Full documentation & reproducible pipeline",                 True),
        ("Recommended Review",   "Annual revalidation / post-CMS model update",               True),
        ("Next Validation Step", "Out-of-time sample test with real encounter data",           True),
    ]

    for label, value, passed in checks:
        icon  = "✅" if passed else "⚠️"
        color = "#27AE60" if passed else "#E67E22"
        col1, col2, col3 = st.columns([2, 4, 1])
        col1.write(f"**{label}**")
        col2.write(value)
        col3.markdown(f"<span style='color:{color}; font-size:1.2rem'>{icon}</span>",
                      unsafe_allow_html=True)

    st.markdown("---")
    st.info("""
    **Model Limitations Note (SR 11-7 §4)**

    1. Trained on synthetic data — performance on real CMS encounter data requires revalidation.
    2. RAF score is inherently rule-based and additive by CMS design; non-normal residuals are expected.
    3. XGBoost overfit gap (0.20) is noted — linear model remains competitive for this use case.
    4. Missing HCC detection relies on prior year coding completeness.
    5. Annual revalidation recommended following CMS HCC model version updates (v24 → v28 transition).
    """)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    df         = load_data()
    model      = load_model()
    df_eng     = engineer_features(df)
    leakage_df = build_leakage_df(df)

    page, sel_region, age_range, dual_only = render_sidebar(df)
    df_filtered = apply_filters(df, sel_region, age_range, dual_only)
    leakage_filtered = apply_filters(leakage_df, sel_region, age_range, dual_only)

    if page == "🏠 Population Overview":
        page_overview(df_filtered)
    elif page == "📊 Risk Score Analysis":
        page_risk_analysis(df_filtered)
    elif page == "💰 Revenue Leakage Alerts":
        page_leakage(leakage_filtered)
    elif page == "🔍 Member Lookup":
        page_member_lookup(df, leakage_df, model)
    elif page == "🏥 Provider Coding Accuracy":
        page_provider(leakage_filtered)
    elif page == "🛡️ Model Governance":
        page_governance()


if __name__ == "__main__":
    main()
