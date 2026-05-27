import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from app_utils import (inject_global_css, section_header, subsection, info_card,
                       spacer, divider, sidebar_brand,
                       COLORS, RISK_COLORS, BASE_DIR)

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_global_css()

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_attrition_model.pkl")

@st.cache_data
def load_risk_scores():
    p = os.path.join(BASE_DIR,"data","exports","employee_attrition_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_segmented():
    p = os.path.join(BASE_DIR,"data","processed","segmented_hr_data.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def encode_travel(val):
    return {"Non-Travel":0,"Travel_Rarely":1,"Travel_Frequently":2}.get(str(val),1)

def apply_fe(df):
    df = df.copy()
    tn = df["business_travel"].apply(encode_travel)
    df["income_band"] = pd.cut(df["monthly_income"],[0,3000,6000,12000,1e9],labels=["Low","Medium","High","Executive"])
    df["tenure_group"] = pd.cut(df["years_at_company"],[-1,1,4,9,1e9],labels=["New Hire","Junior","Experienced","Veteran"])
    df["promotion_delay_risk"]         = df["years_since_last_promotion"]/(df["years_in_current_role"]+1.0)
    df["overtime_risk_score"]          = df["overtime"]*(5-df["work_life_balance"])*(5-df["job_involvement"])
    df["work_life_balance_risk"]       = (5-df["work_life_balance"])*(1+tn)+(df["distance_from_home"]/10.0)
    df["satisfaction_composite_score"] = (df["environment_satisfaction"]+df["job_satisfaction"]+df["relationship_satisfaction"])/3.0
    df["manager_stability_score"]      = df["years_with_curr_manager"]/(df["years_at_company"]+1.0)
    df["career_growth_risk_indicator"] = df["total_working_years"]/(df["job_level"]+1.0)
    df["travel_burden_score"]          = tn*df["distance_from_home"]
    df["burnout_risk_index"]           = df["overtime_risk_score"]+df["work_life_balance_risk"]+(5-df["job_satisfaction"])
    return df

risk_df = load_risk_scores()
seg_df  = load_segmented()

# ── Header ────────────────────────────────────────────────────────────────────
section_header("Workforce Risk Intelligence",
               "Browse, filter, and drill into the complete risk-scored workforce. Upload CSVs for bulk scoring.")

# ── Layout columns below header ────────────────────────────────────────────────
col_main, col_sidebar = st.columns([4.0, 1.0])

with col_sidebar:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size:0.68rem;color:#1B4332;font-weight:700;
             letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            Filter Console
        </div>
        """, unsafe_allow_html=True)
        dept_opts = ["All"] + sorted(seg_df["department"].dropna().unique().tolist()) if not seg_df.empty else ["All"]
        role_opts = ["All"] + sorted(seg_df["job_role"].dropna().unique().tolist()) if not seg_df.empty else ["All"]
        sel_dept  = st.selectbox("Department", dept_opts)
        sel_risk  = st.selectbox("Risk Level", ["All","High","Medium","Low"])
        sel_role  = st.selectbox("Job Role", role_opts)

full_df = risk_df.copy()
if not risk_df.empty and not seg_df.empty:
    full_df = risk_df.merge(
        seg_df[["employee_id","department","job_role","monthly_income",
                "years_at_company","overtime","primary_segment"
                ]].rename(columns={"employee_id":"EmployeeID"}),
        on="EmployeeID", how="left")

filtered = full_df.copy()
if sel_dept != "All" and "department" in filtered.columns:
    filtered = filtered[filtered["department"] == sel_dept]
if sel_risk != "All":
    filtered = filtered[filtered["RiskLevel"] == sel_risk]
if sel_role != "All" and "job_role" in filtered.columns:
    filtered = filtered[filtered["job_role"] == sel_role]

with col_main:
    # ── 4 KPI Cards ───────────────────────────────────────────────────────────
    h_count = f"{(filtered['RiskLevel']=='High').sum():,}" if not filtered.empty else "—"
    h_pct = f"{(filtered['RiskLevel']=='High').mean():.1%}" if not filtered.empty else "—"
    avg_risk = f"{filtered['AttritionProbability'].mean():.1%}" if not filtered.empty else "—"
    avg_income = f"${filtered['monthly_income'].mean():,.0f}" if "monthly_income" in filtered.columns and not filtered.empty else "—"

    kpi_list = [
        (f"{len(filtered):,}", "Employees Shown", COLORS["blue"], ""),
        (h_count, "High Risk", COLORS["red"], f"{h_pct} of filter"),
        (avg_risk, "Avg Risk Score", COLORS["amber"], ""),
        (avg_income, "Avg Monthly Income", COLORS["blue"], ""),
    ]

    cards_html = "".join([f"""
    <div style="background:#FFFFFF;
         border:1px solid #E5E7EB;border-top:3px solid {color};
         border-radius:12px;padding:20px 16px;
         box-shadow:0 1px 3px rgba(0,0,0,0.08);
         display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:0.62rem;color:#6B7280;font-weight:700;letter-spacing:0.12em;
             text-transform:uppercase;margin-bottom:8px;">{label}</div>
        <div style="font-size:1.7rem;font-weight:800;line-height:1;
             color:{color};">{val}</div>
        {f'<div style="font-size:0.72rem;color:#EF4444;margin-top:4px;font-weight:500;">{sub}</div>' if sub else ''}
    </div>""" for val, label, color, sub in kpi_list])

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:16px;margin-bottom:12px;">
        {cards_html}
    </div>""", unsafe_allow_html=True)

    # Use exact 24px gap
    spacer(24)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "  Employee Risk Register  ",
        "  Visual Analytics  ",
        "  Bulk CSV Scoring  "
    ])

    with tab1:
        if not filtered.empty:
            spacer(24)
            with st.container(border=True):
                display_cols = ["EmployeeID","RiskLevel","AttritionProbability","TopRiskFactors",
                                "RecommendedRetentionAction","department","job_role","monthly_income"]
                disp = filtered[[c for c in display_cols if c in filtered.columns]].copy()
                if "RecommendedRetentionAction" in disp.columns:
                    disp["RecommendedRetentionAction"] = disp["RecommendedRetentionAction"].apply(
                        lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x
                    )
                if "AttritionProbability" in disp.columns:
                    disp["Risk %"] = (disp["AttritionProbability"]*100).round(1).astype(str)+"%"
                    disp = disp.drop(columns=["AttritionProbability"])
                st.dataframe(
                    disp.sort_values("Risk %", ascending=False) if "Risk %" in disp.columns else disp,
                    use_container_width=True, height=480
                )
                spacer(12)
                st.download_button(
                    "Download Filtered Risk Report (CSV)",
                    filtered.to_csv(index=False).encode("utf-8"),
                    "hr_risk_report_filtered.csv", "text/csv",
                )
        else:
            spacer(24)
            st.info("No data matches the current filters.")

    with tab2:
        if not filtered.empty:
            spacer(24)
            c1, c2 = st.columns(2)
            with c1:
                if "AttritionProbability" in filtered.columns:
                    with st.container(border=True):
                        subsection("Risk Probability Distribution")
                        fig = px.histogram(filtered, x="AttritionProbability", nbins=20,
                                           color="RiskLevel", color_discrete_map=RISK_COLORS,
                                           barmode="overlay", opacity=0.8)
                        fig.update_layout(
                            paper_bgcolor="white", plot_bgcolor="white",
                            font=dict(color="#374151", family="Inter"),
                            xaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Attrition Probability"),
                            yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Employee Count"),
                            legend=dict(bgcolor="white"),
                            margin=dict(t=10, b=10), height=280,
                            transition_duration=300,
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=13,
                                font_family="Inter, sans-serif"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with c2:
                if "monthly_income" in filtered.columns:
                    with st.container(border=True):
                        subsection("Income vs. Attrition Risk")
                        fig = px.scatter(filtered, x="monthly_income", y="AttritionProbability",
                                         color="RiskLevel", color_discrete_map=RISK_COLORS, opacity=0.6,
                                         labels={"monthly_income":"Monthly Income ($)","AttritionProbability":"Risk"})
                        fig.update_layout(
                            paper_bgcolor="white", plot_bgcolor="white",
                            font=dict(color="#374151", family="Inter"),
                            xaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
                            yaxis=dict(showgrid=True, gridcolor="#F3F4F6"),
                            legend=dict(bgcolor="white"),
                            margin=dict(t=10, b=10), height=280,
                            transition_duration=300,
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=13,
                                font_family="Inter, sans-serif"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            if "department" in filtered.columns and "primary_segment" in filtered.columns:
                spacer(24)
                with st.container(border=True):
                    subsection("Risk Heatmap: Department × Workforce Segment")
                    pivot = (filtered.groupby(["department","primary_segment"])["AttritionProbability"]
                             .mean().reset_index()
                             .pivot(index="primary_segment", columns="department", values="AttritionProbability")
                             .fillna(0))
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                        colorscale=[[0, "#DCFCE7"], [0.5, "#FEF3C7"], [1, "#FEE2E2"]],
                        text=np.round(pivot.values*100, 1),
                        texttemplate="%{text}%",
                        hovertemplate="Dept: %{x}<br>Segment: %{y}<br>Avg Risk: %{z:.1%}<extra></extra>",
                    ))
                    fig.update_layout(
                        paper_bgcolor="white", plot_bgcolor="white",
                        font=dict(color="#374151", family="Inter"),
                        margin=dict(t=10, b=10), height=300,
                        transition_duration=300,
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=13,
                            font_family="Inter, sans-serif"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        spacer(24)
        with st.container(border=True):
            info_card("Upload a CSV of employees. The system applies feature engineering and runs the XGBoost "
                      "model to return <strong>AttritionProbability</strong> and <strong>RiskLevel</strong> for each row.",
                      COLORS["blue"])
            uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])
            if uploaded:
                try:
                    udf = pd.read_csv(uploaded)
                    st.success(f"Loaded {len(udf):,} employee records.")
                    model = load_model()
                    if model is None:
                        st.error("Model not found.")
                    else:
                        with st.spinner("Scoring employees..."):
                            fe = apply_fe(udf)
                            X = fe.drop(columns=[c for c in ["attrition","employee_id","EmployeeID"] if c in fe.columns], errors="ignore")
                            probs = model.predict_proba(X)[:, 1]
                            udf["AttritionProbability"] = np.round(probs, 4)
                            udf["RiskLevel"] = ["High" if p>=0.7 else ("Medium" if p>=0.3 else "Low") for p in probs]

                        h = int((udf["RiskLevel"]=="High").sum())
                        m = int((udf["RiskLevel"]=="Medium").sum())
                        lo = int((udf["RiskLevel"]=="Low").sum())
                        st.markdown(f"""
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:16px 0;">
                            {"".join([f'''<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-top:3px solid {c};border-radius:10px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
                                <div style="font-size:1.8rem;font-weight:800;color:{c};">{v}</div>
                                <div style="font-size:0.65rem;color:#6B7280;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-top:4px;">{l}</div>
                            </div>''' for v,l,c in [(h,"High Risk",COLORS["red"]),(m,"Medium Risk",COLORS["amber"]),(lo,"Low Risk",COLORS["green"])]])}
                        </div>""", unsafe_allow_html=True)
                        st.dataframe(udf.head(50), use_container_width=True, height=300)
                        spacer(12)
                        st.download_button("Download Full Scored Report",
                                           udf.to_csv(index=False).encode("utf-8"),
                                           "bulk_risk_scored.csv", "text/csv")
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
