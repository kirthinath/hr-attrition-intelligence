import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# ── Data ──────────────────────────────────────────────────────────────────────
from app_utils import load_scorecard, load_risk_scores, load_segmented

seg_df  = load_segmented(BASE_DIR)
risk_df = load_risk_scores(BASE_DIR)
sc      = load_scorecard(BASE_DIR)

# ── Page Header ───────────────────────────────────────────────────────────────
section_header("Executive Command Center",
               "Real-time workforce risk overview for HR leadership and executive stakeholders.")

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
        sel_dept  = st.selectbox("Department", dept_opts)
        sel_risk  = st.selectbox("Risk Level", ["All", "High", "Medium", "Low"])

# Apply filters
fseg = seg_df.copy()
frisk = risk_df.copy()
if sel_dept != "All" and not fseg.empty:
    fseg = fseg[fseg["department"] == sel_dept]
if not frisk.empty and not seg_df.empty:
    dmap = seg_df[["employee_id","department"]].rename(columns={"employee_id":"EmployeeID"})
    frisk = frisk.merge(dmap, on="EmployeeID", how="left")
    if sel_dept != "All":
        frisk = frisk[frisk["department"] == sel_dept]
if sel_risk != "All" and not frisk.empty:
    frisk = frisk[frisk["RiskLevel"] == sel_risk]

with col_main:
    # ── 6 KPI Cards ───────────────────────────────────────────────────────────
    kpis = sc.get("workforce_kpis", {})
    fin  = sc.get("financial_impact", {})
    pred = sc.get("predictive_risk_exposure", {})
    ot   = sc.get("overtime_vulnerability", {})

    kpi_list = [
        (f"{kpis.get('current_active_headcount', 1233):,}",   "Active Employees",        COLORS["blue"]),
        (f"{kpis.get('overall_attrition_rate', 0.161):.1%}",  "Historical Attrition",    COLORS["red"]),
        (f"${fin.get('total_estimated_attrition_cost', 20421738)/1e6:.1f}M", "Est. Annual Cost", COLORS["amber"]),
        (f"{pred.get('active_high_risk_headcount', 18)}",     "High-Risk Employees",     COLORS["red"]),
        (f"{pred.get('active_medium_risk_headcount', 244)}",  "Medium-Risk Employees",   COLORS["amber"]),
        (f"{ot.get('overtime_attrition_multiplier', 2.93):.2f}×", "Overtime Risk Multiplier", COLORS["red"]),
    ]

    # Render KPI cards in a flex/grid markdown directly to avoid nested Streamlit column wrappers that render as empty boxes
    cards_html = "".join([f"""
    <div style="background:#FFFFFF;
         border:1px solid #E5E7EB;border-top:3px solid {color};
         border-radius:12px;padding:20px 16px;
         box-shadow:0 1px 3px rgba(0,0,0,0.08);
         display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:0.62rem;color:#6B7280;font-weight:700;letter-spacing:0.12em;
             text-transform:uppercase;margin-bottom:8px;">{label}</div>
        <div style="font-size:1.9rem;font-weight:800;line-height:1;
             color:{color};">{val}</div>
    </div>""" for val, label, color in kpi_list])

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:16px;margin-bottom:12px;">
        {cards_html}
    </div>""", unsafe_allow_html=True)

    # ── Risk Distribution + Dept Attrition ───────────────────────────────────
    subsection("Workforce Risk Distribution")
    col_donut, col_dept = st.columns([1, 2])

    with col_donut:
        # Wrap container strictly inside the non-empty check to prevent empty cards from rendering
        if not frisk.empty and "RiskLevel" in frisk.columns:
            with st.container(border=True):
                rc = frisk["RiskLevel"].value_counts().reset_index()
                rc.columns = ["Risk Level", "Count"]
                fig = px.pie(rc, names="Risk Level", values="Count",
                             color="Risk Level", color_discrete_map=RISK_COLORS, hole=0.65)
                fig.update_traces(
                    textposition="outside",
                    textinfo="label+percent",
                    textfont=dict(color="#374151", size=11),
                    marker=dict(line=dict(color="#FFFFFF", width=3)),
                    domain=dict(x=[0.1, 0.9]) # Give space on left/right for labels to prevent left cutoff
                )
                fig.add_annotation(
                    text=f"<b>{len(frisk)}</b><br><span style='font-size:9px'>employees</span>",
                    x=0.5, y=0.5, font=dict(size=15, color="#1A1A1A"), showarrow=False,
                )
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    showlegend=False, margin=dict(t=30, b=30, l=60, r=60), height=270,
                    font=dict(color="#374151", family="Inter"),
                    transition_duration=300,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=13,
                        font_family="Inter, sans-serif"
                    )
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_dept:
        # Wrap container strictly inside the non-empty check to prevent empty cards from rendering
        if not fseg.empty and "department" in fseg.columns:
            with st.container(border=True):
                subsection("Attrition Rate by Department")
                da = fseg.groupby("department").agg(
                    Total=("attrition", "count"), Dep=("attrition", "sum")
                ).reset_index()
                da["Rate"] = da["Dep"] / da["Total"]
                da = da.sort_values("Rate", ascending=True)
                fig = px.bar(da, x="Rate", y="department", orientation="h",
                             color="Rate",
                             color_continuous_scale=[[0, COLORS["green"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
                             text=da["Rate"].apply(lambda x: f"{x:.1%}"))
                fig.update_traces(textposition="outside", textfont=dict(color="#374151"))
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="#374151", family="Inter"), coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
                    yaxis=dict(showgrid=False, title="", tickfont=dict(size=13, color="#374151")),
                    margin=dict(t=10, b=10, l=10, r=100), height=270, # Increased right margin to prevent percentage label cutoff
                    transition_duration=300,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=13,
                        font_family="Inter, sans-serif"
                    )
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col_box, col_seg = st.columns(2)

    with col_box:
        # Wrap container strictly inside the non-empty check to prevent empty cards from rendering
        if not fseg.empty and "monthly_income" in fseg.columns and "RiskLevel" in fseg.columns:
            with st.container(border=True):
                subsection("Income Distribution by Risk Level")
                fig = px.box(fseg, x="RiskLevel", y="monthly_income",
                             color="RiskLevel", color_discrete_map=RISK_COLORS,
                             category_orders={"RiskLevel": ["High", "Medium", "Low"]})
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="#374151", family="Inter"), showlegend=False,
                    xaxis=dict(showgrid=False, title="Risk Level"),
                    yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Monthly Income ($)"),
                    margin=dict(t=10, b=10), height=270,
                    transition_duration=300,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=13,
                        font_family="Inter, sans-serif"
                    )
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_seg:
        # Wrap container strictly inside the non-empty check to prevent empty cards from rendering
        if not fseg.empty and "primary_segment" in fseg.columns:
            with st.container(border=True):
                subsection("Workforce Segment Composition")
                ss = fseg["primary_segment"].value_counts().reset_index()
                ss.columns = ["Segment", "Count"]
                ss = ss.sort_values("Count", ascending=True)
                fig = px.bar(ss, x="Count", y="Segment", orientation="h",
                             color="Count",
                             color_continuous_scale=[[0, "#E6F4EA"], [1, "#1B4332"]])
                fig.update_traces(texttemplate="%{x}", textposition="outside",
                                  textfont=dict(color="#374151"))
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="#374151", family="Inter"), coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, title="Employees"),
                    yaxis=dict(showgrid=False, title=""),
                    margin=dict(t=10, b=10), height=270,
                    transition_duration=300,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=13,
                        font_family="Inter, sans-serif"
                    )
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Financial Impact ──────────────────────────────────────────────────────
    subsection("Financial Impact Summary")
    fi_items = [
        (f"${fin.get('average_cost_per_departed_employee', 86167):,.0f}", "Avg. Cost per Departure", COLORS["red"]),
        (f"${fin.get('total_lifetime_value_loss_elv', 111842780)/1e6:.1f}M", "Total Lifetime Value Lost", COLORS["amber"]),
        (f"${fin.get('average_active_employee_elv', 815725):,.0f}", "Avg. Active Employee ELV", COLORS["green"]),
        (f"${fin.get('average_departed_employee_elv', 471910):,.0f}", "Avg. Departed Employee ELV", COLORS["blue"]),
    ]
    fcols = st.columns(4)
    for fc, (val, label, color) in zip(fcols, fi_items):
        fc.markdown(f"""
        <div style="background:#FFFFFF;
             border:1px solid #E5E7EB;border-top:3px solid {color};
             border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
            <div style="font-size:0.62rem;color:#6B7280;font-weight:700;letter-spacing:0.12em;
                 text-transform:uppercase;margin-bottom:8px;">{label}</div>
            <div style="font-size:1.7rem;font-weight:800;
                 color:{color};">{val}</div>
        </div>""", unsafe_allow_html=True)
