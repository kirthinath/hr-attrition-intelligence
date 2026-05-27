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

FIG_DIR = os.path.join(BASE_DIR,"reports","figures")

@st.cache_data
def load_seg():
    path = os.path.join(BASE_DIR,"data","processed","segmented_hr_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

seg = load_seg()

# ── Page Header ───────────────────────────────────────────────────────────────
section_header("Attrition Drivers & Explainability",
               "Understand the specific factors driving attrition via SHAP, feature importance, and model validation.")

# ── Layout columns below header ────────────────────────────────────────────────
col_main, col_sidebar = st.columns([4.0, 1.0])

with col_sidebar:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size:0.68rem;color:#1B4332;font-weight:700;
             letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            Model Drivers
        </div>
        <div style="font-size:0.8rem;color:#6B7280;line-height:1.6;margin:0;">
            This page visualizes the global drivers of attrition across the entire workforce. Use the tabs to explore different facets of model explainability and performance.
        </div>
        """, unsafe_allow_html=True)

with col_main:
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Feature Importance  ",
        "  SHAP Global Analysis  ",
        "  Deep-Dive Analytics  ",
        "  Model Validation  "
    ])

    # ── TAB 1: Feature Importance ─────────────────────────────────────────────
    with tab1:
        spacer(24)
        with st.container(border=True):
            info_card("Feature importance shows how strongly each variable correlates with attrition. "
                      "Higher values indicate variables the model relies on most when making predictions.", COLORS["blue"])

            if not seg.empty:
                skip = {"attrition","employee_id","segment_champions","segment_loyal",
                        "segment_high_performers_at_risk","segment_burnout_risk",
                        "segment_disengaged","segment_new_employees","segment_flight_risk"}
                num_cols = [c for c in seg.select_dtypes(include=[np.number]).columns if c not in skip]
                corr = seg[num_cols+["attrition"]].corr()["attrition"].drop("attrition").abs().sort_values(ascending=False)
                fi = corr.reset_index(); fi.columns = ["Feature","Importance"]
                fi["Feature"] = fi["Feature"].str.replace("_"," ").str.title()
                fi = fi.head(20)

                colors = [COLORS["blue"] if i < 3 else "#2D6A4F" for i in range(len(fi))]

                fig = go.Figure(go.Bar(
                    x=fi.sort_values("Importance")["Importance"],
                    y=fi.sort_values("Importance")["Feature"],
                    orientation="h",
                    marker=dict(
                        color=colors[::-1],
                        line=dict(color="#E5E7EB", width=1)
                    ),
                    text=fi.sort_values("Importance")["Importance"].apply(lambda x: f"{x:.3f}"),
                    textposition="outside", textfont=dict(color="#374151",size=11),
                ))
                fig.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="#374151",family="Inter"),
                    xaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False, title="Absolute Correlation with Attrition"),
                    yaxis=dict(showgrid=False, title=""),
                    margin=dict(t=10,b=10,l=10,r=80), height=580,
                    transition_duration=300,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=13,
                        font_family="Inter, sans-serif"
                    )
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            fi_path = os.path.join(FIG_DIR,"14_feature_importance.png")
            if os.path.exists(fi_path):
                spacer(12)
                with st.expander("View Full Model Feature Importance Plot (from training run)"):
                    st.image(fi_path, caption="XGBoost Model Feature Importance — Generated During Training Pipeline", use_container_width=True)

    # ── TAB 2: SHAP ───────────────────────────────────────────────────────────
    with tab2:
        spacer(24)
        with st.container(border=True):
            info_card("SHAP (SHapley Additive exPlanations) assigns each feature an exact contribution to the model's output. "
                      "Red dots push the model toward predicting departure. Blue dots push toward retention.", COLORS["blue"])

            shap_path = os.path.join(FIG_DIR,"15_shap_summary.png")
            if os.path.exists(shap_path):
                st.image(shap_path, caption="SHAP Global Summary — Trained on 1,102 employees | XGBoost Champion Model", use_container_width=True)
            else:
                st.info("SHAP summary plot not found. Run main.py to regenerate.")

            divider()
            subsection("Key SHAP Interpretations")

            insights = [
                ("Overtime",         COLORS["red"],   "Overtime is the single strongest driver of attrition. Employees working overtime have a 2.93x higher probability of departing than those who do not."),
                ("Monthly Income",   COLORS["amber"], "Low monthly income consistently produces large positive SHAP contributions, pushing predictions toward attrition. Bottom 20% salary band is the highest risk group."),
                ("Total Working Yrs",COLORS["blue"],  "Both very early-career and highly experienced employees show elevated risk — new hires due to instability, veterans due to ceiling effects."),
                ("Stock Options",    COLORS["amber"], "Employees with zero equity (StockOptionLevel=0) show strongly positive SHAP values. Financial ownership alignment is a key retention lever."),
                ("Job Satisfaction", COLORS["red"],   "Low job satisfaction is a reliable and consistent predictor across all departments, job levels, and seniority bands."),
                ("Burnout Risk Index",COLORS["red"],  "The engineered composite feature combining overtime, work-life balance risk, and job dissatisfaction ranks as the top overall predictor."),
            ]
            cols = st.columns(2)
            for i, (feat, color, text) in enumerate(insights):
                with cols[i%2]:
                    st.markdown(f"""
                    <div style="background:#FFFFFF;
                         border:1px solid #E5E7EB; border-left:3px solid {color};
                         border-radius:0 10px 10px 0; padding:16px 18px; margin-bottom:14px;
                         box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                        <div style="font-size:0.72rem; color:{color}; font-weight:700;
                             text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">{feat}</div>
                        <div style="color:#374151; font-size:0.87rem; line-height:1.6;">{text}</div>
                    </div>""", unsafe_allow_html=True)

    # ── TAB 3: Deep Dive ──────────────────────────────────────────────────────
    with tab3:
        spacer(24)
        if seg.empty:
            st.info("Segmented data not available.")
        else:
            c1, c2 = st.columns(2)
            for col, (grp_col, title, x_label, x_fmt) in zip(
                [c1,c2,c1,c2],
                [("overtime","Overtime vs No Overtime Attrition Rate","Overtime Status",None),
                 ("job_role","Attrition Rate by Job Role","Job Role",None),
                 ("job_satisfaction","Job Satisfaction vs Attrition Rate","Satisfaction Level","%.0%"),
                 ("stock_option_level","Stock Option Level vs Attrition Rate","Stock Option Level","%.0%")]
            ):
                with col:
                    with st.container(border=True):
                        subsection(title)
                        gdf = seg.groupby(grp_col).agg(Total=("attrition","count"),Dep=("attrition","sum")).reset_index()
                        gdf["Rate"] = gdf["Dep"]/gdf["Total"]

                        if grp_col == "overtime":
                            gdf["Label"] = gdf[grp_col].map({1:"Works Overtime",0:"No Overtime"})
                            fig = px.bar(gdf, x="Label", y="Rate",
                                         color="Label", color_discrete_map={"Works Overtime":COLORS["red"],"No Overtime":COLORS["green"]},
                                         text=gdf["Rate"].apply(lambda x: f"{x:.1%}"))
                            fig.update_traces(textposition="outside",textfont=dict(color="#374151"))
                            fig.update_layout(showlegend=False,
                                xaxis=dict(showgrid=False,title=""),
                                yaxis=dict(showgrid=True,gridcolor="#F3F4F6",tickformat=".0%"))
                        elif grp_col == "job_role":
                            gdf = gdf.sort_values("Rate",ascending=True)
                            fig = px.bar(gdf, x="Rate", y=grp_col, orientation="h",
                                         color="Rate",color_continuous_scale=[[0,COLORS["green"]],[1,COLORS["red"]]],
                                         text=gdf["Rate"].apply(lambda x: f"{x:.1%}"))
                            fig.update_traces(textposition="outside",textfont=dict(color="#374151"))
                            fig.update_layout(coloraxis_showscale=False,
                                xaxis=dict(showgrid=False,showticklabels=False,title=""),
                                yaxis=dict(showgrid=False,title=""))
                        else:
                            fig = px.line(gdf, x=grp_col, y="Rate", markers=True)
                            fig.update_traces(line_color=COLORS["blue"],marker_color=COLORS["amber"],marker_size=10,line_width=3)
                            fig.update_layout(
                                xaxis=dict(showgrid=True,gridcolor="#F3F4F6",title=x_label),
                                yaxis=dict(showgrid=True,gridcolor="#F3F4F6",tickformat=".0%"))

                        fig.update_layout(
                            paper_bgcolor="white",plot_bgcolor="white",
                            font=dict(color="#374151",family="Inter"),
                            margin=dict(t=10,b=10,l=5,r=70),height=260,
                            transition_duration=300,
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=13,
                                font_family="Inter, sans-serif"
                            )
                        )
                        st.plotly_chart(fig,use_container_width=True, config={"displayModeBar": False})

    # ── TAB 4: Model Validation ───────────────────────────────────────────────
    with tab4:
        spacer(24)
        with st.container(border=True):
            info_card("Four models were trained and evaluated using 5-fold stratified cross-validation on an imbalanced dataset "
                      "(16.1% base attrition rate). F1-Score — not Accuracy — was the selection criterion because it balances "
                      "precision and recall for rare-event prediction.", COLORS["blue"])

            models = [
                {"Model":"XGBoost","Acc":0.8152,"Prec":0.4328,"Rec":0.4915,"F1":0.4603,"AUC":0.7779,"champ":True},
                {"Model":"Random Forest","Acc":0.8397,"Prec":0.5000,"Rec":0.4237,"F1":0.4587,"AUC":0.7868,"champ":False},
                {"Model":"Logistic Regression","Acc":0.8614,"Prec":0.6333,"Rec":0.3220,"F1":0.4270,"AUC":0.8226,"champ":False},
                {"Model":"Gradient Boosting","Acc":0.8614,"Prec":0.6667,"Rec":0.2712,"F1":0.3855,"AUC":0.7990,"champ":False},
            ]
            col_r, col_m = st.columns([3,2])

            with col_r:
                subsection("Model Comparison Radar Chart")
                cats = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
                model_colors = [COLORS["blue"],COLORS["green"],COLORS["amber"],"#8B5CF6"]
                fig = go.Figure()
                for m, color in zip(models, model_colors):
                    vals = [m["Acc"],m["Prec"],m["Rec"],m["F1"],m["AUC"]]
                    vals_closed = vals + [vals[0]]
                    cats_closed = cats + [cats[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=vals_closed, theta=cats_closed, fill="toself",
                        name=m["Model"], line=dict(color=color, width=3 if m["champ"] else 1.5),
                        opacity=0.9 if m["champ"] else 0.45,
                    ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="white",
                        radialaxis=dict(visible=True,range=[0,1],gridcolor="#F3F4F6",
                                        tickfont=dict(color="#6B7280",size=10)),
                        angularaxis=dict(gridcolor="#F3F4F6",tickfont=dict(color="#374151",size=12)),
                    ),
                    paper_bgcolor="white",
                    font=dict(color="#374151",family="Inter"),
                    legend=dict(bgcolor="white",bordercolor="#E5E7EB",borderwidth=1),
                    margin=dict(t=30,b=20), height=400,
                    transition_duration=300,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=13,
                        font_family="Inter, sans-serif"
                    )
                )
                st.plotly_chart(fig,use_container_width=True, config={"displayModeBar": False})

            with col_m:
                subsection("Model Performance Cards")
                for m in models:
                    border_color = COLORS["blue"] if m["champ"] else "#E5E7EB"
                    badge = '<span style="background:linear-gradient(135deg,#1B4332,#059669);color:#fff;border-radius:4px;padding:2px 8px;font-size:0.65rem;font-weight:700;margin-left:8px;letter-spacing:0.05em;">CHAMPION</span>' if m["champ"] else ""
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid {border_color};
                         border-top:3px solid {COLORS['blue'] if m['champ'] else '#E5E7EB'};
                         border-radius:10px;padding:16px 18px;margin-bottom:12px;
                         box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                        <div style="font-weight:700;color:#1A1A1A;font-size:0.95rem;margin-bottom:12px;">
                            {m['Model']}{badge}
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:0.8rem;">
                            <div><div style="color:#6B7280;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;">F1-Score</div>
                                 <div style="color:{COLORS['blue'] if m['champ'] else '#374151'};font-weight:800;font-size:1.1rem;">{m['F1']:.3f}</div></div>
                            <div><div style="color:#6B7280;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;">Recall</div>
                                 <div style="color:#374151;font-weight:700;">{m['Rec']:.3f}</div></div>
                            <div><div style="color:#6B7280;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;">AUC</div>
                                 <div style="color:#374151;font-weight:700;">{m['AUC']:.3f}</div></div>
                            <div><div style="color:#6B7280;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;">Accuracy</div>
                                 <div style="color:#6B7280;font-weight:600;">{m['Acc']:.3f}</div></div>
                            <div><div style="color:#6B7280;font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;">Precision</div>
                                 <div style="color:#6B7280;font-weight:600;">{m['Prec']:.3f}</div></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            existing_plots = [(p,n) for p,n in [
                (os.path.join(FIG_DIR,"11_confusion_matrix.png"),"Confusion Matrix"),
                (os.path.join(FIG_DIR,"12_roc_curve.png"),"ROC Curve"),
                (os.path.join(FIG_DIR,"13_precision_recall_curve.png"),"Precision-Recall Curve"),
            ] if os.path.exists(p)]
            if existing_plots:
                divider()
                subsection("Evaluation Plots from Training Run")
                pcols = st.columns(len(existing_plots))
                for pc,(path,name) in zip(pcols,existing_plots):
                    pc.image(path,caption=name,use_container_width=True)
