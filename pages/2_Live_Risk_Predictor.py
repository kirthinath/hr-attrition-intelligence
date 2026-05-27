import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, shap
from app_utils import (inject_global_css, section_header, subsection, info_card,
                       spacer, divider, sidebar_brand,
                       COLORS, RISK_COLORS, RISK_GRADIENTS, RISK_MESSAGES, BASE_DIR)

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_global_css()

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_attrition_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def encode_travel(val):
    return {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}.get(str(val), 1)

def compute_features(d):
    t = encode_travel(d["business_travel"])
    d["income_band"] = ("Low" if d["monthly_income"] < 3000 else
                        "Medium" if d["monthly_income"] < 6000 else
                        "High" if d["monthly_income"] < 12000 else "Executive")
    d["tenure_group"] = ("New Hire" if d["years_at_company"] <= 1 else
                         "Junior" if d["years_at_company"] <= 4 else
                         "Experienced" if d["years_at_company"] <= 9 else "Veteran")
    d["promotion_delay_risk"]         = d["years_since_last_promotion"] / (d["years_in_current_role"] + 1.0)
    d["overtime_risk_score"]          = d["overtime"] * (5 - d["work_life_balance"]) * (5 - d["job_involvement"])
    d["work_life_balance_risk"]       = (5 - d["work_life_balance"]) * (1 + t) + (d["distance_from_home"] / 10.0)
    d["satisfaction_composite_score"] = (d["environment_satisfaction"] + d["job_satisfaction"] + d["relationship_satisfaction"]) / 3.0
    d["manager_stability_score"]      = d["years_with_curr_manager"] / (d["years_at_company"] + 1.0)
    d["career_growth_risk_indicator"] = d["total_working_years"] / (d["job_level"] + 1.0)
    d["travel_burden_score"]          = t * d["distance_from_home"]
    d["burnout_risk_index"]           = d["overtime_risk_score"] + d["work_life_balance_risk"] + (5 - d["job_satisfaction"])
    return d

def retention_action(risk_level, factors):
    if risk_level == "Low":
        return ["Maintain standard career engagement and annual performance reviews.",
                "Continue monitoring satisfaction scores quarterly."]
    fs = " ".join(factors).lower()
    if "overtime" in fs or "burnout" in fs:
        return ["Implement immediate overtime cap of 15 hours/month.",
                "Distribute workload across team and schedule mandatory recovery days.",
                "Initiate manager-level wellness check-in within 7 days."]
    if "income" in fs or "salary" in fs or "rate" in fs:
        return ["Initiate off-cycle compensation review against market benchmarks.",
                "Consider a retention bonus or equity grant to create financial alignment."]
    if "promotion" in fs or "career" in fs:
        return ["Conduct career development roadmap meeting within 30 days.",
                "Define clear promotion milestones and document them formally."]
    if "satisfaction" in fs:
        return ["Schedule structured stay-interview to surface underlying concerns.",
                "Evaluate department culture fit or potential lateral transfer opportunity."]
    if "manager" in fs:
        return ["Facilitate skip-level leadership check-in within 2 weeks.",
                "Pair employee with a senior mentor outside the direct reporting line."]
    if "travel" in fs or "distance" in fs:
        return ["Transition employee to hybrid/remote arrangement immediately.",
                "Reduce business travel frequency by 25% for this employee."]
    return ["Schedule strategic check-in with HR Business Partner.",
            "Discuss career progression, expectations, and role alignment."]

# ── Page Header ───────────────────────────────────────────────────────────────
section_header("Live Attrition Risk Predictor",
               "Enter employee details below to receive a real-time risk score with SHAP-driven explanations.")

# ── Horizontal Full-Width Configuration Form ──────────────────────────────────
with st.form("live_predictor_form"):
    with st.container(border=True):
        st.markdown("""
        <div style="font-size:0.68rem;color:#1B4332;font-weight:700;
             letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;">
            Employee Profile Configuration
        </div>
        """, unsafe_allow_html=True)

        # Demographics
        st.markdown('<div style="font-size:0.65rem;color:#059669;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #E5E7EB;">Demographics</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        age       = d1.slider("Age", 18, 65, 35)
        gender    = d2.selectbox("Gender", ["Male", "Female"])
        marital   = d3.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        d4, d5 = st.columns(2)
        education = d4.slider("Education Level", 1, 5, 3, help="1=Below College · 5=Doctor")
        edu_field = d5.selectbox("Education Field", ["Life Sciences","Medical","Marketing",
                                 "Technical Degree","Human Resources","Other"])
        spacer(8)

        # Job Details
        st.markdown('<div style="font-size:0.65rem;color:#059669;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #E5E7EB;">Job Details</div>', unsafe_allow_html=True)
        j1, j2 = st.columns(2)
        department = j1.selectbox("Department", ["Sales","Research & Development","Human Resources"])
        job_role   = j2.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician",
                     "Manufacturing Director","Healthcare Representative","Manager",
                     "Sales Representative","Research Director","Human Resources"])
        j3, j4, j5 = st.columns(3)
        job_level  = j3.slider("Job Level", 1, 5, 2, help="1=Entry · 5=Executive")
        job_involv = j4.slider("Job Involvement", 1, 4, 3, help="1=Low · 4=High")
        job_sat    = j5.slider("Job Satisfaction", 1, 4, 2, help="1=Low · 4=High")
        spacer(8)

        # Compensation
        st.markdown('<div style="font-size:0.65rem;color:#059669;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #E5E7EB;">Compensation</div>', unsafe_allow_html=True)
        c1i, c2i = st.columns(2)
        monthly_income = c1i.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
        daily_rate     = c2i.number_input("Daily Rate", 100, 1500, 800, step=50)
        c3i, c4i, c5i = st.columns(3)
        hourly_rate    = c3i.number_input("Hourly Rate", 30, 100, 65, step=5)
        monthly_rate   = c4i.number_input("Monthly Rate", 2000, 27000, 14000, step=500)
        pct_hike       = c5i.slider("Last Salary Hike (%)", 10, 25, 15)
        stock_opt      = st.slider("Stock Option Level", 0, 3, 0, help="0=None · 3=High")
        spacer(8)

        # Work Conditions
        st.markdown('<div style="font-size:0.65rem;color:#059669;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #E5E7EB;">Work Conditions</div>', unsafe_allow_html=True)
        w1, w2 = st.columns(2)
        overtime        = w1.radio("Works Overtime?", ["Yes", "No"], horizontal=True)
        business_travel = w2.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
        w3, w4, w5 = st.columns(3)
        distance_home  = w3.slider("Distance from Home (mi)", 1, 29, 10)
        work_life_bal  = w4.slider("Work-Life Balance", 1, 4, 2, help="1=Bad · 4=Best")
        env_sat        = w5.slider("Environment Satisfaction", 1, 4, 2)
        w6, w7 = st.columns(2)
        rel_sat        = w6.slider("Relationship Satisfaction", 1, 4, 3)
        perf_rating    = w7.slider("Performance Rating", 3, 4, 3, help="3=Excellent · 4=Outstanding")
        spacer(8)

        # Career History
        st.markdown('<div style="font-size:0.65rem;color:#059669;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #E5E7EB;">Career History</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        total_yrs      = h1.slider("Total Working Years", 0, 40, 10)
        num_companies  = h2.slider("Companies Worked At", 0, 9, 3)
        yrs_company    = h3.slider("Years at Company", 0, 40, 5)
        h4c, h5c, h6c = st.columns(3)
        yrs_role       = h4c.slider("Years in Current Role", 0, 18, 3)
        yrs_promo      = h5c.slider("Years Since Promotion", 0, 15, 2)
        yrs_manager    = h6c.slider("Years with Manager", 0, 17, 3)
        training_times = st.slider("Training Sessions Last Year", 0, 6, 2)

    spacer(16)
    submitted = st.form_submit_button(
        "Run Risk Analysis →",
        width='stretch',
    )

# ── Placeholder / Results rendered below the form ─────────────────────────────
if not submitted:
    spacer(24)
    st.markdown(f"""
    <div style="background:#FFFFFF;
         border:1px solid #E5E7EB;border-radius:12px;padding:40px 24px;
         text-align:center;max-width:600px;margin:0 auto;
         box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:2.5rem;color:#E5E7EB;margin-bottom:16px;">◈</div>
        <div style="font-size:0.9rem;color:#6B7280;font-weight:500;">
            Results will appear here
        </div>
        <div style="font-size:0.8rem;color:#9CA3AF;margin-top:8px;font-style:italic;">
            Configure the profile and click<br><strong style="color:{COLORS['blue']};">Run Risk Analysis →</strong>
        </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# RESULTS — shown after submission (outside form)
# ════════════════════════════════════════════════════════════════════════════
if submitted:
    # Task 7: Loading Spinner with sequential clean messages
    with st.spinner("Analyzing employee profile..."):
        raw = {
            "age": age, "daily_rate": daily_rate, "distance_from_home": distance_home,
            "education": education, "environment_satisfaction": env_sat,
            "hourly_rate": hourly_rate, "job_involvement": job_involv,
            "job_level": job_level, "job_satisfaction": job_sat,
            "monthly_income": monthly_income, "monthly_rate": monthly_rate,
            "num_companies_worked": num_companies,
            "overtime": 1 if overtime == "Yes" else 0,
            "percent_salary_hike": pct_hike, "performance_rating": perf_rating,
            "relationship_satisfaction": rel_sat, "stock_option_level": stock_opt,
            "total_working_years": total_yrs, "training_times_last_year": training_times,
            "work_life_balance": work_life_bal, "years_at_company": yrs_company,
            "years_in_current_role": yrs_role, "years_since_last_promotion": yrs_promo,
            "years_with_curr_manager": yrs_manager, "business_travel": business_travel,
            "department": department, "education_field": edu_field,
            "gender": gender, "job_role": job_role, "marital_status": marital,
        }
        raw = compute_features(raw)
        input_df = pd.DataFrame([raw])

    with st.spinner("Running XGBoost prediction..."):
        pipeline = load_model()
        if pipeline is None:
            st.error("Model not found at models/best_attrition_model.pkl")
            st.stop()
        prob     = float(pipeline.predict_proba(input_df)[0, 1])
        risk     = "High" if prob >= 0.7 else ("Medium" if prob >= 0.3 else "Low")
        r_color  = RISK_COLORS[risk]
        r_grad   = RISK_GRADIENTS[risk]
        r_msg    = RISK_MESSAGES[risk]
        pct_int  = int(round(prob * 100))

    spacer(24)

    # ── Full-Width Risk Banner ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{r_grad};border:1px solid {r_color}33;
         border-radius:16px;padding:40px 48px;margin-bottom:2rem;
         box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
             flex-wrap:wrap;gap:24px;">
            <div>
                <div style="font-size:0.7rem;color:#6B7280;font-weight:700;
                     letter-spacing:0.18em;text-transform:uppercase;margin-bottom:10px;">
                    Attrition Probability Score
                </div>
                <div style="font-size:5rem;font-weight:800;color:{r_color};line-height:1;
                     letter-spacing:-0.03em;">{prob:.1%}</div>
                <div style="margin-top:10px;display:inline-block;background:{r_grad};
                     border:1px solid {r_color}44;border-radius:99px;
                     padding:4px 16px;font-size:0.8rem;font-weight:700;color:{r_color};
                     letter-spacing:0.1em;text-transform:uppercase;">{risk} RISK</div>
            </div>
            <div style="text-align:right;max-width:340px;">
                <div style="font-size:0.7rem;color:#6B7280;font-weight:700;
                     letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">
                    Contextual Analysis
                </div>
                <div style="font-size:0.9rem;color:#374151;
                     line-height:1.7;font-style:italic;">{r_msg}</div>
                <div style="margin-top:14px;font-size:0.72rem;color:#9CA3AF;">
                    XGBoost Champion &nbsp;·&nbsp; F1: 0.460 &nbsp;·&nbsp; AUC: 0.778
                </div>
            </div>
        </div>
        <div style="margin-top:24px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:0.68rem;color:#6B7280;font-weight:600;
                      letter-spacing:0.1em;text-transform:uppercase;">Risk Level</span>
                <span style="font-size:0.68rem;color:#374151;font-weight:700;">{pct_int}%</span>
            </div>
            <div style="background:#E5E7EB;border-radius:99px;height:8px;overflow:hidden;">
                <div style="width:{pct_int}%;background:{r_color};
                     height:100%;border-radius:99px;transition:width 0.8s ease;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
                <span style="font-size:0.65rem;color:#9CA3AF;">0% — Low</span>
                <span style="font-size:0.65rem;color:#9CA3AF;">30%</span>
                <span style="font-size:0.65rem;color:#9CA3AF;">70%</span>
                <span style="font-size:0.65rem;color:#9CA3AF;">100% — High</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP Full Width ───────────────────────────────────────────────────────
    top_risk_factors = []
    shap_ok = False
    sv_arr, feat_names_arr = None, None

    with st.spinner("Generating SHAP explanations..."):
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier   = pipeline.named_steps["classifier"]
            X_t = preprocessor.transform(input_df)
            if hasattr(X_t, "toarray"):
                X_t = X_t.toarray()
            feat_names_arr = [n.split("__")[-1].replace("_", " ").title()
                              for n in preprocessor.get_feature_names_out()]
            explainer = shap.TreeExplainer(classifier)
            sv = explainer.shap_values(X_t)
            if isinstance(sv, list):
                sv = sv[1][0] if len(sv) == 2 else sv[0][0]
            else:
                sv = sv[0]
                if len(sv.shape) == 2:
                    sv = sv[:, 1]
            sv_arr = sv
            top_idx  = np.argsort(np.abs(sv))[-14:][::-1]
            top_vals = sv[top_idx]
            top_names = [feat_names_arr[i] for i in top_idx]
            top_risk_factors = [top_names[i] for i, v in enumerate(top_vals) if v > 0][:3]
            shap_ok = True
        except Exception as e:
            top_risk_factors = ["Burnout Risk Index", "Overtime Risk Score", "Monthly Income"]

    with st.container(border=True):
        subsection("Feature Contribution Analysis (SHAP)")
        st.markdown('<p style="color:#6B7280;font-size:0.82rem;margin:-8px 0 16px;font-style:italic;">Red bars push the prediction toward attrition risk · Green bars push toward retention</p>', unsafe_allow_html=True)

        if shap_ok:
            colors_shap = [COLORS["red"] if v > 0 else COLORS["green"] for v in top_vals]
            fig = go.Figure(go.Bar(
                x=top_vals[::-1], y=top_names[::-1], orientation="h",
                marker=dict(
                    color=colors_shap[::-1],
                    opacity=0.85,
                ),
                text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in top_vals[::-1]],
                textposition="outside",
                textfont=dict(color="#374151", size=11),
            ))
            fig.add_vline(x=0, line_color="#E5E7EB", line_width=1.5)
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(color="#374151", family="Inter"),
                xaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False,
                           title=dict(text="← Reduces Risk  ·  SHAP Value  ·  Increases Risk →",
                                      font=dict(size=11, color="#6B7280"))),
                yaxis=dict(showgrid=False, title="", tickfont=dict(size=11, color="#374151")),
                margin=dict(t=10, b=10, l=10, r=90), height=400,
                transition_duration=300,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=13,
                    font_family="Inter, sans-serif"
                )
            )
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        else:
            st.info("SHAP unavailable for this prediction. Check model compatibility.")

    spacer(24)

    # ── Risk Drivers + Recommendations Row ───────────────────────────────────
    rc1, rc2 = st.columns([1, 2])

    with rc1:
        with st.container(border=True):
            subsection("Top Risk Drivers")
            factors = top_risk_factors or ["Burnout Risk Index", "Overtime Risk Score", "Monthly Income"]
            for i, factor in enumerate(factors):
                rank_colors = [COLORS["red"], COLORS["amber"], COLORS["amber"]]
                rc = rank_colors[i] if i < len(rank_colors) else COLORS["muted"]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:12px 14px;
                     background:#FFFFFF;
                     border:1px solid #E5E7EB;
                     border-left:3px solid {rc};
                     border-radius:10px;margin-bottom:10px;
                     box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                    <div style="background:{rc};color:#fff;border-radius:6px;width:26px;height:26px;
                         display:flex;align-items:center;justify-content:center;
                         font-weight:800;font-size:0.78rem;flex-shrink:0;">{i+1}</div>
                    <div style="color:#1A1A1A;font-weight:600;font-size:0.88rem;">{factor}</div>
                </div>""", unsafe_allow_html=True)

            # Gauge
            spacer(8)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": r_color, "size": 28, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6B7280",
                             "tickfont": {"color": "#6B7280", "size": 10}},
                    "bar": {"color": r_color, "thickness": 0.3},
                    "bgcolor": "#F3F4F6",
                    "bordercolor": "#E5E7EB",
                    "borderwidth": 1,
                    "steps": [{"range": [0, 30], "color": "rgba(16,185,129,0.08)"},
                               {"range": [30, 70], "color": "rgba(245,158,11,0.08)"},
                               {"range": [70, 100], "color": "rgba(239,68,68,0.08)"}],
                    "threshold": {"value": prob * 100,
                                  "line": {"color": r_color, "width": 2},
                                  "thickness": 0.85}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(color="#374151", family="Inter"),
                margin=dict(t=20, b=10, l=20, r=20), height=190,
                transition_duration=300,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=13,
                    font_family="Inter, sans-serif"
                )
            )
            st.plotly_chart(fig_g, width='stretch', config={"displayModeBar": False})

    with rc2:
        with st.container(border=True):
            subsection("Recommended HR Actions")
            rec_steps = retention_action(risk, factors)
            for i, step in enumerate(rec_steps):
                st.markdown(f"""
                <div style="display:flex;gap:14px;padding:14px 16px;
                     border:1px solid #E5E7EB;border-radius:10px;
                     background:#FFFFFF;margin-bottom:10px;
                     box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                    <div style="background:linear-gradient(135deg,#1B4332,#059669);
                         color:#fff;border-radius:7px;min-width:32px;height:32px;
                         display:flex;align-items:center;justify-content:center;
                         font-weight:800;font-size:0.82rem;flex-shrink:0;">
                        {str(i+1).zfill(2)}
                    </div>
                    <div>
                        <div style="color:#374151;font-size:0.88rem;line-height:1.6;">{step}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            spacer(12)
            # Timeline urgency badge
            urgency = {"High": ("Immediate", COLORS["red"]),
                       "Medium": ("30-Day Window", COLORS["amber"]),
                       "Low": ("Ongoing", COLORS["green"])}[risk]
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #E5E7EB;
                 border-radius:10px;padding:14px 16px;display:flex;
                 align-items:center;justify-content:space-between;
                 box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size:0.7rem;color:#6B7280;font-weight:600;
                     letter-spacing:0.1em;text-transform:uppercase;">Recommended Action Timeline</div>
                <div style="background:{urgency[1]}15;border:1px solid {urgency[1]};border-radius:99px;
                     padding:4px 14px;color:{urgency[1]};font-weight:700;font-size:0.8rem;">
                    {urgency[0]}
                </div>
            </div>""", unsafe_allow_html=True)

    spacer(24)
    with st.expander("View Full Employee Input Summary"):
        st.dataframe(
            pd.DataFrame([raw]).T.reset_index().rename(columns={"index": "Feature", 0: "Value"}),
            width='stretch', height=300
        )
