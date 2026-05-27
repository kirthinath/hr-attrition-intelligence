import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_utils import inject_global_css, load_scorecard, sidebar_brand, COLORS

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_global_css()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:4rem 0 2.5rem;text-align:center;">
    <div style="display:inline-flex;align-items:center;gap:8px;
         background:rgba(27,67,50,0.05);border:1px solid rgba(27,67,50,0.15);
         border-radius:99px;padding:5px 16px;margin-bottom:1.8rem;">
        <div style="width:6px;height:6px;background:#1B4332;border-radius:50%;
             animation:pulse 2s infinite;"></div>
        <span style="font-size:0.72rem;color:#1B4332;font-weight:700;
             letter-spacing:0.15em;text-transform:uppercase;">
            Enterprise Analytics Platform &nbsp;·&nbsp; Powered by XGBoost
        </span>
    </div>
    <h1 style="font-size:3.2rem;font-weight:800;line-height:1.1;margin:0 auto 1.2rem;
         max-width:820px;background:linear-gradient(135deg,#1A1A1A 0%,#1B4332 100%);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         background-clip:text;letter-spacing:-0.03em;">
        HR Attrition Risk &amp;<br>Retention Intelligence
    </h1>
    <p style="color:#6B7280;font-size:1rem;max-width:580px;margin:0 auto;
         line-height:1.8;font-weight:400;font-style:italic;">
        Machine learning-powered workforce analytics. Predict flight risk,
        understand departure drivers, and act before talent walks out the door.
    </p>
</div>
<style>
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(1.3); }
}
</style>
""", unsafe_allow_html=True)

# ── Live Metric Pills ─────────────────────────────────────────────────────────
try:
    sc = load_scorecard()
    kpis = sc.get("workforce_kpis", {})
    pred = sc.get("predictive_risk_exposure", {})
    pills = [
        (f"{kpis.get('total_historical_headcount', 1470):,}", "Employees Analyzed", COLORS["blue"]),
        (f"{kpis.get('overall_attrition_rate', 0.161):.1%}", "Attrition Rate", COLORS["red"]),
        (f"{pred.get('active_high_risk_headcount', 18)}", "High-Risk Employees", COLORS["amber"]),
        ("77.8%", "Model AUC Score", COLORS["green"]),
    ]
except Exception:
    pills = [
        ("1,470", "Employees Analyzed", COLORS["blue"]),
        ("16.1%", "Attrition Rate", COLORS["red"]),
        ("18", "High-Risk Employees", COLORS["amber"]),
        ("77.8%", "Model AUC Score", COLORS["green"]),
    ]

# Use hex values directly for light backgrounds to prevent hard-coding dark colors
c_map = {
    COLORS["blue"]: ("#1B4332", "rgba(27,67,50,0.06)", "rgba(27,67,50,0.15)"),
    COLORS["red"]: ("#EF4444", "rgba(239,68,68,0.06)", "rgba(239,68,68,0.15)"),
    COLORS["amber"]: ("#F59E0B", "rgba(245,158,11,0.06)", "rgba(245,158,11,0.15)"),
    COLORS["green"]: ("#10B981", "rgba(16,185,129,0.06)", "rgba(16,185,129,0.15)"),
}

pill_html = "".join([f"""
<div style="background:{c_map.get(c, ('#1B4332','rgba(27,67,50,0.06)','rgba(27,67,50,0.15)'))[1]};
     border:1px solid {c_map.get(c, ('#1B4332','rgba(27,67,50,0.06)','rgba(27,67,50,0.15)'))[2]};
     border-radius:99px;padding:10px 22px;text-align:center;
     min-width:130px;flex:1;">
    <div style="font-size:1.4rem;font-weight:800;color:{c_map.get(c, ('#1B4332','','#'))[0]};line-height:1;">{v}</div>
    <div style="font-size:0.65rem;color:#6B7280;font-weight:600;
         letter-spacing:0.1em;text-transform:uppercase;margin-top:3px;">{l}</div>
</div>""" for v, l, c in pills])

st.markdown(f"""
<div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;
     margin-bottom:3rem;">
    {pill_html}
</div>""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #E5E7EB;margin:0 0 2.5rem;'/>",
            unsafe_allow_html=True)

# ── Navigate the Platform ─────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem;">
    <div style="font-size:0.68rem;color:#6B7280;letter-spacing:0.12em;
         text-transform:uppercase;font-weight:600;margin-bottom:6px;">Platform Modules</div>
    <h2 style="font-size:1.5rem;font-weight:700;color:#1A1A1A;margin:0;">
        Navigate the Platform
    </h2>
</div>""", unsafe_allow_html=True)

pages = [
    ("01", "Executive Command Center",   "Real-time workforce KPIs, risk distribution, and departmental attrition analysis for HR leadership.", COLORS["blue"]),
    ("02", "Live Risk Predictor",        "Enter employee attributes and get an instant XGBoost prediction with SHAP-driven explanations.",        COLORS["red"]),
    ("03", "Workforce Risk Intelligence","Browse and filter the complete risk-scored workforce. Bulk-upload CSVs for batch scoring.",              COLORS["amber"]),
    ("04", "Attrition Drivers",          "Feature importance, SHAP global analysis, deep-dive analytics, and full model validation metrics.",      COLORS["green"]),
    ("05", "Strategic Insights",         "Ten executive insights and a sequenced 90-day retention roadmap with projected financial impact.",        COLORS["blue"]),
]

c1, c2, c3 = st.columns(3)
cols = [c1, c2, c3, c1, c2]
for col, (num, title, desc, color) in zip(cols, pages):
    display_color = c_map.get(color, ("#1B4332", "", ""))[0]
    col.markdown(f"""
    <div style="background:#FFFFFF;
         border:1px solid #E5E7EB;border-left:3px solid {display_color};border-radius:12px;
         padding:22px 20px;margin-bottom:14px;height:148px;
         transition:transform 0.2s,box-shadow 0.2s;
         box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:1.5rem;font-weight:800;color:{display_color};
             margin-bottom:6px;font-variant-numeric:tabular-nums;">{num}</div>
        <div style="font-size:0.9rem;font-weight:700;color:#1A1A1A;margin-bottom:5px;">{title}</div>
        <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid #E5E7EB;margin:0 0 2.5rem;'/>",
            unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom:1.5rem;text-align:center;">
    <div style="font-size:0.68rem;color:#6B7280;letter-spacing:0.12em;
         text-transform:uppercase;font-weight:600;margin-bottom:6px;">Methodology</div>
    <h2 style="font-size:1.5rem;font-weight:700;color:#1A1A1A;margin:0;">
        How It Works
    </h2>
</div>""", unsafe_allow_html=True)

steps = [
    (COLORS["blue"],  "01", "Data & Feature Engineering",
     "1,470 IBM HR records cleaned, validated, and enriched with 12 composite risk features — burnout index, overtime risk score, career growth indicator, and more."),
    (COLORS["amber"], "02", "Model Training & Selection",
     "Four ML models trained with 5-fold stratified cross-validation. XGBoost selected as champion on F1-Score, the most appropriate metric for imbalanced attrition data."),
    (COLORS["green"], "03", "Risk Scoring & Recommendations",
     "Every active employee receives a calibrated probability score, risk tier classification, top SHAP drivers, and a specific HR retention action recommendation."),
]

sc1, sc2, sc3 = st.columns(3)
for scol, (color, num, title, desc) in zip([sc1, sc2, sc3], steps):
    display_color = c_map.get(color, ("#1B4332", "", ""))[0]
    scol.markdown(f"""
    <div style="background:#FFFFFF;
         border:1px solid #E5E7EB;border-top:3px solid {display_color};border-radius:12px;
         padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:1.8rem;font-weight:800;color:{display_color};
             margin-bottom:10px;opacity:0.6;">{num}</div>
        <div style="font-size:0.92rem;font-weight:700;color:#1A1A1A;margin-bottom:8px;">{title}</div>
        <div style="font-size:0.8rem;color:#6B7280;line-height:1.6;">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 0;color:#6B7280;font-size:0.78rem;">
    IBM HR Analytics Dataset (n=1,470) &nbsp;·&nbsp; XGBoost Champion Model &nbsp;·&nbsp;
    SHAP Explainability &nbsp;·&nbsp; Scikit-Learn Pipeline
</div>""", unsafe_allow_html=True)
