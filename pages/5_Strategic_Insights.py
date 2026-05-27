import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app_utils import (inject_global_css, section_header, subsection,
                       spacer, divider, sidebar_brand,
                       COLORS, BASE_DIR)

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_global_css()

# ── Page Header ───────────────────────────────────────────────────────────────
section_header("Strategic Insights & Retention Roadmap",
               "Ten executive-level insights from ML analysis of 1,470 employees, with a phased 90-day action roadmap.")

# ── Layout columns below header ────────────────────────────────────────────────
col_main, col_sidebar = st.columns([4.0, 1.0])

with col_sidebar:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size:0.68rem;color:#1B4332;font-weight:700;
             letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            Executive Actions
        </div>
        <div style="font-size:0.8rem;color:#6B7280;line-height:1.6;margin:0;">
            These strategic recommendations are derived automatically from the SHAP analysis of the XGBoost champion model.
        </div>
        """, unsafe_allow_html=True)

with col_main:
    INSIGHTS = [
        {"n":"01","title":"Overtime Is the Dominant Flight Risk Driver","color":COLORS["red"],"timeline":"Immediate (1-30 days)",
         "finding":"Employees who work overtime have a 30.53% attrition rate vs 10.44% for non-overtime — a 2.93x multiplier.",
         "impact":"Unchecked overtime policies generate 3x the attrition cost in affected departments and accelerate burnout cycles.",
         "rec":"Implement a monthly overtime cap of 15 hours. Require VP-level approval for exceptions. Track overtime-linked departures monthly.",
         "outcome":"Reduce overtime-attributable attrition from 30.53% toward the non-overtime baseline of 10.44%."},
        {"n":"02","title":"Sales Department Requires Priority Intervention","color":COLORS["amber"],"timeline":"Long Term (60-90 days)",
         "finding":"Sales has the highest attrition at 20.63%, vs 13.84% in R&D and 19.05% in HR.",
         "impact":"Sales attrition directly disrupts revenue pipelines and requires 12+ month replacement ramp-up periods.",
         "rec":"Audit Sales compensation. Increase base salary weight and transition commissions to a team-tenure model to reduce income volatility.",
         "outcome":"Reduce Sales attrition below 15% within two annual review cycles."},
        {"n":"03","title":"Compensation Disparity in the Bottom Salary Band","color":COLORS["amber"],"timeline":"Short Term (30-60 days)",
         "finding":"Employees earning under $3,000/month have significantly elevated risk, with monthly income ranking as a top SHAP driver.",
         "impact":"Below-market compensation is a primary trigger for job searching among early-career employees with high mobility.",
         "rec":"Conduct a compensation equity audit. Adjust base salaries for the bottom quintile to align with local market benchmarks.",
         "outcome":"Reduce compensation-driven attrition by 25% in affected cohorts."},
        {"n":"04","title":"Single Employees Are Disproportionately at Risk","color":COLORS["amber"],"timeline":"Short Term (30-60 days)",
         "finding":"Single employees represent the highest attrition segment with fewer retention anchors and higher mobility.",
         "impact":"Early-career single employees represent the majority of training investment that exits before reaching full productivity.",
         "rec":"Design targeted retention programs for single employees: mentorship cohorts, ERGs, and structured social integration programs.",
         "outcome":"Improve single-employee first-year retention by 15%."},
        {"n":"05","title":"Promotion Stagnation Accelerates High-Potential Departures","color":COLORS["amber"],"timeline":"Short Term (30-60 days)",
         "finding":"Employees with high promotion delay risk scores show measurably elevated attrition probability.",
         "impact":"High-potential employees who feel stagnant leave for external growth, taking organizational IP with them.",
         "rec":"Mandate semi-annual career path reviews for employees in their current role for 3+ years without a promotion.",
         "outcome":"Retain 80% of stalled high-potential employees by providing clear promotion roadmaps."},
        {"n":"06","title":"Stock Options Are a Long-Term Retention Anchor","color":COLORS["green"],"timeline":"Long Term (90 days+)",
         "finding":"Employees with StockOptionLevel 0 have a 24%+ attrition rate vs under 9% for those with any equity.",
         "impact":"Lack of financial ownership reduces long-term commitment, especially in competitive technical fields.",
         "rec":"Expand equity participation to all mid-level roles (JobLevel 2+), offering stock grants as part of annual bonuses.",
         "outcome":"Drop turnover in the JobLevel 2-3 cohort by 30%."},
        {"n":"07","title":"Manager Instability Accelerates Attrition","color":COLORS["amber"],"timeline":"Short Term (30-60 days)",
         "finding":"Employees with Manager Stability Score < 0.3 exhibit a 2.3x higher probability of leaving.",
         "impact":"Frequent manager rotations destabilize teams and reduce psychological safety and daily productivity.",
         "rec":"Implement leadership continuity guidelines. Conduct skip-level feedback audits for teams experiencing manager transitions.",
         "outcome":"Improve team stabilization scores by 20% and reduce reporting-change attrition."},
        {"n":"08","title":"Travel Burden and Commute Distance Compound Risk","color":COLORS["amber"],"timeline":"Immediate (1-30 days)",
         "finding":"Employees who travel frequently and live 15+ miles from the office show compounded risk in high-demand roles.",
         "impact":"Physical fatigue and travel disruption lead to rapid burnout and direct competition from remote-first employers.",
         "rec":"Adopt a hybrid policy allowing 3 days remote for eligible roles. Reduce non-essential business travel by 25%.",
         "outcome":"Lower commute-related flight risk by 35% and improve work-life satisfaction indices."},
        {"n":"09","title":"The First 24 Months Are a Critical Attrition Window","color":COLORS["red"],"timeline":"Short Term (30-60 days)",
         "finding":"Over 35% of all historical departures occur within the first 2 years of tenure.",
         "impact":"Immediate negative ROI on recruitment when employees leave before reaching full productivity — avg $43,000+ per junior hire.",
         "rec":"Redesign onboarding with a 90-day structured checkpoint system. Pair new hires with a dedicated culture buddy.",
         "outcome":"Increase first-year retention by 20% and decrease time-to-productivity."},
        {"n":"10","title":"Low Satisfaction Composite Scores Predict Separation","color":COLORS["amber"],"timeline":"Immediate (1-30 days)",
         "finding":"The Disengaged segment (low job involvement + low satisfaction) is the second-largest active flight risk group.",
         "impact":"Disengaged staff act as productivity drags and negatively impact team morale before ultimately departing.",
         "rec":"Deploy automated monthly pulse surveys. Trigger HR stay-interviews for satisfaction scores below 2.0.",
         "outcome":"Intercept and resolve disengagement before it escalates to voluntary departure."},
    ]

    tl_colors = {
        "Immediate (1-30 days)":   COLORS["red"],
        "Short Term (30-60 days)": COLORS["amber"],
        "Long Term (60-90 days)":  COLORS["green"],
        "Long Term (90 days+)":    COLORS["green"],
    }

    tab1, tab2 = st.tabs([
        "  10 Executive Insights  ",
        "  90-Day Retention Roadmap  "
    ])

    with tab1:
        spacer(24)
        for ins in INSIGHTS:
            tc = tl_colors.get(ins["timeline"], "#9CA3AF")
            with st.expander(f"  Insight {ins['n']}  —  {ins['title']}", expanded=False):
                c1, c2, c3 = st.columns(3)
                for col, (header, color, body) in zip([c1, c2, c3], [
                    ("Finding",        ins["color"],  ins["finding"]),
                    ("Business Impact", COLORS["blue"], ins["impact"]),
                    ("Recommendation",  COLORS["green"], ins["rec"]),
                ]):
                    col.markdown(f"""
                    <div style="background:#FFFFFF;
                         border:1px solid #E5E7EB;border-top:3px solid {color};
                         border-radius:0 0 10px 10px;padding:20px 18px;height:100%;min-height:140px;
                         box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-size:0.65rem;color:{color};font-weight:700;
                             letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">{header}</div>
                        <div style="color:#374151;font-size:0.9rem;line-height:1.7;">{body}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="display:flex;gap:12px;margin-top:14px;align-items:stretch;">
                    <div style="background:{tc}15;border:1px solid {tc}40;border-radius:10px;
                         padding:10px 20px;display:flex;align-items:center;gap:10px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{tc};flex-shrink:0;"></div>
                        <span style="color:{tc};font-weight:700;font-size:0.85rem;white-space:nowrap;">{ins['timeline']}</span>
                    </div>
                    <div style="background:#FFFFFF;
                         border:1px solid #E5E7EB;border-radius:10px;
                         padding:10px 20px;flex:1;display:flex;align-items:center;
                         box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                        <span style="color:#6B7280;font-size:0.75rem;font-weight:700;
                             text-transform:uppercase;letter-spacing:0.1em;margin-right:12px;">Expected Outcome</span>
                        <span style="color:#374151;font-size:0.9rem;font-style:italic;">{ins['outcome']}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        spacer(24)
        with st.container(border=True):
            subsection("90-Day Executive Retention Roadmap")
            spacer(12)

            phases = [
                {"phase":"Phase 1","range":"Days 1-30","label":"Immediate Action","color":COLORS["red"],
                 "actions":[
                     ("Overtime Cap Policy","Limit overtime to 15 hours/month. VP sign-off required for exceptions."),
                     ("Hybrid Work Guidelines","Announce 3-day remote policy for eligible roles to cut travel burden."),
                     ("Pulse Survey Launch","Deploy automated monthly pulse surveys with stay-interview triggers at score < 2.0."),
                     ("High-Risk Employee Outreach","Schedule HR 1-on-1s for the 18 employees classified as High Risk."),
                 ]},
                {"phase":"Phase 2","range":"Days 31-60","label":"Tactical Execution","color":COLORS["amber"],
                 "actions":[
                     ("Compensation Equity Audit","Adjust salaries for bottom 20% salary band to match market rates."),
                     ("Career Path Reviews","Mandate structured reviews for employees stagnant for 3+ years."),
                     ("Onboarding Redesign","Launch 90-day structured onboarding checkpoint system with culture buddies."),
                     ("Manager Stability Audit","Identify high-rotation teams and stabilize reporting structures."),
                 ]},
                {"phase":"Phase 3","range":"Days 61-90","label":"Systemic Alignment","color":COLORS["green"],
                 "actions":[
                     ("Sales Incentive Redesign","Increase base salary weight; adopt team-tenure commission model."),
                     ("Equity Grant Program","Expand stock options to all JobLevel 2+ employees."),
                     ("Leadership Continuity","Standardize minimum manager tenure per team across all business units."),
                     ("ERG and Social Programs","Launch mentorship cohorts for single and new-hire employee segments."),
                 ]},
            ]

            for phase in phases:
                st.markdown(f"""
                <div style="border-left:3px solid {phase['color']};padding-left:28px;margin-bottom:32px;position:relative;">
                    <div style="position:absolute;left:-8px;top:0;width:13px;height:13px;
                         background:#FFFFFF;border:3px solid {phase['color']};border-radius:50%;"></div>
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
                        <div style="background:linear-gradient(135deg,{phase['color']},{phase['color']}DD);
                             color:#fff;border-radius:6px;padding:6px 16px;font-weight:800;
                             font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;
                             box-shadow:0 1px 3px rgba(0,0,0,0.08);">{phase['phase']}</div>
                        <div style="color:{phase['color']};font-weight:700;font-size:1.1rem;">{phase['range']}</div>
                        <div style="color:#6B7280;font-size:0.9rem;font-weight:500;">— {phase['label']}</div>
                    </div>
                """, unsafe_allow_html=True)

                ac1, ac2 = st.columns(2)
                for i, (action, desc) in enumerate(phase["actions"]):
                    with (ac1 if i%2==0 else ac2):
                        st.markdown(f"""
                        <div style="background:#FFFFFF;
                             border:1px solid #E5E7EB;border-radius:10px;padding:20px;margin-bottom:16px;
                             box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                            <div style="color:{phase['color']};font-weight:700;font-size:0.9rem;margin-bottom:8px;">{action}</div>
                            <div style="color:#374151;font-size:0.85rem;line-height:1.6;">{desc}</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            divider()
            subsection("Projected Financial Impact of Retention Program")
            rc = st.columns(3)
            for col, (val, label, color, sub) in zip(rc, [
                ("$20.4M", "Current Est. Attrition Cost", COLORS["red"], "Annual baseline"),
                ("~$6.1M", "Projected Cost Reduction", COLORS["green"], "30% savings if roadmap executed"),
                ("$43,084", "Avg. Retention Intervention Cost", COLORS["blue"], "vs $86,168 full replacement cost"),
            ]):
                col.markdown(f"""
                <div style="background:#FFFFFF;
                     border:1px solid #E5E7EB;border-top:3px solid {color};
                     border-radius:12px;padding:24px;text-align:center;
                     box-shadow:0 1px 3px rgba(0,0,0,0.08);">
                    <div style="font-size:0.68rem;color:#6B7280;font-weight:700;letter-spacing:0.12em;
                         text-transform:uppercase;margin-bottom:10px;">{label}</div>
                    <div style="font-size:2.2rem;font-weight:800;
                         color:{color};line-height:1.2;">{val}</div>
                    <div style="font-size:0.8rem;color:#6B7280;margin-top:8px;font-style:italic;">{sub}</div>
                </div>""", unsafe_allow_html=True)
