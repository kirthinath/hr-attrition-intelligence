"""
Shared utilities — HR Attrition Intelligence Platform
Global CSS, color system, helpers, and cached data loaders.
"""
import os
import json
import streamlit as st
import pandas as pd

# ── Path Resolution ───────────────────────────────────────────────────────────
def get_base_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here) if os.path.basename(here) == "pages" else here

BASE_DIR = get_base_dir()

# ── Color System ──────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#F8F7F4",
    "card":        "#FFFFFF",
    "card_hover":  "#F9FAFB",
    "border":      "#E5E7EB",
    "border_dim":  "#F3F4F6",
    "blue":        "#1B4332", # Primary accent (dark teal green)
    "blue_light":  "#059669", # Secondary accent (medium green)
    "green":       "#10B981", # Positive
    "amber":       "#F59E0B", # Warning
    "red":         "#EF4444", # Negative
    "text":        "#1A1A1A", # Primary text
    "body":        "#374151", # Secondary text / dark gray
    "muted":       "#6B7280", # Gray
    "dim":         "#9CA3AF",
}

RISK_COLORS = {"High": "#DC2626", "Medium": "#D97706", "Low": "#16A34A"}

RISK_GRADIENTS = {
    "High":   "linear-gradient(135deg, #FEF2F2 0%, #FEF2F2 100%)",
    "Medium": "linear-gradient(135deg, #FFFBEB 0%, #FFFBEB 100%)",
    "Low":    "linear-gradient(135deg, #F0FDF4 0%, #F0FDF4 100%)",
}

RISK_MESSAGES = {
    "High":   "This employee shows critical burnout and flight-risk patterns requiring immediate HR attention.",
    "Medium": "Moderate risk indicators detected. Proactive engagement is recommended within 30 days.",
    "Low":    "Employee shows stable satisfaction and engagement patterns consistent with long-term retention.",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Task 9: Consistent Font across entire app */
* {
    box-sizing: border-box;
}

html, body, p, span:not(.notranslate):not([data-testid="stIconMaterial"]), li, a, label, h1, h2, h3, h4, h5, h6, input, button, select, textarea, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Restore Material Icons Font Family for Streamlit interface elements */
.notranslate, [data-testid="stIconMaterial"] {
    font-family: 'Material Icons', 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Symbols Sharp' !important;
}

/* Task 4: Clean Typography (line height & letter spacing) */
body, p, span, li, a, label {
    line-height: 1.6 !important;
}
h1, h2, h3, h4, h5, h6 {
    letter-spacing: 0.02em !important;
}

/* Ensure no text is too close to edges - padding and prevent layout shifts */
.main .block-container {
    padding-top: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 100% !important;
}

/* Task 1: Smooth Page Transitions */
.stApp {
    background-color: #F8F7F4 !important;
    animation: fadeIn 0.4s ease-in-out;
    overflow-x: hidden !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar & Task 6 ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    min-width: 240px !important;
    max-width: 240px !important;
    width: 240px !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}
section[data-testid="stSidebar"] label {
    color: #6B7280 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #1B4332, #059669) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    color: #374151 !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: #1B4332 !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover:not([aria-current="page"]) {
    background-color: #F0FDF4 !important;
    color: #1B4332 !important;
}
section[data-testid="stSidebar"] a:hover {
    background: #F0FDF4 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

/* ── Task 2: Smooth Card Hover Effects ── */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 20px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderContainer"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderContainer"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

div[data-testid="stMetricValue"] {
    color: #1B4332 !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
}
div[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricDelta"] > div {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

/* ── Headings ────────────────────────────── */
h1 {
    color: #1A1A1A !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
h2 {
    color: #1A1A1A !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
h3 {
    color: #1A1A1A !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h4 {
    color: #6B7280 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Tabs ────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 5px 6px !important;
    border: 1px solid #E5E7EB !important;
    gap: 4px !important;
    flex-wrap: wrap !important;
}
button[data-baseweb="tab"] {
    color: #6B7280 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    white-space: nowrap !important;
    transition: all 0.15s ease !important;
    background: transparent !important;
    border: none !important;
}
button[data-baseweb="tab"]:hover {
    color: #1A1A1A !important;
    background-color: #F3F4F6 !important;
}
button[aria-selected="true"] {
    background-color: #E8F5E9 !important;
    color: #1B4332 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #1B4332 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

/* ── Expanders ───────────────────────────── */
details {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    transition: box-shadow 0.2s ease !important;
}
details:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}
details summary {
    color: #1A1A1A !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 14px 18px !important;
    cursor: pointer !important;
}
details > div { padding: 0 18px 16px !important; }

/* ── Task 3: Smooth Button Interactions ── */
.stButton > button {
    background: #1B4332 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
}
.stButton > button:hover {
    background: #14532D !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(27,67,50,0.3) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stDownloadButton > button {
    background: #059669 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
}
.stDownloadButton > button:hover {
    background: #047857 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(4,120,87,0.2) !important;
}

/* ── Inputs ──────────────────────────────── */
input[type="number"], input[type="text"] {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 7px !important;
}
input:focus { border-color: #1B4332 !important; outline: none !important; }
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #E5E7EB !important;
    color: #1A1A1A !important;
    border-radius: 7px !important;
}

/* ── Dataframe ───────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── File Uploader ───────────────────────── */
div[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #E5E7EB !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Dividers ────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E5E7EB !important;
    margin: 1.5rem 0 !important;
}

/* ── Task 8: Clean Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #F8F7F4;
}
::-webkit-scrollbar-thumb {
    background: #D1D5DB;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF;
}

/* ── Alerts ──────────────────────────────── */
div[data-testid="stAlert"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    color: #374151 !important;
}

/* ── Progress bar ────────────────────────── */
div[data-testid="stProgress"] > div {
    background-color: #E5E7EB !important;
    border-radius: 99px !important;
}
div[data-testid="stProgress"] > div > div {
    border-radius: 99px !important;
}

/* Task 10: Remove Streamlit Branding & Chrome */
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Reset Streamlit Default Styling ── */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="column"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"] {
    gap: 16px !important;
}
.stContainer {
    background: transparent !important;
    border: none !important;
}
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}
.stSlider {
    padding: 0 !important;
}

/* Custom overrides for light theme requested by USER */
.stApp {
    background-color: #F8F7F4;
}
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px;
    transition: all 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stButton button {
    background-color: #1B4332;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(27,67,50,0.3);
}

/* Smooth professional transitions and micro-animations */

/* Smooth page fade in */
.main .block-container {
    animation: pageLoad 0.3s ease-out;
}
@keyframes pageLoad {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Smooth card hover lift */
div[data-testid="metric-container"] {
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
}

/* Smooth button press */
.stButton > button {
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(27,67,50,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Smooth tab transitions */
.stTabs [data-baseweb="tab"] {
    transition: all 0.2s ease !important;
}

/* Smooth select and slider */
.stSelectbox > div {
    transition: all 0.2s ease !important;
}

/* Smooth dataframe rows */
.stDataFrame tbody tr {
    transition: background 0.15s ease !important;
}
.stDataFrame tbody tr:hover {
    background: #F0FDF4 !important;
}

/* Smooth chart container */
.stPlotlyChart {
    transition: opacity 0.3s ease !important;
    animation: chartLoad 0.4s ease-out !important;
}
@keyframes chartLoad {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Smooth sidebar items */
section[data-testid="stSidebar"] a {
    transition: all 0.15s ease !important;
    border-radius: 6px !important;
}

/* Smooth scrollbar */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #D1D5DB;
    border-radius: 10px;
    transition: background 0.2s ease;
}
::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF;
}

/* Smooth expander */
.streamlit-expanderHeader {
    transition: all 0.2s ease !important;
}

</style>
"""

def inject_global_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Layout Helpers ────────────────────────────────────────────────────────────
def section_header(label: str, subtitle: str = ""):
    sub_html = (f'<p style="color:#6B7280;font-size:0.88rem;font-weight:400;'
                f'font-style:italic;margin:6px auto 0;line-height:1.6;max-width:800px;text-align:center;">{subtitle}</p>'
                if subtitle else "")
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;">
        <div style="font-size:0.68rem;color:#1B4332;letter-spacing:0.18em;
             text-transform:uppercase;font-weight:700;margin-bottom:6px;text-align:center;">
            HR Attrition & Talent Intelligence System
        </div>
        <h1 style="margin:0 auto;color:#1A1A1A;text-align:center;display:inline-block;padding-bottom:10px;border-bottom:1px solid #E5E7EB;width:100%;max-width:600px;">{label}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def subsection(label: str, color: str = "#1B4332"):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 1rem;padding-left:12px;border-bottom:1px solid #F3F4F6;padding-bottom:6px;">
        <div style="width:3px;height:16px;background:{color};border-radius:2px;flex-shrink:0;"></div>
        <div style="font-size:0.68rem;color:#6B7280;font-weight:700;
             letter-spacing:0.12em;text-transform:uppercase;">{label}</div>
    </div>""", unsafe_allow_html=True)


def card_open(accent: str = "#E5E7EB"):
    """Open a styled card div."""
    st.markdown(f"""
    <div style="background:#FFFFFF;
         border:1px solid #E5E7EB;border-top:3px solid {accent};
         border-radius:12px;padding:24px;margin-bottom:16px;
         box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    """, unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def info_card(text: str, border_color: str = "#1B4332"):
    st.markdown(f"""
    <div style="background:rgba(27,67,50,0.03);border:1px solid #E5E7EB;
         border-left:3px solid {border_color};border-radius:0 10px 10px 0;
         padding:14px 18px;margin-bottom:1.2rem;">
        <p style="color:#6B7280;font-size:0.86rem;line-height:1.7;margin:0;">{text}</p>
    </div>""", unsafe_allow_html=True)


def spacer(px: int = 24):
    st.markdown(f"<div style='height:{px}px;margin:0;padding:0;'></div>", unsafe_allow_html=True)


def divider():
    st.markdown(
        "<hr style='border:none;border-top:1px solid #E5E7EB;margin:1.5rem 0;'/>",
        unsafe_allow_html=True
    )


def sidebar_brand():
    st.sidebar.markdown("""
    <div style="padding:16px 0 20px;border-bottom:1px solid #E5E7EB;margin-bottom:20px;">
        <div style="color:#1B4332;font-size:1.05rem;font-weight:800;
             letter-spacing:-0.01em;">HR Intelligence</div>
        <div style="font-size:0.68rem;color:#059669;font-weight:600;
             letter-spacing:0.1em;text-transform:uppercase;margin-top:3px;">
             Powered by XGBoost</div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_section(label: str):
    st.sidebar.markdown(f"""
    <div style="font-size:0.65rem;color:#1B4332;font-weight:700;letter-spacing:0.12em;
         text-transform:uppercase;margin:16px 0 8px;padding-top:8px;
         border-top:1px solid #E5E7EB;">{label}</div>
    """, unsafe_allow_html=True)


# ── Cached Data Loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_scorecard(base=None):
    if base is None:
        base = BASE_DIR
    path = os.path.join(base, "data", "exports", "workforce_scorecard.json")
    if not os.path.exists(path):
        return {
            "workforce_kpis": {"total_historical_headcount": 1470,
                               "current_active_headcount": 1233,
                               "historical_departed_count": 237,
                               "overall_attrition_rate": 0.1612,
                               "overall_retention_rate": 0.8388},
            "financial_impact": {"total_estimated_attrition_cost": 20421738.0,
                                 "average_cost_per_departed_employee": 86167.67,
                                 "total_lifetime_value_loss_elv": 111842780.0,
                                 "average_active_employee_elv": 815725.38,
                                 "average_departed_employee_elv": 471910.46},
            "predictive_risk_exposure": {"active_high_risk_headcount": 18,
                                         "active_medium_risk_headcount": 244,
                                         "aggregate_workforce_risk_score": 0.196},
            "overtime_vulnerability": {"overtime_attrition_rate": 0.3053,
                                       "no_overtime_attrition_rate": 0.1044,
                                       "overtime_attrition_multiplier": 2.93},
            "financial_impact_summary": {},
            "departmental_attrition_rates": {"Human Resources": 0.1905,
                                              "Research & Development": 0.1384,
                                              "Sales": 0.2063},
        }
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_risk_scores(base=None):
    if base is None:
        base = BASE_DIR
    path = os.path.join(base, "data", "exports", "employee_attrition_risk_scores.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_segmented(base=None):
    if base is None:
        base = BASE_DIR
    path = os.path.join(base, "data", "processed", "segmented_hr_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
