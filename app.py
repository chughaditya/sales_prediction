"""
app.py
-------
Main entry point for AI Sales Prediction Pro.

Responsibilities of this file (kept deliberately thin):
    1. Configure the Streamlit page (title, icon, layout).
    2. Inject custom CSS + apply the selected light/dark theme.
    3. Render the sidebar navigation.
    4. Dispatch to the correct page module's `render()` function.

All actual page logic lives in `pages_app/*.py` so this file never grows
beyond a simple router. This separation is what makes the codebase
"modular" and easy to extend (adding a new page = one new file + one
line in the NAV_ITEMS dict below).

Run with:  streamlit run app.py
"""

import os
import streamlit as st

from utils.logger import get_logger
from pages_app import (
    home, dataset_analysis, eda_dashboard, train_model,
    prediction, prediction_history, settings
)

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")

st.set_page_config(
    page_title="AI Sales Prediction Pro",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- #
# Session state defaults
# --------------------------------------------------------------------- #
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --------------------------------------------------------------------- #
# CSS injection (base stylesheet + dynamic theme overrides)
# --------------------------------------------------------------------- #
def inject_css():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        base_css = f.read()

    font_import = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    </style>
    """

    if st.session_state.theme == "dark":
        theme_overrides = """
        <style>
        :root {
            --bg-primary: #080B14;
            --bg-secondary: #101826;
            --card-bg: rgba(20,28,45,.75);
            --grad: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #2563EB 100%);
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }
        .stApp { background: var(--bg-primary); color: #E8EBF5; }
        section[data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid rgba(124,58,237,.15);
        }
        .info-card { background: var(--card-bg) !important; border: 1px solid rgba(124,58,237,.2) !important;
            color: #E8EBF5; border-radius: 20px !important; backdrop-filter: blur(12px); }
        .section-header { color: #B9A6FF !important; }
        .section-subtext { color: #9AA3BE !important; }
        h1, h2, h3, h4, h5, h6, p, label, span, div { color: #E8EBF5; }
        .stDataFrame { background-color: var(--card-bg); border-radius: 16px; }
        .stTabs [data-baseweb="tab"] { color: #9AA3BE; }
        .stTabs [aria-selected="true"] { color: #B9A6FF !important; }

        /* Cards, buttons */
        .stButton button, .stDownloadButton button, button[kind="secondary"] {
            background: var(--card-bg) !important;
            border: 1px solid rgba(124,58,237,.25) !important;
            border-radius: 14px !important;
            transition: all .18s ease;
        }
        .stButton button:hover, .stDownloadButton button:hover, button[kind="secondary"]:hover {
            border-color: #7C3AED !important;
            box-shadow: 0 0 16px rgba(124,58,237,.35);
            transform: translateY(-1px);
        }
        .stButton button *, .stDownloadButton button *, button[kind] * { color: #E8EBF5 !important; }
        button[kind="primary"] {
            background: var(--grad) !important;
            border: none !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 20px rgba(79,70,229,.4);
            transition: all .18s ease;
        }
        button[kind="primary"]:hover {
            box-shadow: 0 6px 28px rgba(124,58,237,.55);
            transform: translateY(-1px);
        }
        button[kind="primary"] * { color: #FFFFFF !important; }

        /* Select / multiselect / dropdown inputs and their popup menus */
        div[data-baseweb="select"] > div {
            background: var(--card-bg) !important;
            border-color: rgba(124,58,237,.3) !important;
            color: #E8EBF5 !important;
        }
        div[data-baseweb="select"] * { color: #E8EBF5 !important; }

        /* Dropdown popover (portal rendered outside the normal app tree).
           Newer Streamlit/BaseWeb versions sometimes use <div> instead of
           <ul>/<li> for the menu, and put the text colour on a deeply
           nested inline-styled span, so we force background + colour on
           every possible container/level with high specificity. */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] div[data-baseweb="menu"],
        ul[data-baseweb="menu"],
        div[data-baseweb="menu"] {
            background: #101826 !important;
            border: 1px solid rgba(124,58,237,.25) !important;
        }
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li *,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] *,
        ul[data-baseweb="menu"] li,
        ul[data-baseweb="menu"] li *,
        div[data-baseweb="menu"] [role="option"],
        div[data-baseweb="menu"] [role="option"] * {
            background: #101826 !important;
            color: #E8EBF5 !important;
        }
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] [role="option"][aria-selected="true"],
        ul[data-baseweb="menu"] li:hover,
        div[data-baseweb="menu"] [role="option"]:hover {
            background: rgba(124,58,237,.25) !important;
        }
        div[data-baseweb="tag"] {
            background: var(--grad) !important; color: #FFFFFF !important;
        }
        div[data-baseweb="tag"] * { color: #FFFFFF !important; }

        /* Text / number inputs and text areas */
        input, textarea, .stNumberInput input, .stTextInput input {
            background: var(--card-bg) !important;
            color: #E8EBF5 !important;
            border-color: rgba(124,58,237,.3) !important;
        }

        /* Expander header */
        details summary, .streamlit-expanderHeader {
            background: var(--card-bg) !important;
            color: #E8EBF5 !important;
            border-radius: 14px !important;
        }

        /* Sidebar nav buttons */
        section[data-testid="stSidebar"] .stButton button { text-align: left !important; justify-content: flex-start !important; }

        /* Scaler selector cards */
        .scaler-card {
            background: var(--card-bg); border: 1px solid rgba(124,58,237,.2);
            border-radius: 20px; padding: 18px; margin-bottom: 8px; backdrop-filter: blur(10px);
            transition: all .18s ease;
        }
        .scaler-card.selected { border: 1.5px solid #7C3AED; box-shadow: 0 0 20px rgba(124,58,237,.3); }
        .scaler-icon { font-size: 26px; }
        .scaler-title { font-weight: 700; font-size: 16px; margin: 4px 0; }
        .scaler-desc { font-size: 12.5px; color: #9AA3BE; }

        /* Live training card */
        .live-training-card {
            background: var(--card-bg); border: 1px solid rgba(124,58,237,.25);
            border-radius: 20px; padding: 16px 20px; margin: 10px 0; backdrop-filter: blur(10px);
        }
        .lt-row { display: flex; justify-content: space-between; padding: 5px 0;
                  border-bottom: 1px solid rgba(255,255,255,.06); }
        .lt-row:last-child { border-bottom: none; }
        .lt-label { color: #9AA3BE; font-size: 13px; }
        .lt-value { font-weight: 700; font-size: 13px; background: var(--grad);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        /* Sidebar profile card */
        .profile-card {
            background: var(--card-bg); border: 1px solid rgba(124,58,237,.2);
            border-radius: 18px; padding: 12px 14px; display: flex; align-items: center; gap: 10px;
        }
        .profile-avatar { font-size: 26px; }
        .profile-name { font-weight: 700; font-size: 13px; }
        .profile-role { font-size: 11px; color: #9AA3BE; }

        /* --------------------------------------------------------- */
        /* Premium Prediction Page polish (glassmorphism, animations) */
        /* --------------------------------------------------------- */
        .glass-card, .metric-tile, .insight-card {
            background: var(--card-bg) !important;
            border: 1px solid rgba(124,58,237,.22) !important;
            backdrop-filter: blur(14px);
        }
        .glass-card:hover { box-shadow: 0 8px 28px rgba(124,58,237,.18); }
        .metric-tile:hover { box-shadow: 0 10px 26px rgba(124,58,237,.28); border-color: rgba(124,58,237,.5) !important; }
        .metric-violet .metric-tile-value, .metric-tile.metric-violet .metric-tile-value { color: #B9A6FF; }
        .metric-blue .metric-tile-value { color: #7FB4FF; }
        .metric-teal .metric-tile-value { color: #5EEAD4; }
        .metric-amber .metric-tile-value { color: #FBBF24; }
        .loading-pulse { background: var(--card-bg) !important; border: 1px solid rgba(124,58,237,.25); }

        /* Tabs on the Prediction page -> pill style */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
        .stTabs [data-baseweb="tab"] {
            background: var(--card-bg); border-radius: 12px !important; padding: 10px 18px !important;
            border: 1px solid rgba(124,58,237,.18); font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: var(--grad) !important; color: #FFFFFF !important; border-color: transparent !important;
        }
        .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

        /* Sliders */
        .stSlider [data-baseweb="slider"] > div > div { background: var(--grad) !important; }
        </style>
        """
    else:
        theme_overrides = """
        <style>
        .stApp { background-color: #FFFFFF; }
        </style>
        """

    st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
    st.markdown(font_import, unsafe_allow_html=True)
    st.markdown(theme_overrides, unsafe_allow_html=True)


inject_css()

# --------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------- #
NAV_ITEMS = {
    "Home": ("\U0001F3E0", home),
    "Dataset Analysis": ("\U0001F4C2", dataset_analysis),
    "EDA Dashboard": ("\U0001F4CA", eda_dashboard),
    "Train Model": ("\U0001F9E0", train_model),
    "Prediction": ("\U0001F3AF", prediction),
    "Prediction History": ("\U0001F551", prediction_history),
    "Settings": ("\u2699\ufe0f", settings),
}

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 6px 0 4px 0;">
            <div style="font-size:34px;">\U0001F4C8</div>
            <div style="font-size:19px; font-weight:800; color:#4A6CF7;">Sales Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    for name, (icon, _) in NAV_ITEMS.items():
        is_active = st.session_state.selected_page == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}", width='stretch',
                     type="primary" if is_active else "secondary"):
            st.session_state.selected_page = name
            st.rerun()

    selected_page = st.session_state.selected_page

    st.divider()
    st.markdown(
        """
        <div class="profile-card">
            <div class="profile-avatar">\U0001F464</div>
            <div>
                <div class="profile-name">Aditya Chugh</div>
                <div class="profile-role">Data Science Engineer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------- #
# Page dispatch
# --------------------------------------------------------------------- #
try:
    _, page_module = NAV_ITEMS[selected_page]
    page_module.render()
except Exception as exc:
    logger.exception(f"Unhandled error while rendering page '{selected_page}'")
    st.error(f"Something went wrong while loading this page: {exc}")
    st.caption("Check `logs/app.log` for the full traceback.")

st.markdown(
    '<div class="app-footer">AI Sales Prediction Pro &copy; 2026 &mdash; '
    'Built for portfolio &amp; production demonstration purposes.</div>',
    unsafe_allow_html=True,
)
