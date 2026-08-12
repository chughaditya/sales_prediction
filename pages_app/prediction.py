"""
prediction.py
---------------
Top-level "Prediction" page. Hosts three specialized forecasting
modules as tabs: Company Sales, Retail Sales, and House Price. Each tab
delegates entirely to its own module_*.render() function.
"""

import streamlit as st

from utils.ui_components import section_header
from pages_app import module_company, module_retail, module_house


def render():
    section_header("Prediction", "Choose a module and get an instant AI-powered forecast")

    tab_company, tab_retail, tab_house = st.tabs(
        ["\U0001F3E2 Company Sales", "\U0001F6D2 Retail Sales", "\U0001F3E0 House Price"]
    )

    with tab_company:
        module_company.render()
    with tab_retail:
        module_retail.render()
    with tab_house:
        module_house.render()
