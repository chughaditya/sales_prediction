"""
settings.py
-------------
App-level settings: theme toggle, cache controls, and quick diagnostics
(which model bundles are present, database location, etc).
"""

import streamlit as st

from utils.ui_components import section_header


def render():
    section_header("Settings", "Appearance, cache, and system diagnostics")

    st.markdown("**Appearance**")
    theme = st.radio(
        "Theme", ["dark", "light"],
        index=0 if st.session_state.get("theme", "dark") == "dark" else 1,
        horizontal=True, key="settings_theme_radio",
    )
    if theme != st.session_state.get("theme"):
        st.session_state.theme = theme
        st.rerun()

    st.write("")
    with st.expander("\u26A0\uFE0F Reset session"):
        st.caption("Clears the in-memory session (uploaded dataset, form values, etc). Prediction history in the database is not affected.")
        if st.button("Reset current session", width='stretch'):
            for key in list(st.session_state.keys()):
                if key not in ("theme",):
                    del st.session_state[key]
            st.success("Session reset.")
            st.rerun()

    st.write("")
    st.caption("AI Sales Prediction Pro \u2014 built with Streamlit, scikit-learn, and Plotly.")
