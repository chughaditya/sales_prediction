"""
ui_components.py
------------------
Small reusable HTML/Streamlit snippets used by the "classic" pages
(Home, Dataset Analysis, EDA Dashboard, Train Model, Settings). The
newer glassmorphism Prediction page has its own component set in
components/glass_ui.py -- kept separate on purpose so neither has to
worry about breaking the other's look.
"""

import streamlit as st

_VARIANT_COLORS = {
    "default": ("#7C3AED", "rgba(124,58,237,.12)"),
    "accent": ("#2563EB", "rgba(37,99,235,.12)"),
    "warning": ("#F59E0B", "rgba(245,158,11,.12)"),
    "danger": ("#EF4444", "rgba(239,68,68,.12)"),
    "success": ("#10B981", "rgba(16,185,129,.12)"),
}


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin: 6px 0 14px 0;">
            <div class="section-header" style="font-size:22px; font-weight:800;">{title}</div>
            {f'<div class="section-subtext" style="font-size:13px; margin-top:2px;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", variant: str = "default"):
    color, bg = _VARIANT_COLORS.get(variant, _VARIANT_COLORS["default"])
    st.markdown(
        f"""
        <div class="info-card" style="padding:16px 18px; border-radius:16px;">
            <div style="font-size:12px; opacity:.7; text-transform:uppercase; letter-spacing:.4px;">{label}</div>
            <div style="font-size:26px; font-weight:800; color:{color}; margin:4px 0;">{value}</div>
            {f'<div style="font-size:12px; opacity:.65;">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str, icon: str = "\U0001F4ED"):
    st.markdown(
        f"""
        <div class="info-card" style="text-align:center; padding:40px 20px; border-radius:20px;">
            <div style="font-size:34px; margin-bottom:10px;">{icon}</div>
            <div style="font-size:14px; opacity:.75; max-width:480px; margin:0 auto;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
