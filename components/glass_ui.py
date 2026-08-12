"""
glass_ui.py
-------------
Reusable glassmorphism-styled HTML snippets for the premium Prediction
page. Kept separate from utils/ui_components.py (which serves the older
generic pages) so the new modules have full creative freedom without
touching the pages that already work.
"""

import time
import streamlit as st


def module_hero(icon: str, title: str, subtitle: str, gradient: str = "linear-gradient(120deg, #7C3AED 0%, #4F46E5 50%, #2563EB 100%)"):
    st.markdown(
        f"""
        <div class="glass-hero" style="background:{gradient};">
            <div class="glass-hero-icon">{icon}</div>
            <div>
                <div class="glass-hero-title">{title}</div>
                <div class="glass-hero-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_card_start(extra_class: str = ""):
    st.markdown(f'<div class="glass-card {extra_class}">', unsafe_allow_html=True)


def glass_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def metric_tile(icon: str, label: str, value: str, sub: str = "", accent: str = "violet"):
    st.markdown(
        f"""
        <div class="metric-tile metric-{accent}">
            <div class="metric-tile-icon">{icon}</div>
            <div class="metric-tile-label">{label}</div>
            <div class="metric-tile-value">{value}</div>
            <div class="metric-tile-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str):
    level_l = level.lower()
    color = {"low": "#10B981", "medium": "#F59E0B", "high": "#EF4444"}.get(level_l, "#9AA3BE")
    bg = {"low": "rgba(16,185,129,.15)", "medium": "rgba(245,158,11,.15)", "high": "rgba(239,68,68,.15)"}.get(level_l, "rgba(154,163,190,.15)")
    return f'<span class="risk-badge" style="color:{color};background:{bg};border:1px solid {color}55;">{level} Risk</span>'


def insight_card(title: str, points: list, icon: str = "\U0001F4A1"):
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-card-title">{icon} {title}</div>
            <ul class="insight-list">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_loading(messages: list = None):
    messages = messages or ["Crunching the numbers...", "Consulting the model...", "Almost there..."]
    placeholder = st.empty()
    for msg in messages:
        placeholder.markdown(
            f"""
            <div class="loading-pulse">
                <div class="loading-spinner"></div>
                <span>{msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.35)
    placeholder.empty()


def result_headline(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="result-headline">
            <div class="result-headline-label">{label}</div>
            <div class="result-headline-value">{value}</div>
            <div class="result-headline-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
