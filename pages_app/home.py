"""
home.py
--------
Premium enterprise-style landing dashboard: aurora animated hero with
live stat counters, KPI grid, activity/usage charts, recent activity
timeline, quick-access cards, and live model/engine status.

Data is derived entirely from utils/database.py (module_predictions,
predictions, model_performance) plus which model bundles exist on disk
-- no mock/fake numbers once real usage exists, but sensible zero-states
before that.
"""

import json
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

from utils.database import get_db
from utils.helpers import format_number, artifact_exists
from utils.ui_components import section_header
from charts.premium_charts import BASE_LAYOUT, MUTED_COLOR, TEXT_COLOR, GRID_COLOR

MODULE_LABELS = {
    "company_sales": ("\U0001F3E2", "Company Sales"),
    "retail_sales": ("\U0001F6D2", "Retail Sales"),
    "house_price": ("\U0001F3E0", "House Price"),
}


# ===================================================================== #
# Aurora hero with animated (JS count-up) live stats
# ===================================================================== #
def _aurora_hero(total_predictions: int, active_models: int, avg_confidence: float, today_count: int):
    html = f"""
        <style>
            body {{ margin: 0; }}
            .aurora-hero {{
                position: relative; overflow: hidden; border-radius: 28px;
                padding: 44px 44px 40px 44px; margin: 0;
                background: radial-gradient(circle at 20% 20%, rgba(124,58,237,.55), transparent 55%),
                            radial-gradient(circle at 80% 0%, rgba(37,99,235,.5), transparent 50%),
                            radial-gradient(circle at 60% 90%, rgba(23,195,178,.4), transparent 55%),
                            linear-gradient(120deg, #1B1035 0%, #241a4a 45%, #10192f 100%);
                box-shadow: 0 20px 50px rgba(79,70,229,.32);
                color: white; font-family: 'Inter', -apple-system, sans-serif;
            }}
            .aurora-hero::before {{
                content: ""; position: absolute; inset: -40%;
                background: conic-gradient(from 0deg, rgba(124,58,237,.25), rgba(37,99,235,.2), rgba(23,195,178,.22), rgba(124,58,237,.25));
                animation: spin 18s linear infinite; opacity: .55;
            }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
            .inner {{ position: relative; z-index: 1; }}
            .eyebrow {{
                display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: 1px;
                text-transform: uppercase; padding: 5px 14px; border-radius: 999px;
                background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
                margin-bottom: 14px; backdrop-filter: blur(6px);
            }}
            .title {{ font-size: 34px; font-weight: 800; margin-bottom: 22px; line-height: 1.15; }}
            .subtitle {{ font-size: 15px; opacity: .88; max-width: 640px; margin-bottom: 26px; }}
            .stats {{ display: flex; gap: 40px; flex-wrap: wrap; }}
            .stat-value {{ font-size: 30px; font-weight: 800; line-height: 1.1; }}
            .stat-label {{ font-size: 12px; opacity: .75; text-transform: uppercase; letter-spacing: .5px; margin-top: 3px; }}
        </style>
        <div class="aurora-hero">
            <div class="inner">
                <span class="eyebrow">\u26A1 AI Analytics Platform &middot; Live</span>
                <div class="title">Welcome back \U0001F44B</div>
                <div class="stats">
                    <div><div class="stat-value" id="s1">0</div><div class="stat-label">Total Predictions</div></div>
                    <div><div class="stat-value" id="s2">0</div><div class="stat-label">Active Models</div></div>
                    <div><div class="stat-value" id="s3">0%</div><div class="stat-label">Avg Confidence</div></div>
                    <div><div class="stat-value" id="s4">0</div><div class="stat-label">Today's Predictions</div></div>
                </div>
            </div>
        </div>
        <script>
            function countUp(id, target, suffix, duration) {{
                const el = document.getElementById(id);
                const start = performance.now();
                function tick(now) {{
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const value = Math.round(eased * target);
                    el.textContent = value.toLocaleString() + suffix;
                    if (progress < 1) requestAnimationFrame(tick);
                }}
                requestAnimationFrame(tick);
            }}
            countUp("s1", {total_predictions}, "", 900);
            countUp("s2", {active_models}, "", 700);
            countUp("s3", {avg_confidence:.0f}, "%", 1000);
            countUp("s4", {today_count}, "", 700);
        </script>
        """
    components.html(html, height=290)


# ===================================================================== #
# Data assembly
# ===================================================================== #
def _load_all_predictions(db):
    """Combines module_predictions (3 specialized models) with the legacy
    generic `predictions` table into one normalized list of dicts."""
    rows = []
    for r in db.get_module_predictions(limit=5000):
        rows.append({
            "source": r["module"],
            "value": r["predicted_value"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
            "input_json": r["input_json"],
        })
    for r in db.get_prediction_history(limit=5000):
        rows.append({
            "source": "generic_model",
            "value": r["predicted_sales"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
            "input_json": r["input_data"],
        })
    return rows


def _active_model_count():
    count = 0
    for f in ("company_sales_bundle.pkl", "retail_sales_bundle.pkl", "house_price_bundle.pkl"):
        if artifact_exists(f):
            count += 1
    if artifact_exists("best_model.pkl"):
        count += 1
    return count


# ===================================================================== #
# Charts
# ===================================================================== #
def _monthly_trend_chart(rows):
    if not rows:
        df = pd.DataFrame({"month": [], "count": []})
    else:
        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at"])
        df["month"] = df["created_at"].dt.strftime("%b %Y")
        monthly = df.groupby("month").size().reset_index(name="count")
        # keep chronological order
        order = df.sort_values("created_at")["month"].unique()
        monthly["month"] = pd.Categorical(monthly["month"], categories=order, ordered=True)
        monthly = monthly.sort_values("month")
        df = monthly

    fig = go.Figure(go.Scatter(
        x=df["month"], y=df["count"], mode="lines+markers",
        line=dict(color="#7C3AED", width=3, shape="spline"),
        marker=dict(size=7, color="#4F46E5"),
        fill="tozeroy", fillcolor="rgba(124,58,237,.12)",
    ))
    fig.update_layout(title=dict(text="Monthly Prediction Trend", font=dict(size=14, color=MUTED_COLOR)),
                       height=300, xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR),
                       yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title="Predictions"), **BASE_LAYOUT)
    return fig


def _model_usage_chart(rows):
    labels_map = {**{k: v[1] for k, v in MODULE_LABELS.items()}, "generic_model": "Generic Model"}
    if not rows:
        names, values = [], []
    else:
        df = pd.DataFrame(rows)
        counts = df["source"].value_counts()
        names = [labels_map.get(s, s) for s in counts.index]
        values = counts.values.tolist()

    fig = go.Figure(go.Pie(
        labels=names, values=values, hole=0.6,
        marker=dict(colors=["#7C3AED", "#4F46E5", "#2563EB", "#17C3B2"]),
        textfont=dict(color=TEXT_COLOR, size=11),
    ))
    fig.update_layout(title=dict(text="Model Usage", font=dict(size=14, color=MUTED_COLOR)),
                       height=300, legend=dict(font=dict(color=MUTED_COLOR, size=10)), **BASE_LAYOUT)
    return fig


def _distribution_chart(rows):
    values = [r["value"] for r in rows if r.get("value") is not None]
    fig = go.Figure(go.Histogram(
        x=values, marker=dict(color="#4F46E5", line=dict(color="#7C3AED", width=1)), nbinsx=20,
    ))
    fig.update_layout(title=dict(text="Prediction Distribution", font=dict(size=14, color=MUTED_COLOR)),
                       height=300, xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title="Predicted value"),
                       yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title="Count"), **BASE_LAYOUT)
    return fig


def _industry_breakdown_chart(rows):
    industries = []
    for r in rows:
        if r["source"] != "company_sales":
            continue
        try:
            data = json.loads(r["input_json"]) if isinstance(r["input_json"], str) else r["input_json"]
            if data and data.get("Industry"):
                industries.append(data["Industry"])
        except Exception:
            continue

    if industries:
        counts = pd.Series(industries).value_counts().head(8)
        names, values = counts.index.tolist(), counts.values.tolist()
    else:
        names, values = [], []

    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker=dict(color=values, colorscale=[[0, "#2563EB"], [1, "#7C3AED"]]),
    ))
    fig.update_layout(title=dict(text="Industry Breakdown (Company Sales)", font=dict(size=14, color=MUTED_COLOR)),
                       height=300, xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR),
                       yaxis=dict(color=MUTED_COLOR), **BASE_LAYOUT)
    return fig


def _revenue_forecast_chart(rows):
    company_rows = [r for r in rows if r["source"] == "company_sales" and r.get("value") is not None]
    if len(company_rows) < 2:
        df = pd.DataFrame({"x": [], "y": []})
        forecast_x, forecast_y = [], []
    else:
        df = pd.DataFrame(company_rows)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at"]).sort_values("created_at")
        df["idx"] = range(len(df))
        # simple linear trend extrapolation for a lightweight "forecast"
        import numpy as np
        coeffs = np.polyfit(df["idx"], df["value"], 1) if len(df) >= 2 else [0, df["value"].mean()]
        future_idx = list(range(len(df), len(df) + 5))
        forecast_y = [coeffs[0] * i + coeffs[1] for i in future_idx]
        forecast_x = [f"+{i - len(df) + 1}" for i in future_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(df))), y=df["value"] if len(df) else [],
                              mode="lines+markers", name="Actual",
                              line=dict(color="#17C3B2", width=3)))
    if forecast_y:
        fig.add_trace(go.Scatter(x=list(range(len(df), len(df) + len(forecast_y))), y=forecast_y,
                                  mode="lines+markers", name="Forecast",
                                  line=dict(color="#F59E0B", width=3, dash="dash")))
    fig.update_layout(title=dict(text="Revenue Forecast (Company Sales)", font=dict(size=14, color=MUTED_COLOR)),
                       height=300, xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title="Prediction #"),
                       yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title="Sales ($K)"),
                       legend=dict(font=dict(color=MUTED_COLOR, size=10)), **BASE_LAYOUT)
    return fig


# ===================================================================== #
# Small UI blocks
# ===================================================================== #
def _kpi_row(rows, perf_history):
    total = len(rows)
    active_models = _active_model_count()
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for r in rows if str(r.get("created_at", "")).startswith(today_str))
    success_rate = (sum(1 for c in confidences if c >= 60) / len(confidences) * 100) if confidences else 0.0
    best_r2 = max((p["r2_score"] for p in perf_history), default=None)
    accuracy_text = f"{best_r2 * 100:.1f}%" if best_r2 is not None else "\u2014"
    health = "Operational" if active_models > 0 else "No Models Loaded"

    cards = [
        ("\U0001F4CA", "Total Predictions", format_number(total), "All-time", "violet"),
        ("\U0001F9E0", "Active Models", str(active_models), "Ready to serve", "blue"),
        ("\U0001F3AF", "Accuracy (Best Model)", accuracy_text, "R\u00b2 score", "teal"),
        ("\U0001F4C8", "Avg Confidence", f"{avg_confidence:.1f}%", "Across all predictions", "amber"),
        ("\U0001F4C5", "Today's Predictions", str(today_count), datetime.now().strftime("%d %b %Y"), "violet"),
        ("\u2705", "Success Rate", f"{success_rate:.0f}%", "Confidence \u2265 60%", "teal"),
        ("\U0001F5C3\uFE0F", "Datasets Uploaded", format_number(len(get_db().get_datasets())), "All-time", "blue"),
        ("\U0001F49A", "System Health", health, "Live engine status", "amber" if active_models == 0 else "teal"),
    ]

    for row_start in (0, 4):
        cols = st.columns(4)
        for col, (icon, label, value, sub, accent) in zip(cols, cards[row_start:row_start + 4]):
            with col:
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
        st.write("")


def _quick_access():
    section_header("Quick Access", "Jump straight into a workflow")
    items = [
        ("\U0001F4C2", "Dataset Analysis", "Upload & clean a CSV", "Dataset Analysis"),
        ("\U0001F9E0", "Train Model", "Train & auto-select best model", "Train Model"),
        ("\U0001F3AF", "Prediction", "Company / Retail / House modules", "Prediction"),
        ("\U0001F551", "Prediction History", "Search, filter, export", "Prediction History"),
    ]
    cols = st.columns(4)
    for col, (icon, title, sub, page) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="floating-card">
                    <div class="qa-icon">{icon}</div>
                    <div class="qa-title">{title}</div>
                    <div class="qa-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open \u2192", key=f"qa_{page}", width='stretch'):
                st.session_state.selected_page = page
                st.rerun()


def _recent_activity(rows):
    section_header("Recent Activity", "Latest predictions across all modules")
    if not rows:
        st.info("\U0001F4AD No activity yet. Make a prediction to see it appear here.")
        return

    labels_map = {**{k: v for k, v in MODULE_LABELS.items()}, "generic_model": ("\U0001F4C8", "Generic Model")}
    sorted_rows = sorted(rows, key=lambda r: str(r.get("created_at", "")), reverse=True)[:8]

    html_items = []
    now = datetime.now()
    for r in sorted_rows:
        icon, label = labels_map.get(r["source"], ("\U0001F4CC", r["source"]))
        try:
            ts = datetime.fromisoformat(r["created_at"])
            delta = now - ts
            if delta < timedelta(minutes=1):
                when = "just now"
            elif delta < timedelta(hours=1):
                when = f"{int(delta.total_seconds() // 60)} min ago"
            elif delta < timedelta(days=1):
                when = f"{int(delta.total_seconds() // 3600)} hr ago"
            else:
                when = ts.strftime("%d %b, %I:%M %p")
        except Exception:
            when = str(r.get("created_at", ""))

        value = r.get("value")
        value_text = f"{value:,.2f}" if isinstance(value, (int, float)) else "\u2014"
        html_items.append(
            f"""<div class="activity-item">
                    <div class="activity-dot"></div>
                    <div>
                        <div class="activity-text">{icon} <b>{label}</b> prediction generated &mdash; value {value_text}</div>
                        <div class="activity-time">{when}</div>
                    </div>
                </div>"""
        )
    st.markdown("".join(html_items), unsafe_allow_html=True)


def _model_status():
    section_header("Live AI Engine Status", "Which models are ready to serve predictions")
    checks = [
        ("Company Sales Model", "company_sales_bundle.pkl"),
        ("Retail Sales Model", "retail_sales_bundle.pkl"),
        ("House Price Model", "house_price_bundle.pkl"),
        ("Generic Dataset Model", "best_model.pkl"),
    ]
    for name, filename in checks:
        ready = artifact_exists(filename)
        pill_cls = "status-ready" if ready else "status-pending"
        pill_text = "Ready" if ready else "Not Trained"
        st.markdown(
            f"""
            <div class="model-status-row">
                <span class="model-status-name">{name}</span>
                <span class="model-status-pill {pill_cls}">{pill_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===================================================================== #
# Main render
# ===================================================================== #
def render():
    db = get_db()
    rows = _load_all_predictions(db)
    perf_history = db.get_model_performance_history()

    total = len(rows)
    active_models = _active_model_count()
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for r in rows if str(r.get("created_at", "")).startswith(today_str))

    _aurora_hero(total, active_models, avg_confidence, today_count)

    st.write("")
    _kpi_row(rows, perf_history)

    st.write("")
    _quick_access()

    st.write("")
    if total == 0:
        st.info(
            "\U0001F4AD **No predictions yet.** Charts below will populate automatically once you use the "
            "**Prediction** page. Head over to Company Sales / Retail Sales / House Price to get started."
        )
    else:
        section_header("Analytics Overview", "Trends across every prediction made in this environment")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_monthly_trend_chart(rows), width='stretch')
        with c2:
            st.plotly_chart(_model_usage_chart(rows), width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(_distribution_chart(rows), width='stretch')
        with c4:
            st.plotly_chart(_industry_breakdown_chart(rows), width='stretch')

        st.plotly_chart(_revenue_forecast_chart(rows), width='stretch')

    st.write("")
    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        _recent_activity(rows)
    with col_right:
        _model_status()
