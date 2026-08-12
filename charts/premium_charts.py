"""
premium_charts.py
--------------------
Plotly chart builders shared by all three Prediction page modules.
Every chart is transparent-background + light text so it drops cleanly
onto the app's dark glassmorphism cards without extra CSS overrides.
"""

import plotly.graph_objects as go

BRAND_GRADIENT = ["#7C3AED", "#4F46E5", "#2563EB"]
TEXT_COLOR = "#E8EBF5"
MUTED_COLOR = "#9AA3BE"
GRID_COLOR = "rgba(255,255,255,.08)"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def gauge_chart(value: float, min_val: float, max_val: float, title: str, suffix: str = ""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 30, "color": TEXT_COLOR}},
        title={"text": title, "font": {"size": 14, "color": MUTED_COLOR}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": MUTED_COLOR, "tickfont": {"color": MUTED_COLOR, "size": 10}},
            "bar": {"color": "#7C3AED", "thickness": 0.32},
            "bgcolor": "rgba(255,255,255,.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, min_val + (max_val - min_val) * 0.5], "color": "rgba(124,58,237,.10)"},
                {"range": [min_val + (max_val - min_val) * 0.5, min_val + (max_val - min_val) * 0.8], "color": "rgba(124,58,237,.20)"},
                {"range": [min_val + (max_val - min_val) * 0.8, max_val], "color": "rgba(124,58,237,.32)"},
            ],
        },
    ))
    fig.update_layout(height=220, **BASE_LAYOUT)
    return fig


def confidence_meter(confidence: float):
    color = "#10B981" if confidence >= 75 else ("#F59E0B" if confidence >= 50 else "#EF4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 30, "color": TEXT_COLOR}},
        title={"text": "Confidence Score", "font": {"size": 14, "color": MUTED_COLOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED_COLOR, "tickfont": {"color": MUTED_COLOR, "size": 10}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(255,255,255,.04)",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(height=220, **BASE_LAYOUT)
    return fig


def donut_breakdown(labels: list, values: list, title: str = ""):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=["#7C3AED", "#4F46E5", "#2563EB", "#17C3B2", "#F59E0B", "#EF4444", "#EC4899"]),
        textfont=dict(color=TEXT_COLOR, size=11),
        hoverinfo="label+percent+value",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=MUTED_COLOR)),
                       height=300, showlegend=True,
                       legend=dict(font=dict(color=MUTED_COLOR, size=10)), **BASE_LAYOUT)
    return fig


def trend_line(x: list, y: list, title: str = "", y_title: str = ""):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color="#7C3AED", width=3, shape="spline"),
        marker=dict(size=7, color="#4F46E5"),
        fill="tozeroy", fillcolor="rgba(124,58,237,.12)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=MUTED_COLOR)),
        height=300, xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title=y_title),
        **BASE_LAYOUT,
    )
    return fig


def bar_comparison(x: list, y: list, title: str = "", y_title: str = "", horizontal: bool = False):
    if horizontal:
        fig = go.Figure(go.Bar(
            y=x, x=y, orientation="h",
            marker=dict(color=y, colorscale=[[0, "#2563EB"], [1, "#7C3AED"]]),
        ))
        fig.update_layout(yaxis=dict(color=MUTED_COLOR), xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title=y_title))
    else:
        fig = go.Figure(go.Bar(
            x=x, y=y,
            marker=dict(color=y, colorscale=[[0, "#2563EB"], [1, "#7C3AED"]]),
        ))
        fig.update_layout(xaxis=dict(color=MUTED_COLOR), yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title=y_title))
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=MUTED_COLOR)), height=320, **BASE_LAYOUT)
    return fig


def feature_contribution_bar(positives: list, negatives: list, title: str = "Top Influencing Features"):
    """positives/negatives: list of (feature_name, value) tuples."""
    items = negatives + positives
    items.sort(key=lambda t: t[1])
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    colors = ["#EF4444" if v < 0 else "#10B981" for v in vals]
    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.2f}" for v in vals], textposition="outside", textfont=dict(color=TEXT_COLOR, size=10),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=MUTED_COLOR)),
        height=max(280, 34 * len(items)),
        xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, zeroline=True, zerolinecolor="rgba(255,255,255,.2)"),
        yaxis=dict(color=MUTED_COLOR),
        **BASE_LAYOUT,
    )
    return fig


def shap_waterfall(base_value: float, contributions: list, prediction: float, title: str = "Prediction Breakdown"):
    """contributions: list of (feature_name, shap_value) sorted by |value| desc."""
    contributions = list(reversed(contributions))
    measures = ["relative"] * len(contributions) + ["total"]
    labels = [f"{n}" for n, _ in contributions] + ["Final Prediction"]
    values = [v for _, v in contributions] + [prediction - base_value - sum(v for _, v in contributions)]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + measures,
        x=["Base Value"] + labels,
        y=[base_value] + values,
        connector={"line": {"color": "rgba(255,255,255,.2)"}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#7C3AED"}},
        textfont=dict(color=TEXT_COLOR, size=10),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=MUTED_COLOR)),
        height=380, xaxis=dict(color=MUTED_COLOR, tickangle=-30),
        yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR),
        showlegend=False, **BASE_LAYOUT,
    )
    return fig
