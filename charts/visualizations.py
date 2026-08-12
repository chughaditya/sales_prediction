"""
visualizations.py
-------------------
All interactive Plotly chart builders live here. Every function takes a
DataFrame (and sometimes column names) and returns a `plotly.graph_objects`
figure that the page module can render with `st.plotly_chart(fig)`.

Keeping every chart as its own pure function (input -> figure, no
Streamlit calls inside) means:
    - Charts are testable outside of Streamlit
    - Charts can be reused across pages (e.g. same "sales trend" chart
      on Home and on EDA Dashboard)
    - Export-to-image logic can call these functions directly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except ImportError:
    st = None

# A consistent, professional colour palette used across every chart so the
# whole dashboard feels designed rather than default-matplotlib.
PALETTE_LIGHT = px.colors.qualitative.Set2
PALETTE_DARK = ["#A78BFA", "#818CF8", "#60A5FA", "#34D399", "#FBBF24", "#F87171", "#F472B6", "#2DD4BF"]


def _current_theme() -> str:
    if st is not None:
        try:
            return st.session_state.get("theme", "light")
        except Exception:
            return "light"
    return "light"


def _template() -> str:
    return "plotly_dark" if _current_theme() == "dark" else "plotly_white"


def _palette():
    return PALETTE_DARK if _current_theme() == "dark" else PALETTE_LIGHT


# Backward-compatible module-level names (evaluated once at import time as a
# sane default); prefer _template()/_palette() inside chart functions so the
# active theme is always respected.
TEMPLATE = "plotly_white"
PALETTE = PALETTE_LIGHT


def _empty_figure(message: str):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(template=_template(), height=300)
    return fig


def sales_trend_chart(df: pd.DataFrame, date_col: str, target_col: str):
    if date_col not in df.columns:
        return _empty_figure(f"Column '{date_col}' not found for trend chart.")
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    daily = temp.groupby(temp[date_col].dt.date)[target_col].sum().reset_index()
    fig = px.line(daily, x=date_col, y=target_col, markers=True,
                   title="Sales Trend Over Time", template=_template(),
                   color_discrete_sequence=_palette())
    fig.update_traces(line=dict(width=2))
    fig.update_layout(hovermode="x unified")
    return fig


def monthly_sales_chart(df: pd.DataFrame, date_col: str, target_col: str):
    if date_col not in df.columns:
        return _empty_figure(f"Column '{date_col}' not found for monthly chart.")
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp["Month"] = temp[date_col].dt.to_period("M").astype(str)
    monthly = temp.groupby("Month")[target_col].sum().reset_index()
    fig = px.bar(monthly, x="Month", y=target_col, title="Monthly Sales",
                 template=_template(), color=target_col, color_continuous_scale="Blues")
    return fig


def category_sales_chart(df: pd.DataFrame, category_col: str, target_col: str, title="Product-wise Sales"):
    if category_col not in df.columns:
        return _empty_figure(f"Column '{category_col}' not found.")
    grouped = df.groupby(category_col)[target_col].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(grouped, x=category_col, y=target_col, title=title,
                 template=_template(), color=category_col, color_discrete_sequence=_palette())
    fig.update_layout(showlegend=False)
    return fig


def region_sales_pie(df: pd.DataFrame, region_col: str, target_col: str):
    if region_col not in df.columns:
        return _empty_figure(f"Column '{region_col}' not found.")
    grouped = df.groupby(region_col)[target_col].sum().reset_index()
    fig = px.pie(grouped, names=region_col, values=target_col, title="Region-wise Sales Share",
                 template=_template(), color_discrete_sequence=_palette(), hole=0.4)
    return fig


def correlation_heatmap(corr_df: pd.DataFrame):
    if corr_df.empty:
        return _empty_figure("No numeric columns available for correlation.")
    fig = px.imshow(corr_df, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdBu_r", title="Correlation Heatmap",
                     template=_template(), zmin=-1, zmax=1)
    return fig


def box_plot(df: pd.DataFrame, column: str, group_col: str = None):
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found.")
    fig = px.box(df, y=column, x=group_col, title=f"Box Plot - {column}",
                 template=_template(), color=group_col, color_discrete_sequence=_palette())
    return fig


def histogram_chart(df: pd.DataFrame, column: str, bins: int = 30):
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found.")
    fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}",
                        template=_template(), color_discrete_sequence=_palette(), marginal="box")
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_figure("Selected columns not found.")
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}",
                      template=_template(), color_discrete_sequence=_palette(),
                      trendline="ols" if df[x_col].dtype != object else None)
    return fig


def pair_plot(df: pd.DataFrame, columns: list, color_col: str = None):
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return _empty_figure("Select at least 2 numeric columns for a pair plot.")
    fig = px.scatter_matrix(df, dimensions=cols, color=color_col, template=_template(),
                             color_discrete_sequence=_palette(), title="Pair Plot")
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    return fig


def feature_distribution_chart(df: pd.DataFrame, column: str):
    """KDE-style smoothed distribution using a histogram + density overlay."""
    if column not in df.columns:
        return _empty_figure(f"Column '{column}' not found.")
    fig = px.histogram(df, x=column, histnorm="probability density", nbins=40,
                        title=f"Feature Distribution - {column}", template=_template(),
                        color_discrete_sequence=_palette(), opacity=0.75)
    try:
        from scipy import stats
        values = df[column].dropna().values
        if len(values) > 5:
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode="lines",
                                      name="Density", line=dict(color="crimson", width=2)))
    except Exception:
        pass
    return fig


def feature_importance_chart(feature_names, importances, title="Feature Importance"):
    order = np.argsort(importances)[::-1]
    fig = px.bar(x=np.array(importances)[order], y=np.array(feature_names)[order],
                 orientation="h", title=title, template=_template(),
                 color=np.array(importances)[order], color_continuous_scale="Viridis",
                 labels={"x": "Importance", "y": "Feature"})
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return fig


def model_comparison_chart(results_df: pd.DataFrame, metric: str = "r2_score"):
    fig = px.bar(results_df.sort_values(metric, ascending=False), x="model_name", y=metric,
                 title=f"Model Comparison - {metric.upper()}", template=_template(),
                 color="model_name", color_discrete_sequence=_palette(), text_auto=".3f")
    fig.update_layout(showlegend=False)
    return fig


def actual_vs_predicted_chart(y_true, y_pred, title="Actual vs Predicted Sales"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                              marker=dict(color=_palette()[0], size=7, opacity=0.6),
                              name="Predictions"))
    min_v, max_v = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                              line=dict(color="red", dash="dash"), name="Perfect Fit"))
    fig.update_layout(title=title, xaxis_title="Actual Sales", yaxis_title="Predicted Sales",
                       template=_template())
    return fig


def residual_plot(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    fig = px.scatter(x=y_pred, y=residuals, template=_template(), color_discrete_sequence=_palette(),
                      labels={"x": "Predicted Sales", "y": "Residual"}, title="Residual Plot")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def error_distribution_chart(y_true, y_pred, title="Prediction Error Distribution"):
    """Histogram of (actual - predicted) errors -- shows whether the model's
    mistakes are roughly centered around zero and normally distributed."""
    errors = np.array(y_true) - np.array(y_pred)
    fig = px.histogram(x=errors, nbins=40, template=_template(), color_discrete_sequence=_palette(),
                        labels={"x": "Prediction Error (Actual - Predicted)"}, title=title)
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(yaxis_title="Count")
    return fig


def model_comparison_radar(results_df: pd.DataFrame):
    """Normalises R2, Adjusted R2, and inverted MAE/RMSE (so lower-is-better
    metrics also point 'outward = better') onto a 0-1 scale per metric, then
    plots one radar trace per model for an at-a-glance strengths overview."""
    metrics = ["r2_score", "adjusted_r2", "mae", "rmse"]
    df = results_df.copy()

    norm = pd.DataFrame({"model_name": df["model_name"]})
    for m in ["r2_score", "adjusted_r2"]:
        rng = df[m].max() - df[m].min()
        norm[m] = (df[m] - df[m].min()) / rng if rng > 0 else 1.0
    for m in ["mae", "rmse"]:
        rng = df[m].max() - df[m].min()
        # invert: lower error should map to a higher (better) radar value
        norm[m] = 1 - (df[m] - df[m].min()) / rng if rng > 0 else 1.0

    labels = ["R\u00b2", "Adjusted R\u00b2", "Low MAE", "Low RMSE"]
    fig = go.Figure()
    for i, row in norm.iterrows():
        values = [row[m] for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]],
            fill="toself", name=row["model_name"],
            line=dict(color=_palette()[i % len(_palette())])
        ))
    fig.update_layout(
        template=_template(), title="Model Performance Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    return fig
