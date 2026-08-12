"""
eda_dashboard.py
------------------
Interactive Exploratory Data Analysis dashboard. Every chart is built
from the raw (unencoded) dataset so labels stay human-readable
(e.g. "North" instead of "0"), and rendered with Plotly for interactivity.
"""

import streamlit as st
import pandas as pd

from charts import visualizations as viz
from utils.ui_components import section_header, empty_state
from preprocessing.preprocessor import DataPreprocessor


def _guess_by_keyword(df, keywords, dtype_filter=None):
    cols = df.columns if dtype_filter is None else df.select_dtypes(include=dtype_filter).columns
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in keywords):
            return c
    return None


def _guess_date_column(df):
    # First try obvious name matches (fast, no data scanning)
    col = _guess_by_keyword(df, ["date", "time", "period", "month", "year"])
    if col:
        return col
    # Fall back: sample a small number of rows per object column and skip
    # long free-text columns (e.g. "Description") which are never dates and
    # are expensive to run through the datetime parser at full row count.
    sample = df.head(300) if len(df) > 300 else df
    for c in df.select_dtypes(include="object").columns:
        try:
            avg_len = sample[c].astype(str).str.len().mean()
            if avg_len is not None and avg_len > 25:
                continue  # unlikely to be a date, skip expensive parse
            parsed = pd.to_datetime(sample[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return c
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def _cached_column_guesses(df, target_hint):
    guessed_date = _guess_date_column(df)
    guessed_target = target_hint or _guess_by_keyword(
        df, ["sales", "revenue", "amount", "total", "price"], dtype_filter="number"
    )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if guessed_target is None and numeric_cols:
        guessed_target = numeric_cols[0]
    guessed_region = _guess_by_keyword(df, ["region", "state", "zone", "area", "location"], dtype_filter="object")
    guessed_product = _guess_by_keyword(df, ["product", "item", "category", "sku"], dtype_filter="object")
    return guessed_date, guessed_target, guessed_region, guessed_product


@st.cache_data(show_spinner=False)
def _cached_correlation_matrix(df):
    return DataPreprocessor.correlation_matrix(df)


def render():
    section_header("EDA Dashboard", "Interactive visual exploration of your dataset")

    if "raw_df" not in st.session_state:
        empty_state("Please upload or select a dataset on the 'Dataset Analysis' page first.")
        return

    df = st.session_state.raw_df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    guessed_date, guessed_target, guessed_region, guessed_product = _cached_column_guesses(
        df, st.session_state.get("target_column")
    )

    with st.expander("Column mapping (auto-detected \u2014 change if needed)", expanded=False):
        m1, m2 = st.columns(2)
        with m1:
            date_col = st.selectbox(
                "Date column", [None] + df.columns.tolist(),
                index=(df.columns.tolist().index(guessed_date) + 1) if guessed_date in df.columns.tolist() else 0
            )
            target_col = st.selectbox(
                "Target / sales column", [None] + numeric_cols,
                index=(numeric_cols.index(guessed_target) + 1) if guessed_target in numeric_cols else 0
            )
        with m2:
            region_col = st.selectbox(
                "Region column (optional)", [None] + cat_cols,
                index=(cat_cols.index(guessed_region) + 1) if guessed_region in cat_cols else 0
            )
            product_col = st.selectbox(
                "Product column (optional)", [None] + cat_cols,
                index=(cat_cols.index(guessed_product) + 1) if guessed_product in cat_cols else 0
            )

    tabs = st.tabs([
        "Sales Trend", "Monthly Sales", "Product / Region", "Correlation Heatmap",
        "Box Plot", "Histogram", "Scatter Plot", "Pair Plot", "Feature Distribution"
    ])

    with tabs[0]:
        if date_col and target_col:
            st.plotly_chart(viz.sales_trend_chart(df, date_col, target_col), width='stretch')
        else:
            empty_state("Need a date column and a numeric target column for the trend chart.")

    with tabs[1]:
        if date_col and target_col:
            st.plotly_chart(viz.monthly_sales_chart(df, date_col, target_col), width='stretch')
        else:
            empty_state("Need a date column and a numeric target column for the monthly chart.")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            if product_col and target_col:
                st.plotly_chart(viz.category_sales_chart(df, product_col, target_col), width='stretch')
            else:
                empty_state("No product-like column detected.")
        with c2:
            if region_col and target_col:
                st.plotly_chart(viz.region_sales_pie(df, region_col, target_col), width='stretch')
            else:
                empty_state("No region-like column detected.")

    with tabs[3]:
        corr_df = _cached_correlation_matrix(df)
        st.plotly_chart(viz.correlation_heatmap(corr_df), width='stretch')

    with tabs[4]:
        col = st.selectbox("Numeric column", numeric_cols, key="box_col")
        group = st.selectbox("Group by (optional)", [None] + cat_cols, key="box_group")
        st.plotly_chart(viz.box_plot(df, col, group), width='stretch')

    with tabs[5]:
        col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
        bins = st.slider("Number of bins", 10, 100, 30)
        st.plotly_chart(viz.histogram_chart(df, col, bins), width='stretch')

    with tabs[6]:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with c2:
            y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
        with c3:
            color_col = st.selectbox("Color by (optional)", [None] + cat_cols, key="scatter_color")
        st.plotly_chart(viz.scatter_plot(df, x_col, y_col, color_col), width='stretch')

    with tabs[7]:
        selected = st.multiselect("Select numeric columns (2-5 recommended)", numeric_cols,
                                   default=numeric_cols[:4])
        color_col = st.selectbox("Color by (optional)", [None] + cat_cols, key="pair_color")
        if selected:
            st.plotly_chart(viz.pair_plot(df, selected, color_col), width='stretch')
        else:
            empty_state("Select at least 2 columns.")

    with tabs[8]:
        col = st.selectbox("Feature", numeric_cols, key="dist_col")
        st.plotly_chart(viz.feature_distribution_chart(df, col), width='stretch')
