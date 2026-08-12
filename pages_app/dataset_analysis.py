"""
dataset_analysis.py
---------------------
Upload page + automatic Data Quality Report, Data Summary, Feature
Statistics and Correlation Matrix. This is the first real step in the
pipeline: everything downstream (EDA, training, prediction) reads from
`st.session_state.raw_df`.
"""

import streamlit as st
import pandas as pd

from preprocessing.preprocessor import DataPreprocessor
from charts.visualizations import correlation_heatmap
from utils.ui_components import section_header, kpi_card, empty_state
from utils.helpers import DATASET_DIR, dataframe_to_csv_bytes
from utils.database import get_db

SAMPLE_PATH = f"{DATASET_DIR}/sample_sales_data.csv"


def _load_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


@st.cache_data(show_spinner=False)
def _load_sample_dataset():
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def _cached_data_summary(df):
    return DataPreprocessor.data_summary(df)


@st.cache_data(show_spinner=False)
def _cached_feature_statistics(df):
    return DataPreprocessor.feature_statistics(df)


@st.cache_data(show_spinner=False)
def _cached_correlation_matrix(df):
    return DataPreprocessor.correlation_matrix(df)


def render():
    section_header("Dataset Analysis", "Upload your sales data or use the bundled sample dataset")

    db = get_db()

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("Use Sample Dataset", width='stretch')

    if use_sample:
        st.session_state.raw_df = _load_sample_dataset()
        st.session_state.dataset_name = "sample_sales_data.csv"
        db.log_dataset("sample_sales_data.csv", *st.session_state.raw_df.shape)
        st.success("Sample dataset loaded successfully.")

    if uploaded_file is not None:
        try:
            df = _load_dataset(uploaded_file)
            st.session_state.raw_df = df
            st.session_state.dataset_name = uploaded_file.name
            db.log_dataset(uploaded_file.name, *df.shape)
            st.success(f"'{uploaded_file.name}' uploaded successfully \u2014 {df.shape[0]} rows, {df.shape[1]} columns.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

    if "raw_df" not in st.session_state:
        empty_state("Upload a CSV file above, or click 'Use Sample Dataset' to explore the app instantly.")
        return

    df = st.session_state.raw_df

    # ------------------------------------------------------------------ #
    # KPI row
    # ------------------------------------------------------------------ #
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Rows", f"{df.shape[0]:,}", st.session_state.get("dataset_name", ""))
    with c2:
        kpi_card("Columns", f"{df.shape[1]}", "", variant="accent")
    with c3:
        kpi_card("Missing Cells", f"{int(df.isna().sum().sum()):,}", "", variant="warning")
    with c4:
        kpi_card("Duplicate Rows", f"{int(df.duplicated().sum()):,}", "", variant="danger")

    st.write("")
    tabs = st.tabs(["Preview", "Data Quality Report", "Data Summary", "Feature Statistics", "Correlation Matrix"])

    with tabs[0]:
        st.dataframe(df.head(50), width='stretch')
        st.download_button(
            "Download Current Dataset (CSV)", data=dataframe_to_csv_bytes(df),
            file_name="dataset_preview.csv", mime="text/csv"
        )

    with tabs[1]:
        st.markdown("Run the automatic cleaning pipeline to see exactly what would change.")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        default_index = numeric_cols.index("SalesAmount") if "SalesAmount" in numeric_cols else 0
        target_col = st.selectbox(
            "Select target column (the value you want to predict)",
            options=numeric_cols,
            index=default_index,
        )
        st.session_state.target_column = target_col

        if st.button("Generate Data Quality Report"):
            with st.spinner("Analysing dataset..."):
                pre = DataPreprocessor(df, target_column=target_col)
                _, report = pre.run_full_pipeline()
                st.session_state.dq_report = report
                st.session_state.cleaned_preview_df = pre.df

        report = st.session_state.get("dq_report")
        if report:
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Initial Shape", f"{report['initial_shape'][0]} x {report['initial_shape'][1]}")
            with r2:
                st.metric("Final Shape", f"{report['final_shape'][0]} x {report['final_shape'][1]}")
            with r3:
                st.metric("Duplicates Removed", report["duplicates_removed"])

            st.markdown("**Missing Values Handled**")
            if report["missing_values_handled"]:
                st.json(report["missing_values_handled"])
            else:
                st.caption("No missing values found.")

            st.markdown("**Outliers Detected & Handled (IQR method)**")
            if report["outliers_detected"]:
                st.json(report["outliers_handled"])
            else:
                st.caption("No significant outliers detected.")

            st.markdown("**Categorical Columns Encoded**")
            st.write(", ".join(report["columns_encoded"]) if report["columns_encoded"] else "None")

            st.markdown("**Numeric Columns Scaled**")
            st.write(", ".join(report["columns_scaled"]) if report["columns_scaled"] else "None")
        else:
            empty_state("Click 'Generate Data Quality Report' to run the automated cleaning analysis.")

    with tabs[2]:
        st.dataframe(_cached_data_summary(df), width='stretch')

    with tabs[3]:
        stats_df = _cached_feature_statistics(df)
        if stats_df.empty:
            empty_state("No numeric columns found for statistics.")
        else:
            st.dataframe(stats_df, width='stretch')

    with tabs[4]:
        corr_df = _cached_correlation_matrix(df)
        if corr_df.empty:
            empty_state("Need at least 2 numeric columns to compute correlations.")
        else:
            st.plotly_chart(correlation_heatmap(corr_df), width='stretch')
