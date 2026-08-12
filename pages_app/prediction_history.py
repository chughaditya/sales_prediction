"""
prediction_history.py
------------------------
Searchable, filterable history of every prediction made across the three
specialized modules (Company Sales, Retail Sales, House Price). Every
module now saves an optional free-text name (Company Name / Store /
Product Name / Property Name) alongside its inputs, so this page can
answer "what did I predict for X last time?" at a glance.
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.database import get_db
from utils.helpers import dataframe_to_csv_bytes
from utils.ui_components import section_header, empty_state, kpi_card
from charts.premium_charts import BASE_LAYOUT, MUTED_COLOR, TEXT_COLOR, GRID_COLOR

MODULE_LABELS = {
    "company_sales": ("\U0001F3E2", "Company Sales", "Company Name"),
    "retail_sales": ("\U0001F6D2", "Retail Sales", "Store/Product Name"),
    "house_price": ("\U0001F3E0", "House Price", "Property Name"),
    "generic_model": ("\U0001F4C8", "Generic Model", None),
}


def _load_all_rows(db) -> list:
    """Combines the specialized module_predictions table with the legacy
    generic `predictions` table into one unified list, so every prediction
    ever made in this app -- old or new -- shows up here."""
    rows = []
    for r in db.get_module_predictions(limit=5000):
        rows.append({
            "id": f"m{r['id']}",
            "module": r["module"],
            "input_json": r["input_json"],
            "result_json": r["result_json"],
            "predicted_value": r["predicted_value"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        })
    for r in db.get_prediction_history(limit=5000):
        rows.append({
            "id": f"g{r['id']}",
            "module": "generic_model",
            "input_json": r["input_data"],
            "result_json": json.dumps({"model_used": r.get("model_used")}, default=str),
            "predicted_value": r["predicted_sales"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        })
    return rows


def _extract_name(module: str, input_json: str) -> str:
    try:
        data = json.loads(input_json) if input_json else {}
    except Exception:
        return "\u2014"
    name_key = MODULE_LABELS.get(module, (None, None, None))[2]
    if name_key and name_key in data:
        return str(data[name_key]) or "\u2014"
    return "\u2014"


def _rows_to_dataframe(rows: list) -> pd.DataFrame:
    records = []
    for r in rows:
        icon, label, _ = MODULE_LABELS.get(r["module"], ("\U0001F4CC", r["module"], None))
        records.append({
            "ID": r["id"],
            "Module": f"{icon} {label}",
            "module_key": r["module"],
            "Name": _extract_name(r["module"], r["input_json"]),
            "Predicted Value": r["predicted_value"],
            "Confidence (%)": r["confidence"],
            "Created At": r["created_at"],
            "input_json": r["input_json"],
            "result_json": r["result_json"],
        })
    return pd.DataFrame(records)


def _trend_chart(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["Created At"] = pd.to_datetime(plot_df["Created At"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Created At"]).sort_values("Created At")
    fig = go.Figure(go.Scatter(
        x=plot_df["Created At"], y=plot_df["Predicted Value"], mode="lines+markers",
        line=dict(color="#7C3AED", width=3, shape="spline"),
        marker=dict(size=7, color="#4F46E5"),
        fill="tozeroy", fillcolor="rgba(124,58,237,.12)",
    ))
    fig.update_layout(
        title=dict(text="Predicted Value Over Time (filtered results)", font=dict(size=14, color=MUTED_COLOR)),
        height=300, xaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, color=MUTED_COLOR, title="Predicted Value"),
        **BASE_LAYOUT,
    )
    return fig


def render():
    section_header("Prediction History", "Search, filter, and export every forecast you've generated")

    db = get_db()
    all_rows = _load_all_rows(db)

    if not all_rows:
        empty_state(
            "No predictions yet. Head to the Prediction page \u2014 Company Sales, Retail Sales, or House "
            "Price \u2014 to generate your first forecast. Every prediction (with the name you give it) will "
            "show up here automatically, including anything predicted with the older Generic Model."
        )
        return

    df = _rows_to_dataframe(all_rows)

    # ------------------------------------------------------------------ #
    # KPI row
    # ------------------------------------------------------------------ #
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Total Predictions", f"{len(df):,}", "All modules combined")
    with k2:
        kpi_card("Company Sales", f"{(df['module_key'] == 'company_sales').sum():,}", "", variant="accent")
    with k3:
        kpi_card("Retail Sales", f"{(df['module_key'] == 'retail_sales').sum():,}", "", variant="success")
    with k4:
        kpi_card("House Price", f"{(df['module_key'] == 'house_price').sum():,}", "", variant="warning")
    with k5:
        kpi_card("Generic Model", f"{(df['module_key'] == 'generic_model').sum():,}", "", variant="default")

    st.write("")

    # ------------------------------------------------------------------ #
    # Filters
    # ------------------------------------------------------------------ #
    f1, f2 = st.columns([1, 2])
    with f1:
        module_filter = st.selectbox(
            "Module",
            ["All"] + [label for _, label, _ in MODULE_LABELS.values()],
            key="history_module_filter",
        )
    with f2:
        search_term = st.text_input(
            "Search by name",
            placeholder="e.g. Nova Retail Pvt Ltd, Green Valley Villa, Downtown Outlet...",
            key="history_search",
            help="Searches the Company Name / Store-Product Name / Property Name you gave each prediction.",
        )

    filtered = df.copy()
    if module_filter != "All":
        module_key = next(k for k, (_, label, _) in MODULE_LABELS.items() if label == module_filter)
        filtered = filtered[filtered["module_key"] == module_key]
    if search_term.strip():
        filtered = filtered[filtered["Name"].str.contains(search_term.strip(), case=False, na=False)]

    if filtered.empty:
        empty_state("No predictions match this filter/search. Try a different name or clear the filters.")
        return

    st.caption(f"Showing {len(filtered):,} of {len(df):,} predictions")

    display_df = filtered[["Module", "Name", "Predicted Value", "Confidence (%)", "Created At"]].copy()
    display_df["Predicted Value"] = display_df["Predicted Value"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "\u2014")
    display_df["Confidence (%)"] = display_df["Confidence (%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "\u2014")

    st.dataframe(display_df, width='stretch', hide_index=True)

    st.write("")
    if len(filtered) >= 2:
        st.plotly_chart(_trend_chart(filtered), width='stretch')

    dl1, dl2 = st.columns([1, 3])
    with dl1:
        st.download_button(
            "\u2B07\uFE0F Export Filtered History (CSV)",
            data=dataframe_to_csv_bytes(display_df),
            file_name="prediction_history.csv", mime="text/csv", width='stretch',
        )

    st.write("")
    with st.expander("\U0001F50D View full details for a specific prediction"):
        options = {
            f"#{rec['ID']} \u2022 {rec['Name']} \u2022 {rec['Module']} \u2022 {rec['Created At']}": rec["ID"]
            for rec in filtered.to_dict("records")
        }
        if options:
            selected_label = st.selectbox("Select a prediction", list(options.keys()), key="history_detail_select")
            selected_id = options[selected_label]
            selected_row = filtered[filtered["ID"] == selected_id].iloc[0]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Inputs**")
                try:
                    st.json(json.loads(selected_row["input_json"]))
                except Exception:
                    st.write(selected_row["input_json"])
            with d2:
                st.markdown("**Result**")
                try:
                    st.json(json.loads(selected_row["result_json"]))
                except Exception:
                    st.write(selected_row["result_json"])

    with st.expander("\u26A0\uFE0F Danger zone"):
        st.caption("This permanently deletes prediction history from the local database.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear filtered module's history", width='stretch'):
                if module_filter == "All":
                    db.clear_module_predictions()
                    db.clear_generic_predictions()
                elif module_filter == "Generic Model":
                    db.clear_generic_predictions()
                else:
                    module_key = next(k for k, (_, label, _) in MODULE_LABELS.items() if label == module_filter)
                    db.clear_module_predictions(module_key)
                st.success("History cleared.")
                st.rerun()
