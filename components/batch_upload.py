"""
batch_upload.py
------------------
Shared "Batch Upload (CSV)" UI used by all three Prediction page modules.
Lets the user upload a CSV of many rows, runs the module's model on every
row, and lets them download the results (with a template + logging to
history built in).
"""

import streamlit as st
import pandas as pd

from utils.helpers import dataframe_to_csv_bytes, dataframe_to_excel_bytes


def render_batch_upload(predictor, module_key: str, module_label: str, db):
    st.markdown("**Batch Upload (CSV)**")
    st.caption(
        "Upload a CSV with one row per record. Column names must match the fields below exactly. "
        "Use the template to get the exact headers right."
    )

    required_cols = predictor.required_columns
    bundle = predictor.bundle

    # Downloadable template: header row + one example row using default values
    example_row = {}
    for c in bundle["cat_cols"]:
        example_row[c] = bundle["categories"][c][0]
    for c in bundle["num_cols"]:
        example_row[c] = bundle["numeric_ranges"][c][1]
    template_df = pd.DataFrame([example_row])[required_cols]

    st.download_button(
        "\U0001F4CB Download CSV Template", data=dataframe_to_csv_bytes(template_df),
        file_name=f"{module_key}_template.csv", mime="text/csv", key=f"{module_key}_template_dl",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{module_key}_batch_uploader")

    if uploaded is None:
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if raw_df.empty:
        st.warning("The uploaded CSV has no rows.")
        return

    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        st.error(
            "This CSV is missing required column(s): "
            + ", ".join(f"`{m}`" for m in missing)
            + ". Download the template above for the exact expected headers."
        )
        with st.expander("Columns found in your file"):
            st.write(list(raw_df.columns))
        return

    run_batch = st.button(f"\u26A1 Run Batch Prediction ({len(raw_df)} rows)", type="primary",
                           key=f"{module_key}_batch_run")
    if not run_batch:
        return

    with st.spinner(f"Predicting {module_label} for {len(raw_df)} rows..."):
        try:
            result_df = predictor.predict_batch(raw_df)
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
            return

        pred_col = [c for c in result_df.columns if c.startswith("Predicted ")][0]
        for _, row in result_df.iterrows():
            input_data = {c: row[c] for c in required_cols}
            db.log_module_prediction(
                module_key, input_data, {"prediction": row[pred_col]},
                row[pred_col], row.get("Confidence (%)"),
            )

    st.success(f"Batch prediction complete for {len(result_df)} rows.")
    st.dataframe(result_df, width='stretch', hide_index=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "\u2B07\uFE0F Download Results (CSV)", data=dataframe_to_csv_bytes(result_df),
            file_name=f"{module_key}_batch_results.csv", mime="text/csv",
            width='stretch', key=f"{module_key}_batch_csv_dl",
        )
    with dl2:
        st.download_button(
            "\u2B07\uFE0F Download Results (Excel)", data=dataframe_to_excel_bytes(result_df, "Batch Predictions"),
            file_name=f"{module_key}_batch_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch', key=f"{module_key}_batch_xlsx_dl",
        )
