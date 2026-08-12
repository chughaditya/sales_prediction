"""
train_model.py
----------------
Generic "Train Model" page: takes whatever dataset was loaded on the
Dataset Analysis page (st.session_state.raw_df + target_column), runs it
through DataPreprocessor, trains several candidate regressors, compares
them, and persists the best one to models/ so Home / future sessions can
pick it up as the "Generic Dataset Model".

This is independent of the three specialized pre-trained bundles
(Company / Retail / House) used on the Prediction page -- those are
fixed, curated models; this page is for the user's own uploaded data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from preprocessing.preprocessor import DataPreprocessor
from charts import visualizations as viz
from utils.ui_components import section_header, empty_state, kpi_card
from utils.helpers import MODELS_DIR
from utils.database import get_db

CANDIDATES = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}


def _train_all(X_train, X_test, y_train, y_test, n_features: int):
    results = []
    fitted_models = {}
    for name, model in CANDIDATES.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        n = len(y_test)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        try:
            cv_score = float(cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean())
        except Exception:
            cv_score = None

        fitted_models[name] = model
        results.append({
            "model_name": name, "r2_score": round(r2, 4), "adjusted_r2": round(adjusted_r2, 4),
            "mae": round(mae, 2), "mse": round(mse, 2), "rmse": round(rmse, 2),
            "cv_score": round(cv_score, 4) if cv_score is not None else None,
            "predictions": preds,
        })
    return results, fitted_models


def render():
    section_header("Train Model", "Train and automatically select the best model for your dataset")

    if "raw_df" not in st.session_state:
        empty_state("Please upload or select a dataset on the 'Dataset Analysis' page first.")
        return
    if "target_column" not in st.session_state:
        empty_state("Please select a target column on the 'Dataset Analysis' page (Data Quality Report tab) first.")
        return

    df = st.session_state.raw_df
    target_col = st.session_state.target_column
    db = get_db()

    st.caption(f"Dataset: **{st.session_state.get('dataset_name', 'uploaded dataset')}** \u2022 "
               f"Target column: **{target_col}** \u2022 {df.shape[0]} rows, {df.shape[1]} columns")

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05, key="train_test_size")
    train_clicked = st.button("\U0001F680 Run Cleaning Pipeline + Train All Models", type="primary")

    if not train_clicked and "train_results" not in st.session_state:
        empty_state("Click the button above to clean the data and train Linear Regression, Ridge, Decision "
                     "Tree, Random Forest, and Gradient Boosting models, then automatically pick the best one.")
        return

    if train_clicked:
        with st.spinner("Cleaning data and training candidate models..."):
            pre = DataPreprocessor(df, target_column=target_col)
            cleaned_df, report = pre.run_full_pipeline()

            feature_cols = [c for c in cleaned_df.columns if c != target_col]
            X = cleaned_df[feature_cols]
            y = cleaned_df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            results, fitted_models = _train_all(X_train, X_test, y_train, y_test, n_features=len(feature_cols))
            best = max(results, key=lambda r: r["r2_score"])

            for r in results:
                db.log_model_performance(
                    r["model_name"], r["r2_score"], r["adjusted_r2"], r["mae"], r["mse"], r["rmse"],
                    r["cv_score"], is_best=(r["model_name"] == best["model_name"]),
                )

            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(fitted_models[best["model_name"]], os.path.join(MODELS_DIR, "best_model.pkl"))
            joblib.dump(best["model_name"], os.path.join(MODELS_DIR, "best_model_name.pkl"))
            joblib.dump(best["r2_score"], os.path.join(MODELS_DIR, "best_model_r2.pkl"))
            joblib.dump(pre.encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
            joblib.dump(pre.scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
            joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_names.pkl"))
            joblib.dump([c for c in feature_cols if c in pre.df.select_dtypes(include="number").columns],
                        os.path.join(MODELS_DIR, "numeric_columns.pkl"))

            st.session_state.train_results = results
            st.session_state.train_best = best
            st.session_state.train_report = report
            st.session_state.train_y_test = y_test.values
            st.session_state.train_best_preds = best["predictions"]

        st.success(f"Training complete. Best model: **{best['model_name']}** (R\u00b2 = {best['r2_score']:.3f})")

    results = st.session_state.get("train_results", [])
    best = st.session_state.get("train_best")
    if not results or best is None:
        return

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Best Model", best["model_name"], variant="success")
    with k2:
        kpi_card("R\u00b2 Score", f"{best['r2_score']:.3f}", variant="accent")
    with k3:
        kpi_card("MAE", f"{best['mae']:,.2f}")
    with k4:
        kpi_card("RMSE", f"{best['rmse']:,.2f}", variant="warning")

    st.write("")
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "predictions"} for r in results])
    st.dataframe(results_df, width='stretch', hide_index=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.model_comparison_chart(results_df, "r2_score"), width='stretch')
    with c2:
        st.plotly_chart(viz.model_comparison_radar(results_df), width='stretch')

    y_test = st.session_state.get("train_y_test")
    best_preds = st.session_state.get("train_best_preds")
    if y_test is not None and best_preds is not None:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(viz.actual_vs_predicted_chart(y_test, best_preds), width='stretch')
        with c4:
            st.plotly_chart(viz.residual_plot(y_test, best_preds), width='stretch')
        st.plotly_chart(viz.error_distribution_chart(y_test, best_preds), width='stretch')
