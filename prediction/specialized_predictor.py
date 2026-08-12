"""
specialized_predictor.py
--------------------------
Thin, cached wrapper around the three pre-trained joblib "bundles" in
models/ (company_sales_bundle.pkl, house_price_bundle.pkl,
retail_sales_bundle.pkl). Each bundle is a dict with:

    model              -- fitted sklearn RandomForestRegressor
    cat_cols           -- list of categorical feature names
    num_cols           -- list of numeric feature names
    categories         -- {cat_col: [allowed values...]}
    scaler             -- fitted StandardScaler (fit on num_cols only)
    feature_names_out  -- final ordered feature list the model expects
                          (num_cols, then one dummy column per category
                          value, grouped by cat_col)
    target_col         -- short target name
    target_name        -- display target name, e.g. "Estimated House Price ($)"
    numeric_ranges     -- {num_col: (min, default, max)} used to seed form defaults
    r2, mae            -- held-out metrics from training
    sample_X           -- a sample of encoded/scaled training rows, used
                          as the explainability baseline

This module turns a friendly `{feature_name: raw_value}` dict into the
exact encoded/scaled feature vector the model expects, and provides a
lightweight (non-SHAP) explainability layer good enough for the
feature-contribution / waterfall charts on the Prediction page.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class SpecializedPredictor:
    def __init__(self, bundle_file: str):
        self.bundle_file = bundle_file
        self.bundle = None
        self.is_ready = False
        path = os.path.join(MODELS_DIR, bundle_file)
        if os.path.isfile(path):
            try:
                self.bundle = joblib.load(path)
                self.is_ready = True
            except Exception:
                self.is_ready = False

    # ------------------------------------------------------------------ #
    @property
    def required_columns(self):
        return list(self.bundle["cat_cols"]) + list(self.bundle["num_cols"])

    # ------------------------------------------------------------------ #
    def _build_feature_vector(self, inputs: dict) -> np.ndarray:
        bundle = self.bundle
        num_cols = bundle["num_cols"]
        cat_cols = bundle["cat_cols"]
        categories = bundle["categories"]
        feature_names_out = bundle["feature_names_out"]

        numeric_df = pd.DataFrame([[float(inputs[c]) for c in num_cols]], columns=num_cols)
        scaled = bundle["scaler"].transform(numeric_df)[0]

        feat = {name: float(val) for name, val in zip(num_cols, scaled)}
        for cat_col in cat_cols:
            selected = inputs.get(cat_col)
            for value in categories[cat_col]:
                feat[f"{cat_col}={value}"] = 1.0 if selected == value else 0.0

        return np.array([feat.get(name, 0.0) for name in feature_names_out], dtype=float)

    # ------------------------------------------------------------------ #
    def predict(self, inputs: dict):
        vector = pd.DataFrame(
            [self._build_feature_vector(inputs)], columns=self.bundle["feature_names_out"]
        )
        model = self.bundle["model"]
        prediction = float(model.predict(vector)[0])

        confidence = None
        try:
            vector_array = vector.to_numpy()
            tree_preds = np.array([tree.predict(vector_array)[0] for tree in model.estimators_])
            mean_pred = tree_preds.mean()
            spread = tree_preds.std()
            stability = max(0.0, 100.0 - (spread / (abs(mean_pred) + 1e-6)) * 100.0)
            r2_component = max(0.0, min(1.0, self.bundle.get("r2", 0.75))) * 100.0
            confidence = round(0.6 * stability + 0.4 * r2_component, 1)
            confidence = max(1.0, min(99.0, confidence))
        except Exception:
            confidence = None

        return prediction, confidence

    # ------------------------------------------------------------------ #
    def explain(self, inputs: dict, top_n: int = 6):
        bundle = self.bundle
        model = bundle["model"]
        feature_names_out = bundle["feature_names_out"]
        cat_cols = bundle["cat_cols"]
        categories = bundle["categories"]

        raw_vector = self._build_feature_vector(inputs)
        vector_df = pd.DataFrame([raw_vector], columns=feature_names_out)
        importances = getattr(model, "feature_importances_", np.ones(len(feature_names_out)) / len(feature_names_out))

        # Pseudo-SHAP: importance * how far this row's value deviates from
        # "typical" (0 for a standardized numeric feature; the average
        # selection rate for a one-hot dummy). Good enough for a directional,
        # human-readable feature-contribution view without requiring SHAP.
        dummy_priors = {}
        for cat_col in cat_cols:
            n = max(1, len(categories[cat_col]))
            for value in categories[cat_col]:
                dummy_priors[f"{cat_col}={value}"] = 1.0 / n

        contributions = []
        for name, imp, val in zip(feature_names_out, importances, raw_vector):
            baseline = dummy_priors.get(name, 0.0)
            deviation = val - baseline
            contributions.append((name, float(imp) * float(deviation)))

        prediction = float(model.predict(vector_df)[0])
        # Scale contributions so they roughly sum toward (prediction - base_value)
        try:
            base_value = float(model.predict(bundle["sample_X"]).mean())
        except Exception:
            base_value = prediction

        raw_total = sum(v for _, v in contributions) or 1.0
        target_total = prediction - base_value
        scale = target_total / raw_total if raw_total != 0 else 0.0
        contributions = [(_pretty_label(name, inputs), v * scale) for name, v in contributions]

        all_sorted_by_abs = sorted(contributions, key=lambda t: abs(t[1]), reverse=True)
        positives = sorted([c for c in contributions if c[1] > 0], key=lambda t: t[1], reverse=True)[:top_n]
        negatives = sorted([c for c in contributions if c[1] < 0], key=lambda t: t[1])[:top_n]

        return {
            "base_value": base_value,
            "positives": positives,
            "negatives": negatives,
            "all_sorted_by_abs": all_sorted_by_abs[:top_n],
        }

    # ------------------------------------------------------------------ #
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        confidences = []
        for _, row in df.iterrows():
            inputs = {c: row[c] for c in self.required_columns}
            pred, conf = self.predict(inputs)
            results.append(pred)
            confidences.append(conf)

        out = df.copy()
        target_label = self.bundle.get("target_name", self.bundle.get("target_col", "Value"))
        out[f"Predicted {target_label}"] = results
        out["Confidence (%)"] = confidences
        return out


def _pretty_label(name: str, inputs: dict) -> str:
    """Turns 'Industry=Retail' into 'Industry=Retail' (already readable) and
    a plain numeric feature name into 'Feature=value' for consistent
    downstream display (module code splits on '=' to get the feature name)."""
    if "=" in name:
        return name
    value = inputs.get(name)
    if isinstance(value, float):
        value = round(value, 2)
    return f"{name}={value}"


@st.cache_resource(show_spinner=False)
def get_specialized_predictor(bundle_file: str) -> SpecializedPredictor:
    return SpecializedPredictor(bundle_file)
