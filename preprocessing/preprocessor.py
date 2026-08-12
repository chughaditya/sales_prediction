"""
preprocessor.py
-----------------
General-purpose cleaning pipeline used by the Dataset Analysis and Train
Model pages. Works on ANY uploaded CSV (not the three specialized
Company/Retail/House datasets, which have their own pre-trained bundles).

Pipeline steps (run_full_pipeline):
    1. Drop exact duplicate rows
    2. Fill missing numeric values with the column median
       Fill missing categorical values with the column mode
    3. Detect + cap outliers in numeric columns using the IQR method
    4. Label/One-Hot encode categorical columns
    5. Scale numeric columns with StandardScaler

Static helpers (data_summary / feature_statistics / correlation_matrix)
are used directly on the raw, unencoded dataframe for the read-only
analysis tabs.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.target_column = target_column
        self.encoders = {}
        self.scaler = None
        self.report = {}

    # ------------------------------------------------------------------ #
    # Static, read-only analysis helpers (no mutation of the input df)
    # ------------------------------------------------------------------ #
    @staticmethod
    def data_summary(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            series = df[col]
            rows.append({
                "Column": col,
                "Dtype": str(series.dtype),
                "Non-Null Count": int(series.notna().sum()),
                "Missing": int(series.isna().sum()),
                "Missing %": round(series.isna().mean() * 100, 2),
                "Unique Values": int(series.nunique()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return pd.DataFrame()
        desc = numeric_df.describe().T
        desc["skew"] = numeric_df.skew()
        desc["kurtosis"] = numeric_df.kurtosis()
        desc = desc.reset_index().rename(columns={"index": "Column"})
        return desc.round(3)

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return pd.DataFrame()
        return numeric_df.corr().round(3)

    # ------------------------------------------------------------------ #
    # Mutating cleaning pipeline
    # ------------------------------------------------------------------ #
    def _handle_missing_values(self):
        handled = {}
        for col in self.df.columns:
            missing = int(self.df[col].isna().sum())
            if missing == 0:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fill_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(fill_value)
                handled[col] = f"filled {missing} missing with median ({fill_value:.2f})"
            else:
                mode_series = self.df[col].mode()
                fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                self.df[col] = self.df[col].fillna(fill_value)
                handled[col] = f"filled {missing} missing with mode ('{fill_value}')"
        return handled

    def _handle_outliers(self):
        detected = {}
        handled = {}
        numeric_cols = self.df.select_dtypes(include="number").columns
        if self.target_column in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != self.target_column]
        for col in numeric_cols:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            count = int(mask.sum())
            if count == 0:
                continue
            detected[col] = count
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            handled[col] = f"capped {count} outlier(s) to [{lower:.2f}, {upper:.2f}]"
        return detected, handled

    def _encode_categoricals(self):
        encoded_cols = []
        cat_cols = self.df.select_dtypes(include="object").columns
        for col in cat_cols:
            if col == self.target_column:
                continue
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
            encoded_cols.append(col)
        return encoded_cols

    def _scale_numeric(self):
        numeric_cols = [c for c in self.df.select_dtypes(include="number").columns
                         if c != self.target_column]
        if not numeric_cols:
            return []
        self.scaler = StandardScaler()
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        return numeric_cols

    def run_full_pipeline(self):
        initial_shape = self.df.shape

        before_dupes = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = before_dupes - len(self.df)

        missing_values_handled = self._handle_missing_values()
        outliers_detected, outliers_handled = self._handle_outliers()
        columns_encoded = self._encode_categoricals()
        columns_scaled = self._scale_numeric()

        final_shape = self.df.shape

        self.report = {
            "initial_shape": initial_shape,
            "final_shape": final_shape,
            "duplicates_removed": duplicates_removed,
            "missing_values_handled": missing_values_handled,
            "outliers_detected": outliers_detected,
            "outliers_handled": outliers_handled,
            "columns_encoded": columns_encoded,
            "columns_scaled": columns_scaled,
        }
        return self.df, self.report
