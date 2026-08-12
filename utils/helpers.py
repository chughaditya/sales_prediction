"""
helpers.py
------------
Small, dependency-light helper functions shared across pages: path
constants, number formatting, artifact existence checks, and dataframe
export helpers (CSV / Excel bytes for st.download_button).
"""

import os
import io
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
HISTORY_DIR = os.path.join(BASE_DIR, "history")

for _d in (DATASET_DIR, MODELS_DIR, EXPORTS_DIR, HISTORY_DIR):
    os.makedirs(_d, exist_ok=True)


def artifact_exists(filename: str) -> bool:
    """Whether a given file exists inside the models/ directory."""
    return os.path.isfile(os.path.join(MODELS_DIR, filename))


def format_number(value) -> str:
    """Human-friendly compact number formatting (1.2K, 3.4M, ...)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}{value / 1_000:.1f}K"
    if value == int(value):
        return f"{sign}{int(value):,}"
    return f"{sign}{value:,.2f}"


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
    buffer.seek(0)
    return buffer.getvalue()
