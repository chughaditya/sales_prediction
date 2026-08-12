"""
database.py
-------------
Thin SQLite wrapper used across the whole app. The schema already exists
inside database/sales_prediction.db (see the CREATE TABLE statements
below, which are idempotent -- they only create tables if missing, they
never wipe existing data).

Tables:
    datasets            -- every CSV uploaded / sample dataset used
    model_performance    -- results from the generic "Train Model" page
    predictions          -- legacy generic-model predictions
    user_inputs          -- raw form submissions (rarely used directly)
    module_predictions   -- Company / Retail / House specialized module
                            predictions, including the free-text "name"
                            fields (Company Name, Property Nickname,
                            Store Name) so history is searchable by name.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "sales_prediction.db")

os.makedirs(DB_DIR, exist_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    rows INTEGER,
    columns INTEGER,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    r2_score REAL,
    adjusted_r2 REAL,
    mae REAL,
    mse REAL,
    rmse REAL,
    cv_score REAL,
    is_best INTEGER DEFAULT 0,
    trained_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_used TEXT,
    input_data TEXT,
    predicted_sales REAL,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_json TEXT,
    submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS module_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL,
    input_json TEXT,
    result_json TEXT,
    predicted_value REAL,
    confidence REAL,
    created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    def log_dataset(self, file_name: str, rows: int, columns: int):
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO datasets (file_name, rows, columns, uploaded_at) VALUES (?, ?, ?, ?)",
                (file_name, int(rows), int(columns), datetime.now().isoformat()),
            )

    def get_datasets(self, limit: int = 200):
        with self._connect() as con:
            cur = con.execute("SELECT * FROM datasets ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Generic model performance (Train Model page)
    # ------------------------------------------------------------------ #
    def log_model_performance(self, model_name: str, r2_score: float, adjusted_r2: float,
                               mae: float, mse: float, rmse: float, cv_score: float = None,
                               is_best: bool = False):
        with self._lock, self._connect() as con:
            if is_best:
                con.execute("UPDATE model_performance SET is_best = 0")
            con.execute(
                """INSERT INTO model_performance
                   (model_name, r2_score, adjusted_r2, mae, mse, rmse, cv_score, is_best, trained_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_name, r2_score, adjusted_r2, mae, mse, rmse, cv_score,
                 1 if is_best else 0, datetime.now().isoformat()),
            )

    def get_model_performance_history(self, limit: int = 200):
        with self._connect() as con:
            cur = con.execute("SELECT * FROM model_performance ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Legacy generic predictions
    # ------------------------------------------------------------------ #
    def log_prediction(self, model_used: str, input_data: dict, predicted_sales: float, confidence: float = None):
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO predictions (model_used, input_data, predicted_sales, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (model_used, json.dumps(input_data, default=str), float(predicted_sales),
                 confidence, datetime.now().isoformat()),
            )

    def get_prediction_history(self, limit: int = 5000):
        with self._connect() as con:
            cur = con.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def clear_generic_predictions(self):
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM predictions")

    def log_user_input(self, input_data: dict):
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO user_inputs (input_json, submitted_at) VALUES (?, ?)",
                (json.dumps(input_data, default=str), datetime.now().isoformat()),
            )

    # ------------------------------------------------------------------ #
    # Specialized module predictions (Company / Retail / House)
    # ------------------------------------------------------------------ #
    def log_module_prediction(self, module: str, input_data: dict, result_data: dict,
                               predicted_value: float, confidence: float = None):
        with self._lock, self._connect() as con:
            con.execute(
                """INSERT INTO module_predictions
                   (module, input_json, result_json, predicted_value, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (module, json.dumps(input_data, default=str), json.dumps(result_data, default=str),
                 float(predicted_value) if predicted_value is not None else None,
                 confidence, datetime.now().isoformat()),
            )

    def get_module_predictions(self, limit: int = 5000, module: str = None):
        with self._connect() as con:
            if module:
                cur = con.execute(
                    "SELECT * FROM module_predictions WHERE module = ? ORDER BY id DESC LIMIT ?",
                    (module, limit),
                )
            else:
                cur = con.execute("SELECT * FROM module_predictions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def delete_module_prediction(self, prediction_id: int):
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM module_predictions WHERE id = ?", (prediction_id,))

    def clear_module_predictions(self, module: str = None):
        with self._lock, self._connect() as con:
            if module:
                con.execute("DELETE FROM module_predictions WHERE module = ?", (module,))
            else:
                con.execute("DELETE FROM module_predictions")


_db_instance = None
_db_lock = threading.Lock()


def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = Database()
                logger.info("Database initialised at %s", DB_PATH)
    return _db_instance
