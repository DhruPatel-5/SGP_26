import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from truthguard.config import DB_PATH, UPLOAD_DIR, REPORTS_DIR


def bootstrap_storage() -> None:
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn():
    bootstrap_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                detector_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                processing_seconds REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                report_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def record_prediction(username: str, detector_type: str, source_file: str, predicted_label: str, confidence: float, processing_seconds: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions (username, detector_type, source_file, predicted_label, confidence, processing_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, detector_type, source_file, predicted_label, confidence, processing_seconds, datetime.utcnow().isoformat()),
        )


def record_report(username: str, report_name: str, file_path: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO reports (username, report_name, file_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, report_name, file_path, datetime.utcnow().isoformat()),
        )


def fetch_predictions(username: str | None = None):
    with get_conn() as conn:
        if username:
            return conn.execute(
                "SELECT * FROM predictions WHERE username = ? ORDER BY created_at DESC", (username,)
            ).fetchall()
        return conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()


def fetch_reports(username: str | None = None):
    with get_conn() as conn:
        if username:
            return conn.execute(
                "SELECT * FROM reports WHERE username = ? ORDER BY created_at DESC", (username,)
            ).fetchall()
        return conn.execute("SELECT * FROM reports ORDER BY created_at DESC").fetchall()


def fetch_users():
    with get_conn() as conn:
        return conn.execute("SELECT id, username, full_name, email, role, created_at FROM users ORDER BY created_at DESC").fetchall()
