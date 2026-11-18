"""
LearnSync AI - Member 3: Database Layer (database.py)
======================================================
Provides persistent storage for students, learning attempts, mastery profiles,
and learning history.

Backend selection (automatic):
  - PostgreSQL: When DATABASE_URL environment variable is set.
  - SQLite:     When DATABASE_URL is not set (default for local dev & demo).

This ensures the application runs on a demo laptop without PostgreSQL installed.

Change from original:
  Original database.py was PostgreSQL-only and raised RuntimeError immediately
  when DATABASE_URL was unset. This rewrite adds an SQLite fallback with an
  identical public interface so the rest of the application is unaffected.
"""

import os
import math
import json
import sqlite3
from numbers import Real
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_USE_POSTGRES = bool(os.environ.get("DATABASE_URL", "").strip())

# SQLite database file location — same directory as this file
_SQLITE_PATH = str(Path(__file__).parent.parent / "learnsync.db")


# ---------------------------------------------------------------------------
# Backend: PostgreSQL
# ---------------------------------------------------------------------------

def _get_postgres_connection():
    """Create a PostgreSQL connection. Requires DATABASE_URL env var."""
    try:
        import psycopg2
    except ImportError as e:
        raise ImportError(
            "psycopg2 is required for PostgreSQL support. "
            "Run: pip install psycopg2-binary"
        ) from e

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Either set DATABASE_URL for PostgreSQL, or leave it unset to use SQLite."
        )
    import psycopg2
    return psycopg2.connect(database_url)


@contextmanager
def _postgres_connection():
    """Managed PostgreSQL connection with automatic commit/rollback/close."""
    import psycopg2
    conn = _get_postgres_connection()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Backend: SQLite
# ---------------------------------------------------------------------------

def _init_sqlite():
    """
    Initialize the SQLite database schema if it does not already exist.
    Creates all required tables (students, attempts, mastery, learning_history).
    """
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            classification TEXT CHECK (
                classification IN ('correct', 'partially_correct', 'incorrect') OR classification IS NULL
            ),
            outcome INTEGER CHECK (outcome IN (0, 1) OR outcome IS NULL),
            evaluation TEXT,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TABLE IF NOT EXISTS mastery (
            student_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            mastery REAL NOT NULL DEFAULT 0.0 CHECK (mastery >= 0 AND mastery <= 1),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (student_id, concept_id),
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TABLE IF NOT EXISTS learning_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            activity TEXT NOT NULL CHECK (
                activity IN ('revision', 'practice', 'reassessment', 'advance')
            ),
            reason TEXT,
            next_action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );
    """)
    conn.commit()
    conn.close()


@contextmanager
def _sqlite_connection():
    """Managed SQLite connection with automatic commit/rollback/close."""
    _init_sqlite()
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unified connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def _managed_connection():
    """
    Provides a database connection using the appropriate backend.
    PostgreSQL when DATABASE_URL is set; SQLite otherwise.
    """
    if _USE_POSTGRES:
        with _postgres_connection() as conn:
            yield conn
    else:
        with _sqlite_connection() as conn:
            yield conn


def get_backend_info():
    """Return a human-readable description of the active database backend."""
    if _USE_POSTGRES:
        return "PostgreSQL (DATABASE_URL configured)"
    return f"SQLite (local file: {_SQLITE_PATH})"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_required_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def create_student(student_id):
    """
    Create a new student record if it does not already exist.

    Args:
        student_id (str): Unique student identifier.

    Returns:
        bool: True on success. Silently ignores existing student IDs.

    Raises:
        ValueError: If student_id is invalid.
    """
    _validate_required_string(student_id, "student_id")

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            cursor.execute(
                "INSERT INTO students (student_id) VALUES (%s) ON CONFLICT (student_id) DO NOTHING;",
                (student_id,)
            )
        else:
            cursor.execute(
                "INSERT OR IGNORE INTO students (student_id) VALUES (?);",
                (student_id,)
            )

    return True


def get_student(student_id):
    """
    Retrieve a student record.

    Args:
        student_id (str): Student identifier.

    Returns:
        dict or None: Student record dict, or None if not found.
    """
    _validate_required_string(student_id, "student_id")

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT student_id, created_at FROM students WHERE student_id = %s;",
                (student_id,)
            )
        else:
            cursor.execute(
                "SELECT student_id, created_at FROM students WHERE student_id = ?;",
                (student_id,)
            )
        row = cursor.fetchone()

    return dict(row) if row else None


def save_attempt(student_id, question_id, concept_id, evaluation):
    """
    Save a student's question attempt and evaluation result.

    Args:
        student_id (str): Student identifier.
        question_id (str): Question identifier.
        concept_id (str): DSA concept identifier.
        evaluation (dict): Evaluation result containing 'classification' and optionally 'outcome'.

    Returns:
        bool: True on success.

    Raises:
        ValueError: If input data is invalid.
    """
    _validate_required_string(student_id, "student_id")
    _validate_required_string(question_id, "question_id")
    _validate_required_string(concept_id, "concept_id")

    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a dictionary.")

    classification = evaluation.get("classification")
    outcome = evaluation.get("outcome") or evaluation.get("bkt_outcome")

    valid_classifications = (None, "correct", "partially_correct", "incorrect")
    if classification not in valid_classifications:
        raise ValueError(f"Invalid classification: {classification!r}")

    if outcome not in (None, 0, 1):
        # Coerce to int if possible
        try:
            outcome = int(outcome)
            if outcome not in (0, 1):
                outcome = None
        except (TypeError, ValueError):
            outcome = None

    evaluation_json = json.dumps(evaluation)

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            from psycopg2.extras import Json
            cursor.execute(
                """INSERT INTO attempts (student_id, question_id, concept_id, classification, outcome, evaluation)
                   VALUES (%s, %s, %s, %s, %s, %s);""",
                (student_id, question_id, concept_id, classification, outcome, Json(evaluation))
            )
        else:
            cursor.execute(
                """INSERT INTO attempts (student_id, question_id, concept_id, classification, outcome, evaluation)
                   VALUES (?, ?, ?, ?, ?, ?);""",
                (student_id, question_id, concept_id, classification, outcome, evaluation_json)
            )

    return True


def save_mastery(student_id, mastery_profile):
    """
    Insert or update concept-level mastery scores.

    Args:
        student_id (str): Student identifier.
        mastery_profile (dict): Mapping of concept_id -> mastery_score (0.0–1.0).
                                Also accepts the full BKT profile format:
                                {"student_id": "...", "mastery": {...}}

    Returns:
        bool: True on success.

    Raises:
        ValueError: If the profile or any scores are invalid.
    """
    _validate_required_string(student_id, "student_id")

    # Accept both {"mastery": {...}} and direct {concept_id: score} formats
    if isinstance(mastery_profile, dict) and "mastery" in mastery_profile:
        mastery_profile = mastery_profile["mastery"]

    if not isinstance(mastery_profile, dict):
        raise ValueError("mastery_profile must be a dictionary.")

    # Validate all scores before inserting anything
    for concept_id, mastery_score in mastery_profile.items():
        _validate_required_string(concept_id, "concept_id")
        if (
            isinstance(mastery_score, bool)
            or not isinstance(mastery_score, Real)
            or not math.isfinite(mastery_score)
            or not 0 <= mastery_score <= 1
        ):
            raise ValueError(
                f"Invalid mastery score for '{concept_id}': {mastery_score}. "
                "Expected a finite number between 0 and 1."
            )

    with _managed_connection() as conn:
        cursor = conn.cursor()
        for concept_id, mastery_score in mastery_profile.items():
            if _USE_POSTGRES:
                cursor.execute(
                    """INSERT INTO mastery (student_id, concept_id, mastery, updated_at)
                       VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                       ON CONFLICT (student_id, concept_id)
                       DO UPDATE SET mastery = EXCLUDED.mastery, updated_at = CURRENT_TIMESTAMP;""",
                    (student_id, concept_id, float(mastery_score))
                )
            else:
                cursor.execute(
                    """INSERT INTO mastery (student_id, concept_id, mastery, updated_at)
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT (student_id, concept_id)
                       DO UPDATE SET mastery = excluded.mastery, updated_at = CURRENT_TIMESTAMP;""",
                    (student_id, concept_id, float(mastery_score))
                )

    return True


def load_mastery(student_id):
    """
    Load a student's saved mastery profile from the database.

    Args:
        student_id (str): Student identifier.

    Returns:
        dict: Mapping of concept_id -> mastery_score for all stored concepts.
              Returns {} if no mastery records exist.
    """
    _validate_required_string(student_id, "student_id")

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            cursor.execute(
                "SELECT concept_id, mastery FROM mastery WHERE student_id = %s;",
                (student_id,)
            )
        else:
            cursor.execute(
                "SELECT concept_id, mastery FROM mastery WHERE student_id = ?;",
                (student_id,)
            )
        rows = cursor.fetchall()

    return {row["concept_id"]: float(row["mastery"]) for row in rows}


def get_student_history(student_id):
    """
    Retrieve a student's learning history (curriculum decisions).

    Args:
        student_id (str): Student identifier.

    Returns:
        list[dict]: Learning history records, newest first. Empty list if none.
    """
    _validate_required_string(student_id, "student_id")

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """SELECT history_id, student_id, concept_id, activity, reason, next_action, created_at
                   FROM learning_history WHERE student_id = %s ORDER BY created_at DESC;""",
                (student_id,)
            )
        else:
            cursor.execute(
                """SELECT history_id, student_id, concept_id, activity, reason, next_action, created_at
                   FROM learning_history WHERE student_id = ? ORDER BY created_at DESC;""",
                (student_id,)
            )
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


def get_student_attempts(student_id):
    """
    Retrieve all attempts made by a student, newest first.

    Args:
        student_id (str): Student identifier.

    Returns:
        list[dict]: Attempt records. Empty list if none.
    """
    _validate_required_string(student_id, "student_id")

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """SELECT attempt_id, student_id, question_id, concept_id, classification, outcome, evaluation, attempted_at
                   FROM attempts WHERE student_id = %s ORDER BY attempted_at DESC;""",
                (student_id,)
            )
        else:
            cursor.execute(
                """SELECT attempt_id, student_id, question_id, concept_id, classification, outcome, evaluation, attempted_at
                   FROM attempts WHERE student_id = ? ORDER BY attempted_at DESC;""",
                (student_id,)
            )
        rows = cursor.fetchall()

    result = []
    for row in rows:
        d = dict(row)
        # Parse JSON evaluation field for SQLite (PostgreSQL stores JSONB natively)
        if isinstance(d.get("evaluation"), str):
            try:
                d["evaluation"] = json.loads(d["evaluation"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)

    return result


def save_learning_history(student_id, concept_id, activity, reason, next_action):
    """
    Save a curriculum decision to the learning history table.

    Args:
        student_id (str): Student identifier.
        concept_id (str): DSA concept identifier.
        activity (str): One of: 'revision', 'practice', 'reassessment', 'advance'.
        reason (str): Human-readable reason for this decision.
        next_action (str): Next recommended action code.

    Returns:
        int: The new history_id.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_required_string(student_id, "student_id")
    _validate_required_string(concept_id, "concept_id")
    _validate_required_string(activity, "activity")
    _validate_required_string(reason, "reason")
    _validate_required_string(next_action, "next_action")

    allowed_activities = {"revision", "practice", "reassessment", "advance"}
    if activity not in allowed_activities:
        raise ValueError(f"Invalid activity: {activity!r}. Must be one of {sorted(allowed_activities)}.")

    with _managed_connection() as conn:
        cursor = conn.cursor()
        if _USE_POSTGRES:
            cursor.execute(
                """INSERT INTO learning_history (student_id, concept_id, activity, reason, next_action)
                   VALUES (%s, %s, %s, %s, %s) RETURNING history_id;""",
                (student_id, concept_id, activity, reason, next_action)
            )
            history_id = cursor.fetchone()[0]
        else:
            cursor.execute(
                """INSERT INTO learning_history (student_id, concept_id, activity, reason, next_action)
                   VALUES (?, ?, ?, ?, ?);""",
                (student_id, concept_id, activity, reason, next_action)
            )
            history_id = cursor.lastrowid

    return history_id
