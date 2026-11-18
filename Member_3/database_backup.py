
import os
import psycopg2
from psycopg2.extras import Json, RealDictCursor


# DATABASE CONNECTION
def get_connection():

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    return psycopg2.connect(database_url)


# 1. CREATE STUDENT
def create_student(student_id):

    query = """
        INSERT INTO students (student_id)
        VALUES (%s)
        ON CONFLICT (student_id) DO NOTHING;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (student_id,))

    return True


# 2. SAVE ATTEMPT
def save_attempt(student_id, question_id,
                 concept_id, evaluation):

    classification = evaluation.get("classification")
    outcome = evaluation.get("outcome")

    if classification not in (
        None, "correct", "partially_correct", "incorrect"
    ):
        raise ValueError("Invalid classification")

    if outcome not in (None, 0, 1):
        raise ValueError("Outcome must be 0, 1 or None")

    query = """
        INSERT INTO attempts (
            student_id,
            question_id,
            concept_id,
            classification,
            outcome,
            evaluation
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    student_id,
                    question_id,
                    concept_id,
                    classification,
                    outcome,
                    Json(evaluation)
                )
            )

    return True


# 3. SAVE MASTERY
def save_mastery(student_id, mastery_profile):

    query = """
        INSERT INTO mastery (
            student_id,
            concept_id,
            mastery,
            updated_at
        )
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (student_id, concept_id)
        DO UPDATE SET
            mastery = EXCLUDED.mastery,
            updated_at = CURRENT_TIMESTAMP;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            for concept_id, mastery_score in mastery_profile.items():

                if not 0 <= mastery_score <= 1:
                    raise ValueError(
                        f"Invalid mastery score: {mastery_score}"
                    )

                cur.execute(
                    query,
                    (student_id, concept_id, mastery_score)
                )

    return True


# 4. GET STUDENT HISTORY
def get_student_history(student_id):

    query = """
        SELECT
            history_id,
            student_id,
            concept_id,
            activity,
            reason,
            next_action,
            created_at
        FROM learning_history
        WHERE student_id = %s
        ORDER BY created_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            cur.execute(query, (student_id,))
            history = cur.fetchall()

    return [dict(row) for row in history]
