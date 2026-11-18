"""
LearnSync AI - Member 3: Orchestrator (orchestrator.py)
=======================================================
Central integration layer connecting Member 1 (Assessment + BKT),
Member 2 (RAG + Evaluator), and Member 3 (Database + Curriculum).

This module implements the complete learning cycle:

    Student Input
         ↓
    1. Create / retrieve student
    2. Load persisted mastery from DB → restore BKT state
    3. Curriculum selects next concept
    4. RAG retrieves content → generate explanation
    5. Assessment question is served
    6. Student submits answer
    7. Member 2 evaluates answer
    8. Member 1 BKT updates mastery
    9. Persist attempt + mastery to DB
    10. Curriculum selects next activity
    11. Return full result to Streamlit

Changes from original orchestrator.py:
  - Removed duplicate run_learning_cycle (was defined twice)
  - Removed all mock_agents usage from production path
  - Wired real Member 1 (assessment, bkt_engine) and Member 2 (rag, evaluator)
  - Added run_learning_cycle(student_id, student_input) as primary interface
  - Added session resume: loads mastery from DB and restores BKT state
  - mock_agents remain importable for unit testing only
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup — ensure all member modules are importable from any directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for _p in [
    os.path.join(_ROOT, "data"),      # Member 1: assessment, bkt_engine, data_pipeline
    os.path.join(_ROOT, "Member_2"),  # Member 2: rag, evaluator
    _HERE,                             # Member 3: database, curriculum
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Member 1 imports
# ---------------------------------------------------------------------------
import assessment as _assessment
import bkt_engine as _bkt

# ---------------------------------------------------------------------------
# Member 2 imports
# ---------------------------------------------------------------------------
import rag as _rag
import evaluator as _evaluator

# ---------------------------------------------------------------------------
# Member 3 imports
# ---------------------------------------------------------------------------
import database as _db
import curriculum as _curriculum


# ---------------------------------------------------------------------------
# Session helpers — restore BKT state from persisted mastery
# ---------------------------------------------------------------------------

def _restore_bkt_from_db(student_id):
    """
    Load persisted mastery scores from the database and inject them into
    the BKT engine's in-memory state so sessions resume correctly.

    This bridges the gap between BKT's in-memory state and DB persistence.
    """
    persisted = _db.load_mastery(student_id)
    for concept_id, score in persisted.items():
        key = (str(student_id), str(concept_id))
        _bkt._mastery[key] = float(score)
    return persisted


# ---------------------------------------------------------------------------
# Primary Interface: run_learning_cycle
# ---------------------------------------------------------------------------

def run_learning_cycle(student_id, student_input=None):
    """
    Execute one complete learning cycle for a student.

    This is the primary interface used by the Streamlit application and tests.

    Args:
        student_id (str): Student identifier (created if new).
        student_input (str, optional): Student's free-text answer to the
            current question. If None, the cycle returns a new question
            without evaluating an answer (e.g., first visit).

    Returns:
        dict: Full learning cycle result containing:
            - student_id (str)
            - mastery_profile (dict): Updated mastery per concept
            - curriculum_decision (dict): concept selected, activity, reason
            - retrieved_context (list): RAG chunks retrieved
            - explanation (str): Personalized explanation (Markdown)
            - question (dict or None): The assessment question served
            - evaluation (dict or None): Answer evaluation result (if answer given)
            - bkt_update (dict or None): BKT update result (if answer given)
            - learning_history (list): Updated learning history records
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string.")

    # 1. Ensure student exists in DB
    _db.create_student(student_id)

    # 2. Restore BKT state from persisted mastery
    _restore_bkt_from_db(student_id)

    # 3. Get current mastery profile
    mastery_profile_obj = _bkt.get_mastery_profile(student_id)
    mastery_profile = mastery_profile_obj.get("mastery", {})

    # 4. Retrieve learning history
    learning_history = _db.get_student_history(student_id)

    # 5. Curriculum selects the next concept
    prerequisites = _curriculum.DSA_PREREQUISITES
    curriculum_decision = _curriculum.select_next_activity(
        student_id=student_id,
        mastery_profile=mastery_profile,
        prerequisites=prerequisites,
        learning_history=learning_history
    )
    concept_id = curriculum_decision["concept_id"]

    # 6. Save curriculum decision to learning history
    try:
        _db.save_learning_history(
            student_id=student_id,
            concept_id=concept_id,
            activity=curriculum_decision["activity"],
            reason=curriculum_decision["reason"],
            next_action=curriculum_decision["next_action"]
        )
    except Exception:
        pass  # Non-fatal — curriculum decision still served

    # 7. RAG: retrieve content for this concept
    mastery_score = mastery_profile.get(concept_id, 0.0)
    query = f"Explain {concept_id.replace('_', ' ')} for a DSA student."
    try:
        retrieved_context = _rag.retrieve_context(query, concept_id=concept_id, top_k=3)
    except Exception:
        retrieved_context = []

    # 8. Generate personalized explanation
    try:
        explanation = _rag.generate_explanation(query, retrieved_context, mastery_level=mastery_score)
    except Exception as e:
        explanation = f"Content retrieval unavailable: {e}"

    # 9. Get an assessment question for this concept
    questions = _assessment.get_assessment_questions(concept_id=concept_id)
    question = questions[0] if questions else None

    # 10. Evaluate answer if one was provided
    evaluation = None
    bkt_update = None

    if student_input and question:
        question_id = question["question_id"]

        # Use MCQ evaluation for assessment questions (Member 1)
        try:
            assessment_result = _assessment.evaluate_assessment_answer(
                question_id=question_id,
                answer=student_input
            )
            # Build a compatible evaluation dict
            evaluation = {
                "question_id": question_id,
                "concept_id": concept_id,
                "classification": assessment_result["classification"],
                "score": 1.0 if assessment_result["outcome"] == 1 else 0.0,
                "bkt_outcome": assessment_result["outcome"],
                "identified_misconceptions": [],
                "feedback": assessment_result["feedback"],
                "evidence": []
            }
        except Exception:
            # Fallback to Member 2 evaluator for short-answer questions
            try:
                evaluation = _evaluator.evaluate_answer(question_id, student_input)
            except Exception as e2:
                evaluation = {
                    "question_id": question_id,
                    "concept_id": concept_id,
                    "classification": "incorrect",
                    "score": 0.0,
                    "bkt_outcome": 0,
                    "identified_misconceptions": [],
                    "feedback": f"Evaluation unavailable: {e2}",
                    "evidence": []
                }

        # 11. BKT update (Member 1)
        bkt_outcome = evaluation.get("bkt_outcome", 0)
        try:
            bkt_update = _bkt.update_bkt(student_id, concept_id, bkt_outcome)
        except Exception:
            bkt_update = None

        # 12. Get updated mastery profile
        mastery_profile_obj = _bkt.get_mastery_profile(student_id)
        mastery_profile = mastery_profile_obj.get("mastery", {})

        # 13. Persist attempt to DB
        try:
            _db.save_attempt(student_id, question_id, concept_id, evaluation)
        except Exception:
            pass  # Non-fatal

        # 14. Persist updated mastery to DB
        try:
            _db.save_mastery(student_id, mastery_profile)
        except Exception:
            pass  # Non-fatal

    # 15. Retrieve updated learning history
    updated_history = _db.get_student_history(student_id)

    return {
        "student_id": student_id,
        "mastery_profile": mastery_profile,
        "curriculum_decision": curriculum_decision,
        "retrieved_context": retrieved_context,
        "explanation": explanation,
        "question": question,
        "evaluation": evaluation,
        "bkt_update": bkt_update,
        "learning_history": updated_history
    }


# ---------------------------------------------------------------------------
# Assessment-specific flow
# ---------------------------------------------------------------------------

def run_assessment_cycle(student_id, concept_id=None):
    """
    Run a targeted assessment for a specific concept (or all concepts).

    Returns a list of questions for the given concept_id,
    or all assessment questions if concept_id is None.

    Args:
        student_id (str): Student identifier.
        concept_id (str, optional): DSA concept to assess. None = all.

    Returns:
        dict: {
            student_id, questions, mastery_profile
        }
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string.")

    _db.create_student(student_id)
    _restore_bkt_from_db(student_id)

    questions = _assessment.get_assessment_questions(concept_id=concept_id)
    mastery_profile_obj = _bkt.get_mastery_profile(student_id)

    return {
        "student_id": student_id,
        "questions": questions,
        "mastery_profile": mastery_profile_obj.get("mastery", {})
    }


def submit_assessment_answer(student_id, question_id, answer):
    """
    Evaluate a student's MCQ answer, update BKT, and persist results.

    Args:
        student_id (str): Student identifier.
        question_id (str): Assessment question ID (e.g. 'assess_arrays_01').
        answer (str): Student's chosen option (e.g. 'A' or the option text).

    Returns:
        dict: {
            student_id, question_id, concept_id,
            classification, feedback, outcome,
            mastery_profile (updated)
        }
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string.")

    _db.create_student(student_id)
    _restore_bkt_from_db(student_id)

    # Evaluate MCQ answer using Member 1
    result = _assessment.evaluate_assessment_answer(question_id, answer)
    concept_id = result["concept_id"]
    outcome = result["outcome"]

    # Update BKT
    bkt_update = _bkt.update_bkt(student_id, concept_id, outcome)

    # Build evaluation dict
    evaluation = {
        "question_id": question_id,
        "concept_id": concept_id,
        "classification": result["classification"],
        "score": float(outcome),
        "bkt_outcome": outcome,
        "identified_misconceptions": [],
        "feedback": result["feedback"],
        "evidence": []
    }

    # Persist attempt and mastery
    try:
        _db.save_attempt(student_id, question_id, concept_id, evaluation)
    except Exception:
        pass

    mastery_profile = _bkt.get_mastery_profile(student_id).get("mastery", {})
    try:
        _db.save_mastery(student_id, mastery_profile)
    except Exception:
        pass

    return {
        "student_id": student_id,
        "question_id": question_id,
        "concept_id": concept_id,
        "classification": result["classification"],
        "feedback": result["feedback"],
        "outcome": outcome,
        "bkt_update": bkt_update,
        "mastery_profile": mastery_profile
    }


def get_student_dashboard(student_id):
    """
    Retrieve all data needed for the Student Dashboard.

    Args:
        student_id (str): Student identifier.

    Returns:
        dict: {
            student_id, mastery_profile, learning_history,
            attempts, curriculum_recommendation
        }
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string.")

    _db.create_student(student_id)
    _restore_bkt_from_db(student_id)

    mastery_profile = _bkt.get_mastery_profile(student_id).get("mastery", {})
    learning_history = _db.get_student_history(student_id)
    attempts = _db.get_student_attempts(student_id)

    # Get curriculum recommendation
    prerequisites = _curriculum.DSA_PREREQUISITES
    try:
        recommendation = _curriculum.select_next_activity(
            student_id=student_id,
            mastery_profile=mastery_profile,
            prerequisites=prerequisites,
            learning_history=learning_history
        )
    except Exception as e:
        recommendation = {"error": str(e)}

    return {
        "student_id": student_id,
        "mastery_profile": mastery_profile,
        "learning_history": learning_history,
        "attempts": attempts,
        "curriculum_recommendation": recommendation
    }
