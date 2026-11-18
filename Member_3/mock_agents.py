
"""
Temporary mock agents for LearnSyncAI Phase 1.

These simulate Member 1 and Member 2 modules.
They will be replaced with the actual team modules during Phase 2.
"""


def mock_generate_question(concept_id):
    """Simulates Member 2's RAG question generation."""

    return {
        "question_id": f"mock_{concept_id}_001",
        "concept_id": concept_id,
        "question": f"Explain the main idea of {concept_id}."
    }


def mock_evaluate_answer(answer):
    """Simulates Member 2's answer evaluator."""

    if not isinstance(answer, str) or not answer.strip():
        return {
            "classification": "incorrect",
            "outcome": 0,
            "feedback": "Please provide an answer."
        }

    return {
        "classification": "correct",
        "outcome": 1,
        "feedback": "Mock evaluation: answer accepted."
    }


def mock_update_mastery(old_mastery, concept_id, outcome):
    """Simulates Member 1's BKT mastery update."""

    updated_mastery = dict(old_mastery)

    current_score = updated_mastery.get(concept_id, 0.0)

    if outcome == 1:
        updated_mastery[concept_id] = min(
            1.0, current_score + 0.10
        )
    else:
        updated_mastery[concept_id] = max(
            0.0, current_score - 0.05
        )

    return updated_mastery
