
"""
LearnSyncAI - Curriculum Planning Module
Member 3

Selects the next learning activity using:
1. Student mastery from the BKT Knowledge Tracing Agent
2. Strict prerequisite relationships
3. Recommended learning relationships
4. Previous learning history
"""

# All agreed DSA concept IDs
DSA_CONCEPTS = [
    "arrays",
    "linked_lists",
    "stacks",
    "queues",
    "recursion",
    "trees",
    "binary_search_trees",
    "heaps",
    "sorting",
    "hashing",
    "graphs",
    "bfs",
    "dfs"
]

# Strict prerequisites:
# A concept is held back until its strict prerequisites
# reach the configured mastery threshold.
STRICT_PREREQUISITES = {
    "arrays": [],
    "linked_lists": [],
    "stacks": [],
    "queues": [],
    "recursion": [],
    "trees": [],
    "binary_search_trees": ["trees"],
    "heaps": ["trees"],
    "sorting": [],
    "hashing": [],
    "graphs": [],
    "bfs": ["graphs"],
    "dfs": ["graphs"]
}

# Recommended prerequisites:
# These guide learning order but NEVER block a student.
RECOMMENDED_PREREQUISITES = {
    "arrays": [],
    "linked_lists": ["arrays"],
    "stacks": [],
    "queues": [],
    "recursion": [],
    "trees": [],
    "binary_search_trees": [],
    "heaps": [],
    "sorting": ["arrays"],
    "hashing": [],
    "graphs": [],
    "bfs": [],
    "dfs": []
}

# Configurable curriculum settings
STRICT_PREREQUISITE_THRESHOLD = 0.60

REVISION_THRESHOLD = 0.40
PRACTICE_THRESHOLD = 0.70
ADVANCE_THRESHOLD = 0.85

# Default prerequisite configuration
DSA_PREREQUISITES = {
    "strict": STRICT_PREREQUISITES,
    "recommended": RECOMMENDED_PREREQUISITES
}


def _extract_mastery(mastery_profile):
    """
    Accepts either:
    {"student_id": "...", "mastery": {...}}
    or a direct concept-to-mastery dictionary.
    """

    if not isinstance(mastery_profile, dict):
        raise TypeError("mastery_profile must be a dictionary")

    mastery = mastery_profile.get("mastery", mastery_profile)

    if not isinstance(mastery, dict):
        raise TypeError("Mastery values must be provided as a dictionary")

    return mastery


def _get_strictly_eligible_concepts(mastery, strict_prerequisites):
    """
    Returns concepts whose strict prerequisites
    have reached the required mastery threshold.
    """

    eligible = []

    for concept in DSA_CONCEPTS:
        prerequisites = strict_prerequisites.get(concept, [])

        if all(
            mastery.get(prerequisite, 0.0)
            >= STRICT_PREREQUISITE_THRESHOLD
            for prerequisite in prerequisites
        ):
            eligible.append(concept)

    return eligible


def select_next_activity(
    student_id,
    mastery_profile,
    prerequisites,
    learning_history
):
    """
    Selects the next learning activity for a student.

    Required interface:
    select_next_activity(
        student_id,
        mastery_profile,
        prerequisites,
        learning_history
    )

    Returns a structured curriculum decision.
    """

    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string")

    mastery = _extract_mastery(mastery_profile)

    # Use the default configuration if none is supplied.
    if prerequisites is None:
        prerequisites = DSA_PREREQUISITES

    if not isinstance(prerequisites, dict):
        raise TypeError("prerequisites must be a dictionary")

    # Support both the new configurable structure and
    # a simple prerequisite dictionary for compatibility.
    if "strict" in prerequisites or "recommended" in prerequisites:
        strict = prerequisites.get("strict", {})
        recommended = prerequisites.get("recommended", {})
    else:
        strict = prerequisites
        recommended = RECOMMENDED_PREREQUISITES

    if not isinstance(strict, dict) or not isinstance(recommended, dict):
        raise TypeError("Prerequisite configurations must be dictionaries")

    if not isinstance(learning_history, list):
        raise TypeError("learning_history must be a list")

    # Find concepts that satisfy strict prerequisites.
    eligible = _get_strictly_eligible_concepts(mastery, strict)

    if not eligible:
        raise ValueError("No concepts currently satisfy strict prerequisites")

    # Prioritize eligible concepts that have never been assessed.
    unassessed = [
        concept for concept in eligible
        if concept not in mastery
    ]

    if unassessed:
        selected_concept = unassessed[0]
        current_mastery = None
    else:
        # Select the eligible concept with the lowest mastery.
        selected_concept = min(
            eligible,
            key=lambda concept: mastery[concept]
        )
        current_mastery = mastery[selected_concept]

    # Decide the activity using mastery thresholds.
    if current_mastery is None:
        activity = "reassessment"
        reason = "Concept has not been assessed yet"
        next_action = "reassess_mastery"

    elif current_mastery < REVISION_THRESHOLD:
        activity = "revision"
        reason = "Mastery below configured revision threshold"
        next_action = "retrieve_content"

    elif current_mastery < PRACTICE_THRESHOLD:
        activity = "practice"
        reason = "Student needs additional practice"
        next_action = "generate_question"

    elif current_mastery < ADVANCE_THRESHOLD:
        activity = "reassessment"
        reason = "Student is ready for a mastery reassessment"
        next_action = "reassess_mastery"

    else:
        activity = "advance"
        reason = "Mastery reached the configured advancement threshold"
        next_action = "select_next_concept"

    # Recommended prerequisites influence the explanation.
    recommended_for_concept = recommended.get(selected_concept, [])

    unmet_recommended = [
        concept for concept in recommended_for_concept
        if mastery.get(concept, 0.0) < STRICT_PREREQUISITE_THRESHOLD
    ]

    if unmet_recommended:
        reason += (
            ". Recommended prior concepts with lower mastery: "
            + ", ".join(unmet_recommended)
            + ". Learning is not blocked."
        )

    # Check previous learning activity.
    previous_activities = [
        item for item in learning_history
        if item.get("concept_id") == selected_concept
    ]

    if previous_activities:
        reason += ". Previous learning activity found in history"

    return {
        "student_id": student_id,
        "concept_id": selected_concept,
        "activity": activity,
        "reason": reason,
        "next_action": next_action
    }
