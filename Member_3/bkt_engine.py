
import math

# Baseline BKT parameters.
# These are configurable assumptions, not dataset-trained estimates.
P_INIT = 0.20
P_LEARN = 0.10
P_GUESS = 0.20
P_SLIP = 0.10

# Stores current mastery probabilities by (student_id, concept_id).
_mastery = {}


def _validate_parameters(p_init, p_learn, p_guess, p_slip):
    for name, value in {
        "p_init": p_init,
        "p_learn": p_learn,
        "p_guess": p_guess,
        "p_slip": p_slip
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    if p_guess + p_slip >= 2:
        raise ValueError("Invalid guess/slip parameter combination.")


def _update_probability(prior, outcome, p_learn, p_guess, p_slip):
    if outcome == 1:
        numerator = prior * (1 - p_slip)
        denominator = numerator + (1 - prior) * p_guess
    else:
        numerator = prior * p_slip
        denominator = numerator + (1 - prior) * (1 - p_guess)

    posterior = numerator / denominator if denominator else prior

    # Account for the possibility of learning after the attempt.
    updated_mastery = posterior + (1 - posterior) * p_learn

    return posterior, updated_mastery


def update_bkt(student_id, concept_id, outcome):
    """
    Update a student's mastery estimate after a binary response.
    outcome: 1 = correct, 0 = incorrect.
    """
    if outcome not in (0, 1):
        raise ValueError("Outcome must be binary: 0 or 1.")

    key = (str(student_id), str(concept_id))
    prior = _mastery.get(key, P_INIT)

    predicted_correct = (
        prior * (1 - P_SLIP) +
        (1 - prior) * P_GUESS
    )

    posterior, updated = _update_probability(
        prior, outcome, P_LEARN, P_GUESS, P_SLIP
    )

    _mastery[key] = updated

    return {
        "student_id": str(student_id),
        "concept_id": str(concept_id),
        "outcome": int(outcome),
        "prior_mastery": prior,
        "predicted_correct_probability": predicted_correct,
        "posterior_mastery": posterior,
        "mastery": updated
    }


def get_mastery_profile(student_id):
    """
    Return the student's mastery profile in the shared project format.

    Output:
        {
            "student_id": "...",
            "mastery": {"arrays": 0.72}
        }
    """
    student_id = str(student_id)

    mastery = {
        concept_id: value
        for (sid, concept_id), value in _mastery.items()
        if sid == student_id
    }

    return {
        "student_id": student_id,
        "mastery": mastery
    }


def predict_correct_probability(student_id, concept_id):
    """Predict the probability of a correct answer before the next attempt."""
    key = (str(student_id), str(concept_id))
    mastery = _mastery.get(key, P_INIT)

    return mastery * (1 - P_SLIP) + (1 - mastery) * P_GUESS


def reset_bkt_state():
    """Clear all in-memory mastery estimates."""
    _mastery.clear()


def evaluate_bkt(sequences):
    """
    Evaluate one-step-ahead predictions on chronological sequences.
    Each prediction is made before observing that attempt's outcome.
    """
    reset_bkt_state()

    actual = []
    predicted = []

    for (student_id, concept_id), attempts in sequences.items():
        for attempt in attempts:
            probability = predict_correct_probability(
                student_id, concept_id
            )

            actual.append(int(attempt["outcome"]))
            predicted.append(probability)

            update_bkt(
                student_id,
                concept_id,
                int(attempt["outcome"])
            )

    if not actual:
        raise ValueError("No observations available for evaluation.")

    epsilon = 1e-15
    clipped = [
        min(max(p, epsilon), 1 - epsilon)
        for p in predicted
    ]

    accuracy = sum(
        (p >= 0.5) == bool(y)
        for p, y in zip(predicted, actual)
    ) / len(actual)

    brier_score = sum(
        (p - y) ** 2
        for p, y in zip(predicted, actual)
    ) / len(actual)

    log_loss = -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for p, y in zip(clipped, actual)
    ) / len(actual)

    return {
        "observations": len(actual),
        "accuracy": accuracy,
        "brier_score": brier_score,
        "log_loss": log_loss
    }
