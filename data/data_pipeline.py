from pathlib import Path
from collections import defaultdict

import pandas as pd


def load_zenodo_data(data_directory):
    """
    Load the three Zenodo dataset tables.

    Returns a dictionary containing:
    submissions, distribution, and questions.
    """
    data_directory = Path(data_directory)

    submissions_path = data_directory / "java_submissions.csv"
    distribution_path = data_directory / "distribution_per_question.csv"
    questions_path = data_directory / "tst_questions.csv"

    for path in [submissions_path, distribution_path, questions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    return {
        "submissions": pd.read_csv(submissions_path),
        "distribution": pd.read_csv(distribution_path),
        "questions": pd.read_csv(questions_path, sep="~")
    }


def _parse_timestamp_column(series):
    """Parse timestamps stored as epoch milliseconds or date strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(
        numeric, unit="ms", errors="coerce"
    )

    parsed_text = pd.to_datetime(series, errors="coerce")

    return parsed_numeric.fillna(parsed_text)


def preprocess_interactions(raw_data):
    """
    Convert raw Java submissions into cleaned student interactions.

    Success -> outcome 1
    Fail    -> outcome 0
    Empty or unrecognized results are excluded.

    Missing primary timestamps are filled from available alternatives.
    """
    if "submissions" not in raw_data:
        raise KeyError("raw_data must contain a 'submissions' table")

    df = raw_data["submissions"].copy()

    required = {"user", "activity", "result", "timestamp"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing submission columns: {missing}")

    df["student_id"] = df["user"].astype(str)
    df["question_id"] = df["activity"].astype(str)

    result = df["result"].fillna("").astype(str).str.strip().str.lower()

    df["outcome"] = pd.NA
    df.loc[result.str.startswith("success"), "outcome"] = 1
    df.loc[result.str.startswith("fail"), "outcome"] = 0

    # Parse primary timestamps, then use fallback columns where needed.
    df["timestamp"] = _parse_timestamp_column(df["timestamp"])
    df["timestamp_source"] = "timestamp"

    for column in ["timisOrderedmp", "timismp"]:
        if column in df.columns:
            fallback = _parse_timestamp_column(df[column])
            missing_time = df["timestamp"].isna() & fallback.notna()

            df.loc[missing_time, "timestamp"] = fallback[missing_time]
            df.loc[missing_time, "timestamp_source"] = column

    # Keep only valid binary outcomes and usable timestamps.
    df = df[
        df["outcome"].notna() &
        df["timestamp"].notna()
    ].copy()

    df["outcome"] = df["outcome"].astype(int)

    output_columns = [
        "student_id",
        "question_id",
        "timestamp",
        "outcome"
    ]

    for column in ["result", "version", "timestamp_source"]:
        if column in df.columns:
            output_columns.append(column)

    return (
        df[output_columns]
        .sort_values(["student_id", "timestamp"])
        .reset_index(drop=True)
    )


def map_problems_to_concepts(problem_data, mapping_data=None):
    """
    Validate and apply the reviewed problem-to-concept mapping.

    If mapping_data is supplied, problem_data is merged with it by
    question_id. Otherwise, problem_data must already contain concept_id.

    This function does not invent or infer concept assignments.
    """
    required = {
        "question_id", "concept_id", "mapping_reason", "confidence"
    }

    if mapping_data is not None:
        missing = {"question_id"} - set(problem_data.columns)
        if missing:
            raise ValueError(f"Problem data missing columns: {missing}")

        missing_mapping = required - set(mapping_data.columns)
        if missing_mapping:
            raise ValueError(
                f"Mapping data missing columns: {missing_mapping}"
            )

        if mapping_data["question_id"].duplicated().any():
            raise ValueError("Mapping data contains duplicate question IDs")

        result = problem_data.drop(
            columns=["concept_id", "mapping_reason", "confidence"],
            errors="ignore"
        ).merge(
            mapping_data[list(required)],
            on="question_id",
            how="left",
            validate="many_to_one"
        )

        if result["concept_id"].isna().any():
            raise ValueError("Some questions have no concept mapping")

        return result

    missing = required - set(problem_data.columns)
    if missing:
        raise ValueError(
            f"Problem data must contain reviewed mapping columns: {missing}"
        )

    if problem_data["question_id"].duplicated().any():
        raise ValueError("Mapping data contains duplicate question IDs")

    return problem_data.copy()


def build_student_sequences(interactions):
    """
    Build chronological sequences grouped by student and DSA concept.

    Unmapped interactions and invalid outcomes are excluded.
    Returns a dictionary keyed by (student_id, concept_id).
    """
    required = {
        "student_id", "concept_id", "question_id", "timestamp", "outcome"
    }

    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Missing interaction columns: {missing}")

    data = interactions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["outcome"] = pd.to_numeric(data["outcome"], errors="coerce")

    data = data[
        data["concept_id"].notna() &
        (data["concept_id"] != "unmapped") &
        data["outcome"].isin([0, 1]) &
        data["timestamp"].notna()
    ].copy()

    data["student_id"] = data["student_id"].astype(str)
    data = data.sort_values(["student_id", "timestamp"])

    sequences = defaultdict(list)

    for row in data.itertuples(index=False):
        key = (row.student_id, row.concept_id)

        sequences[key].append({
            "question_id": row.question_id,
            "timestamp": row.timestamp.isoformat(),
            "outcome": int(row.outcome)
        })

    return dict(sequences)
