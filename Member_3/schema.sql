
-- ==========================================
-- LearnSync AI
-- Member 3: Database Schema
-- ==========================================

-- 1. STUDENTS TABLE
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);


-- 2. ATTEMPTS TABLE
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL,
    question_id VARCHAR(100) NOT NULL,
    concept_id VARCHAR(50) NOT NULL,
    classification VARCHAR(30),
    outcome INTEGER,
    evaluation JSONB,
    attempted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (student_id)
        REFERENCES students(student_id),

    CHECK (outcome IN (0, 1) OR outcome IS NULL),

    CHECK (
        classification IN (
            'correct',
            'partially_correct',
            'incorrect'
        ) OR classification IS NULL
    )
);


-- 3. MASTERY TABLE
CREATE TABLE IF NOT EXISTS mastery (
    student_id VARCHAR(100) NOT NULL,
    concept_id VARCHAR(50) NOT NULL,
    mastery FLOAT NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (student_id, concept_id),

    FOREIGN KEY (student_id)
        REFERENCES students(student_id),

    CHECK (mastery >= 0 AND mastery <= 1)
);


-- 4. LEARNING HISTORY TABLE
CREATE TABLE IF NOT EXISTS learning_history (
    history_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL,
    concept_id VARCHAR(50) NOT NULL,
    activity VARCHAR(30) NOT NULL,
    reason TEXT,
    next_action VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (student_id)
        REFERENCES students(student_id),

    CHECK (
        activity IN (
            'revision',
            'practice',
            'reassessment',
            'advance'
        )
    )
);
