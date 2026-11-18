# LearnSync AI

**An Agentic Personalized Learning Tutor with Dynamic Curriculum Adaptation for Data Structures & Algorithms**

LearnSync AI is an AI-powered educational system that personalizes learning trajectories for computer science students mastering Data Structures and Algorithms (DSA). The system continuously traces conceptual mastery, detects misconceptions in student reasoning, retrieves educational content via Retrieval-Augmented Generation (RAG), and adaptively directs student curriculum progression.

---

## Architecture & System Overview

LearnSync AI operates as a unified multi-agent system comprising three coordinated modules:

```text
                       [ Student ]
                           │
                           ▼
                  [ Streamlit UI (app.py) ]
                           │
                           ▼
            [ Central Orchestrator (Member 3) ]
             │             │               │
             ▼             ▼               ▼
     [ Assessment & BKT ] [ RAG & Evaluator ] [ Curriculum & DB ]
        (Member 1)           (Member 2)            (Member 3)
```

### End-to-End Learning Cycle

```text
1. Student enters application / selects student ID
        ↓
2. Mastery profile retrieved from Database (PostgreSQL or local SQLite)
        ↓
3. Adaptive Curriculum Planner selects optimal concept & activity
   (Respecting strict prerequisites & pedagogical thresholds)
        ↓
4. RAG Content Agent retrieves relevant OpenDSA curriculum chunks
        ↓
5. LLM / Grounded Synthesizer generates mastery-conditioned explanation
        ↓
6. Assessment Question served to student
        ↓
7. Student submits answer
        ↓
8. Answer Evaluator performs classification & misconception detection
        ↓
9. BKT Engine updates Bayesian Knowledge Tracing posterior mastery
        ↓
10. Database persists attempt, evaluation JSON, and updated mastery profile
        ↓
11. Curriculum evaluates updated student state & recommends next activity
```

---

## Module Breakdown & Team Contributions

### Member 1: Knowledge Tracing & Assessment
- **Assessment Engine (`data/assessment.py`)**: 13 canonical DSA assessment questions covering all core concepts with deterministic matching and formative feedback.
- **Bayesian Knowledge Tracing (`data/bkt_engine.py`)**: Standard 4-parameter BKT model ($P(L_0)=0.20$, $P(T)=0.10$, $P(G)=0.20$, $P(S)=0.10$) updating prior belief to posterior mastery after each observation.
- **Dataset Pipeline (`data/data_pipeline.py`)**: Preprocessing and concept mapping for real student interaction datasets.
- **Canonical Concepts (13)**: `arrays`, `linked_lists`, `stacks`, `queues`, `recursion`, `trees`, `binary_search_trees`, `heaps`, `sorting`, `hashing`, `graphs`, `bfs`, `dfs`.

### Member 2: RAG & Natural Language Answer Evaluation
- **RAG Content Agent (`Member_2/rag.py`)**: Dense vector retrieval using `sentence-transformers` (`all-MiniLM-L6-v2`) and FAISS (`IndexFlatIP` cosine similarity) over 39 curated OpenDSA curriculum chunks.
- **Mastery-Conditioned Generation**: Adapts explanations into Foundational (<0.40), Intermediate (0.40–0.70), or Advanced (>0.70) tiers using Gemini (`gemini-2.5-flash`) with citation verification and transparent deterministic offline fallback.
- **Answer Evaluator (`Member_2/evaluator.py`)**: Rule-based semantic answer evaluator with invariant coverage checking, clause-aware negation/contrast detection, and specific DSA misconception identification.
- **BKT Outcome Conversion**: Bridges continuous/partial scores to binary BKT outcomes (`score >= 0.70 -> 1`, else `0`).

### Member 3: Database, Adaptive Curriculum & Orchestration
- **Dual-Backend Database (`Member_3/database.py`)**: Supports PostgreSQL for deployment (via `DATABASE_URL`) with an automatic, zero-config local SQLite fallback (`learnsync.db`) for laptops and demonstration.
- **Curriculum Planner (`Member_3/curriculum.py`)**: Prerequisite-aware activity selector with mastery thresholds:
  - Revision ($< 0.40$)
  - Targeted Practice ($0.40 \le m < 0.70$)
  - Reassessment ($0.70 \le m < 0.85$)
  - Concept Advancement ($\ge 0.85$)
- **Central Orchestrator (`Member_3/orchestrator.py`)**: Wires real Member 1, Member 2, and Member 3 components into an end-to-end learning cycle without mock dependencies.
- **Interactive Web App (`app.py`)**: Production-ready Streamlit application.

---

## Technology Stack

- **Frontend**: Streamlit, Plotly
- **Backend & Orchestration**: Python 3.10+, SQLite3 / PostgreSQL (`psycopg2-binary`)
- **Machine Learning & NLP**: `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `numpy`, `pandas`
- **Generative AI**: Official Google GenAI SDK (`google-genai`), Gemini 2.5 Flash
- **Knowledge Tracing**: Bayesian Knowledge Tracing (BKT)

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 recommended.
- Git (optional, for version control).

### 2. Clone / Open the Repository
```bash
cd LearnSyncAI
```

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Copy the `.env.example` template to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```env
# Optional: Google Gemini API Key for generative LLM explanations
# If unset, LearnSync AI uses the grounded offline synthesizer automatically.
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: PostgreSQL Database URL
# If unset, LearnSync AI automatically uses a local SQLite database (learnsync.db).
DATABASE_URL=
```

> **Note on Offline Capability:** LearnSync AI is fully runnable completely offline without internet or API keys. When no `GEMINI_API_KEY` is present, explanations are synthesized deterministically directly from retrieved OpenDSA sources.

---

## Database Configuration

### Option A: Local SQLite (Default, Zero Config)
Leave `DATABASE_URL` empty or unset in `.env`. The database automatically creates and initializes `learnsync.db` in the project root on first use.

### Option B: PostgreSQL (Production / Colab)
1. Install PostgreSQL and create a database:
   ```sql
   CREATE DATABASE learnsync;
   ```
2. Initialize tables using `Member_3/schema.sql`:
   ```bash
   psql -U postgres -d learnsync -f Member_3/schema.sql
   ```
3. Set the connection string in `.env`:
   ```env
   DATABASE_URL=postgresql://postgres:password@localhost:5432/learnsync
   ```

---

## Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Once started, open your browser at `http://localhost:8501`.

### Application Features & Navigation:
1. **Student Dashboard (`📊 Dashboard`)**: Overview of concept mastery, Plotly mastery distribution charts, progress metrics, and recommended next action.
2. **Initial Assessment (`📋 Initial Assessment`)**: Diagnostic assessment across all 13 DSA concepts to initialize student mastery profiles.
3. **Learning Session (`📚 Learning Session`)**: Adaptive concept lesson, personalized explanation conditioned on mastery level, and retrieved OpenDSA chunks.
4. **Answer Submission (`✍️ Submit Answer`)**: Question evaluation, feedback display, misconception diagnostics, and instant BKT mastery update.
5. **Adaptive Recommendations (`🎯 Recommendations`)**: Curriculum engine insights detailing why concepts were recommended.
6. **Learning History (`📈 Learning History`)**: Complete timeline of attempts, evaluations, and curriculum decisions.

---

## Running Tests

LearnSync AI includes an integrated test suite verifying all individual modules, integration interfaces, and end-to-end learning cycles:

```bash
python tests/test_integration.py
```

Expected output:
```text
======================================================================
LearnSync AI — Integrated Test Suite
======================================================================
...
Ran 54 tests in 17.397s

OK
======================================================================
ALL 54 TESTS PASSED
======================================================================
```

To run Member 2 specific evaluation benchmarks:
```bash
cd Member_2
python test_member2.py
python run_heldout_eval.py
```

---

## Dataset Information

1. **Zenodo Java Code Interactions (`data/java_submissions.csv`)**: Historical student coding interactions for knowledge tracing validation.
2. **Concept Mapping (`data/concept_mapping.csv`)**: Reviewed mapping linking practice problems to canonical DSA concepts.
3. **OpenDSA Knowledge Base (`Member_2/opendsa_chunks.csv`)**: 39 curated semantic text chunks extracted and validated from OpenDSA curriculum materials.
4. **Short Answer Development Set (`Member_2/short_answer_dataset.csv`)**: 117 natural language responses with ground-truth classifications, scores, and expected misconceptions across all 13 DSA concepts.
5. **Held-Out Evaluation Set (`Member_2/heldout_answers.csv`)**: 26 out-of-sample student answers for generalization testing.

---

## Project Structure

```text
LearnSyncAI/
│
├── app.py                      # Production Streamlit Web Application
├── requirements.txt            # Unified project dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git exclusion rules
├── README.md                   # Complete system documentation
│
├── src/                        # Source package initialization
│   └── __init__.py
│
├── data/                       # Member 1: Assessment & Knowledge Tracing
│   ├── assessment.py           # 13 DSA assessment questions & MCQ evaluation
│   ├── bkt_engine.py           # Bayesian Knowledge Tracing engine
│   ├── data_pipeline.py        # Zenodo interactions dataset pipeline
│   ├── concept_mapping.csv     # Canonical problem-to-concept mapping
│   ├── java_submissions.csv    # Student submission interactions
│   ├── tst_questions.csv       # Question catalog
│   └── distribution_per_question.csv
│
├── Member_1/
│   └── dataset_analysis.ipynb  # Member 1 exploratory data analysis
│
├── Member_2/                   # Member 2: RAG & Semantic Evaluator
│   ├── rag.py                  # FAISS + SentenceTransformer RAG pipeline
│   ├── evaluator.py            # Rule-based semantic answer evaluator
│   ├── opendsa_chunks.csv      # 39 curated OpenDSA knowledge chunks
│   ├── short_answer_dataset.csv# 117 development answer samples
│   ├── heldout_answers.csv     # 26 held-out test samples
│   ├── run_heldout_eval.py     # Held-out benchmark runner
│   ├── test_member2.py         # Member 2 unit tests
│   └── rag_development.ipynb  # Member 2 development notebook
│
├── Member_3/                   # Member 3: Database, Curriculum & Orchestrator
│   ├── database.py             # Dual SQLite / PostgreSQL database layer
│   ├── schema.sql              # Relational database schema (4 tables)
│   ├── curriculum.py           # Adaptive prerequisite & threshold planner
│   ├── orchestrator.py         # Central pipeline orchestrator
│   └── mock_agents.py          # Unit test mock implementations
│
└── tests/
    └── test_integration.py     # 54 comprehensive unit & integration tests
```

---

## Known Limitations & Honest Disclosure

1. **Evaluator Generalization on Held-Out Data**: On the 26 held-out natural language answers (`heldout_answers.csv`), the deterministic rule-based evaluator achieved 34.62% accuracy (9/26), exhibiting a bias toward `partially_correct` when student wording differs from canonical invariant patterns. In the interactive application, MCQ assessment questions (`assessment.py`) provide deterministic 100% precision.
2. **Hardware Constraints**: Sentence transformer embeddings (`all-MiniLM-L6-v2`) require approximately ~200MB RAM. The system is optimized for laptops with 8GB RAM and includes lazy loading to minimize memory footprint.
3. **OpenDSA Coverage**: The knowledge base contains 39 curated passages covering core DSA concepts. Advanced niche algorithmic variants will trigger the transparent "Context Insufficient" fallback.
