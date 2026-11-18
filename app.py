"""
LearnSync AI — Streamlit Application (app.py)
=============================================
Personalized learning tutor for Data Structures and Algorithms.

Run with:
    streamlit run app.py

Requires:
    pip install -r requirements.txt
    Configure .env (optional — defaults to SQLite and offline explanations)
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make all member modules importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.resolve()
for _p in [
    str(_ROOT / "data"),
    str(_ROOT / "Member_2"),
    str(_ROOT / "Member_3"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ROOT / ".env")
except ImportError:
    pass

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LearnSync AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4361ee;
        margin-bottom: 1rem;
    }
    .concept-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-low { background: #fee2e2; color: #991b1b; }
    .badge-mid { background: #fef3c7; color: #92400e; }
    .badge-high { background: #d1fae5; color: #065f46; }
    .feedback-correct { background: #d1fae5; border-radius: 8px; padding: 1rem; border-left: 4px solid #10b981; }
    .feedback-partial { background: #fef3c7; border-radius: 8px; padding: 1rem; border-left: 4px solid #f59e0b; }
    .feedback-incorrect { background: #fee2e2; border-radius: 8px; padding: 1rem; border-left: 4px solid #ef4444; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4361ee;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy orchestrator import with error handling
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LearnSync AI modules...")
def _load_orchestrator():
    try:
        import orchestrator
        return orchestrator, None
    except Exception as e:
        return None, str(e)

orch, orch_error = _load_orchestrator()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "student_id": None,
        "page": "home",
        "assessment_idx": 0,
        "assessment_answers": {},
        "current_question": None,
        "current_cycle": None,
        "assessment_complete": False,
        "assessment_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ---------------------------------------------------------------------------
# Sidebar — Navigation & Student ID
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 LearnSync AI")
    st.markdown("*Personalized DSA Learning*")
    st.divider()

    # Student ID input
    student_id_input = st.text_input(
        "Student ID",
        value=st.session_state.student_id or "",
        placeholder="e.g. student_001",
        help="Enter your student ID to begin or resume your session."
    )
    if student_id_input.strip():
        clean_id = student_id_input.strip().lower().replace(" ", "_")
        if clean_id != st.session_state.student_id:
            st.session_state.student_id = clean_id
            st.session_state.page = "dashboard"
            st.rerun()

    if st.session_state.student_id:
        st.success(f"Logged in as: **{st.session_state.student_id}**")
        st.divider()

        # Navigation
        pages = {
            "📊 Dashboard": "dashboard",
            "📋 Initial Assessment": "assessment",
            "📚 Learning Session": "learning",
            "✍️ Submit Answer": "submit",
            "🎯 Recommendations": "recommendations",
            "📈 Learning History": "history",
        }
        for label, page_key in pages.items():
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.page == page_key else "secondary"):
                st.session_state.page = page_key
                st.rerun()

    st.divider()

    # Backend info
    if orch:
        import database as _db_info
        st.caption(f"🗄️ {_db_info.get_backend_info()}")
    elif orch_error:
        st.error(f"Module error: {orch_error}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _mastery_color(score):
    if score < 0.40:
        return "#ef4444"
    elif score < 0.70:
        return "#f59e0b"
    return "#10b981"

def _mastery_label(score):
    if score < 0.40:
        return "Needs Revision"
    elif score < 0.70:
        return "Developing"
    elif score < 0.85:
        return "Proficient"
    return "Advanced"

def _mastery_badge_class(score):
    if score < 0.40:
        return "badge-low"
    elif score < 0.70:
        return "badge-mid"
    return "badge-high"

DSA_CONCEPTS = [
    "arrays", "linked_lists", "stacks", "queues", "recursion",
    "trees", "binary_search_trees", "heaps", "sorting",
    "hashing", "graphs", "bfs", "dfs"
]

CONCEPT_DISPLAY = {
    "arrays": "Arrays",
    "linked_lists": "Linked Lists",
    "stacks": "Stacks",
    "queues": "Queues",
    "recursion": "Recursion",
    "trees": "Trees",
    "binary_search_trees": "Binary Search Trees",
    "heaps": "Heaps",
    "sorting": "Sorting",
    "hashing": "Hashing",
    "graphs": "Graphs",
    "bfs": "BFS",
    "dfs": "DFS"
}

@st.cache_data(ttl=5, show_spinner=False)
def _get_dashboard(student_id):
    if not orch:
        return None
    try:
        return orch.get_student_dashboard(student_id)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if st.session_state.page == "home" or not st.session_state.student_id:
    st.markdown('<div class="hero-title">🧠 LearnSync AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">An Agentic Personalized Learning Tutor for Data Structures & Algorithms</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Adaptive Curriculum</h4>
            <p>Bayesian Knowledge Tracing continuously estimates your mastery and selects the optimal next topic.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📚 RAG Explanations</h4>
            <p>Content retrieved from curated OpenDSA materials is used to generate personalized explanations.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Misconception Detection</h4>
            <p>Rule-based evaluator identifies specific conceptual errors and provides targeted feedback.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Enter your Student ID in the sidebar to begin your personalized learning journey.")


# ---------------------------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------------------------
elif st.session_state.page == "dashboard":
    st.markdown('<div class="hero-title">📊 Student Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"**Student:** `{st.session_state.student_id}`")

    with st.spinner("Loading your progress..."):
        dash = _get_dashboard(st.session_state.student_id)

    if not dash:
        st.error("Could not load dashboard. Check that modules are installed correctly.")
        st.stop()

    if "error" in dash:
        st.error(f"Dashboard error: {dash['error']}")
        st.stop()

    mastery = dash.get("mastery_profile", {})
    recommendation = dash.get("curriculum_recommendation", {})
    history = dash.get("learning_history", [])
    attempts = dash.get("attempts", [])

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    assessed_concepts = [c for c in DSA_CONCEPTS if c in mastery]
    avg_mastery = sum(mastery.values()) / len(mastery) if mastery else 0.0

    col1.metric("Concepts Assessed", f"{len(assessed_concepts)}/13")
    col2.metric("Avg Mastery", f"{avg_mastery:.0%}")
    col3.metric("Total Attempts", len(attempts))
    col4.metric("History Records", len(history))

    # Mastery chart
    if mastery:
        st.markdown('<div class="section-header">Mastery by Concept</div>', unsafe_allow_html=True)
        chart_concepts = [CONCEPT_DISPLAY.get(c, c) for c in DSA_CONCEPTS if c in mastery]
        chart_values = [mastery[c] for c in DSA_CONCEPTS if c in mastery]
        colors = [_mastery_color(v) for v in chart_values]

        fig = go.Figure(go.Bar(
            x=chart_concepts,
            y=chart_values,
            marker_color=colors,
            text=[f"{v:.0%}" for v in chart_values],
            textposition="outside"
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 1.15], title="Mastery", tickformat=".0%"),
            xaxis=dict(title="Concept"),
            showlegend=False,
            margin=dict(t=10, b=10),
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        # Threshold lines
        fig.add_hline(y=0.70, line_dash="dash", line_color="#f59e0b",
                      annotation_text="Practice threshold (0.70)", annotation_position="top right")
        fig.add_hline(y=0.85, line_dash="dash", line_color="#10b981",
                      annotation_text="Advance threshold (0.85)", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mastery data yet. Complete the Initial Assessment to get started.")

    # Recommendation
    if recommendation and "concept_id" in recommendation:
        st.markdown('<div class="section-header">📌 Current Recommendation</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            rec_concept = CONCEPT_DISPLAY.get(recommendation["concept_id"], recommendation["concept_id"])
            rec_mastery = mastery.get(recommendation["concept_id"], None)
            st.metric("Next Concept", rec_concept,
                      delta=f"{rec_mastery:.0%} mastery" if rec_mastery is not None else "Not yet assessed")
            st.metric("Activity", recommendation["activity"].capitalize())
        with col2:
            st.markdown(f"**Why:** {recommendation.get('reason', 'N/A')}")
            st.markdown(f"**Next Action:** `{recommendation.get('next_action', 'N/A')}`")

    # Recent activity
    if history:
        st.markdown('<div class="section-header">🕐 Recent Learning Activity</div>', unsafe_allow_html=True)
        import pandas as pd
        hist_df = pd.DataFrame(history[:5])
        for col_name in ["concept_id", "activity"]:
            if col_name in hist_df.columns:
                hist_df[col_name] = hist_df[col_name].str.replace("_", " ").str.title()
        display_cols = [c for c in ["concept_id", "activity", "reason", "created_at"] if c in hist_df.columns]
        st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# INITIAL ASSESSMENT PAGE
# ---------------------------------------------------------------------------
elif st.session_state.page == "assessment":
    st.markdown('<div class="hero-title">📋 Initial Assessment</div>', unsafe_allow_html=True)
    st.markdown("Answer one question per concept to initialize your mastery profile.")

    if not orch:
        st.error("Orchestrator not loaded.")
        st.stop()

    # Load all assessment questions
    @st.cache_data(show_spinner=False)
    def _get_all_questions():
        import assessment as a
        return a.get_assessment_questions()

    questions = _get_all_questions()

    if not questions:
        st.error("No assessment questions found.")
        st.stop()

    idx = st.session_state.assessment_idx
    total = len(questions)

    if st.session_state.assessment_complete:
        st.success("✅ Assessment Complete!")

        # Show results
        correct_count = sum(1 for r in st.session_state.assessment_results if r.get("outcome") == 1)
        st.metric("Score", f"{correct_count}/{total}")

        for res in st.session_state.assessment_results:
            icon = "✅" if res["outcome"] == 1 else "❌"
            with st.expander(f"{icon} {CONCEPT_DISPLAY.get(res['concept_id'], res['concept_id'])}"):
                st.markdown(res["feedback"])

        if st.button("🔄 Retake Assessment", type="secondary"):
            st.session_state.assessment_idx = 0
            st.session_state.assessment_answers = {}
            st.session_state.assessment_complete = False
            st.session_state.assessment_results = []
            _get_dashboard.clear()
            st.rerun()

        if st.button("📊 Go to Dashboard", type="primary"):
            _get_dashboard.clear()
            st.session_state.page = "dashboard"
            st.rerun()

    elif idx < total:
        q = questions[idx]
        concept = CONCEPT_DISPLAY.get(q["concept_id"], q["concept_id"])

        # Progress
        st.progress(idx / total)
        st.caption(f"Question {idx + 1} of {total} — Concept: **{concept}**")
        st.divider()

        st.markdown(f"### {q['question']}")
        st.markdown("")

        # MCQ options
        option_key = f"q_{q['question_id']}"
        selected = st.radio(
            "Select your answer:",
            options=list(q["options"].keys()),
            format_func=lambda k: f"**{k}:** {q['options'][k]}",
            key=option_key,
            index=None
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Submit ➡️", type="primary", disabled=(selected is None)):
                result = orch.submit_assessment_answer(
                    st.session_state.student_id,
                    q["question_id"],
                    selected
                )
                st.session_state.assessment_results.append({
                    "question_id": q["question_id"],
                    "concept_id": q["concept_id"],
                    "outcome": result["outcome"],
                    "feedback": result["feedback"],
                    "answer": selected
                })

                if idx + 1 >= total:
                    st.session_state.assessment_complete = True
                else:
                    st.session_state.assessment_idx = idx + 1

                _get_dashboard.clear()
                st.rerun()


# ---------------------------------------------------------------------------
# LEARNING SESSION PAGE
# ---------------------------------------------------------------------------
elif st.session_state.page == "learning":
    st.markdown('<div class="hero-title">📚 Learning Session</div>', unsafe_allow_html=True)

    if not orch:
        st.error("Orchestrator not loaded.")
        st.stop()

    with st.spinner("🧠 Selecting your next concept and generating explanation..."):
        try:
            cycle = orch.run_learning_cycle(st.session_state.student_id)
            st.session_state.current_cycle = cycle
        except Exception as e:
            st.error(f"Learning cycle error: {e}")
            st.stop()

    decision = cycle["curriculum_decision"]
    concept_id = decision["concept_id"]
    concept_display = CONCEPT_DISPLAY.get(concept_id, concept_id)
    mastery_score = cycle["mastery_profile"].get(concept_id, 0.0)

    # Concept header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"## 📖 {concept_display}")
    with col2:
        st.metric("Your Mastery", f"{mastery_score:.0%}")
    with col3:
        st.metric("Activity", decision["activity"].capitalize())

    st.caption(f"**Why this concept:** {decision['reason']}")
    st.divider()

    # Personalized explanation
    st.markdown('<div class="section-header">💡 Personalized Explanation</div>', unsafe_allow_html=True)
    explanation = cycle.get("explanation", "No explanation available.")
    st.markdown(explanation)

    # Retrieved sources
    context = cycle.get("retrieved_context", [])
    if context:
        with st.expander(f"📚 Retrieved OpenDSA Sources ({len(context)} chunks)"):
            for chunk in context:
                st.markdown(f"**{chunk.get('title', 'Source')}** — `{chunk.get('chunk_id', '')}` | "
                            f"Level: {chunk.get('difficulty_level', 'N/A')} | "
                            f"Score: {chunk.get('score', 0):.4f}")
                st.caption(chunk.get("source_url", ""))

    st.divider()

    # Question preview
    q = cycle.get("question")
    if q:
        st.markdown('<div class="section-header">❓ Practice Question</div>', unsafe_allow_html=True)
        st.markdown(f"**{q['question']}**")
        for key, text in q.get("options", {}).items():
            st.markdown(f"- **{key}:** {text}")

        if st.button("✍️ Submit My Answer", type="primary"):
            st.session_state.page = "submit"
            st.rerun()
    else:
        st.info("No question available for this concept right now.")


# ---------------------------------------------------------------------------
# SUBMIT ANSWER PAGE
# ---------------------------------------------------------------------------
elif st.session_state.page == "submit":
    st.markdown('<div class="hero-title">✍️ Submit Answer</div>', unsafe_allow_html=True)

    if not orch:
        st.error("Orchestrator not loaded.")
        st.stop()

    cycle = st.session_state.get("current_cycle")
    if not cycle:
        st.warning("No active learning session. Go to Learning Session first.")
        if st.button("📚 Go to Learning Session"):
            st.session_state.page = "learning"
            st.rerun()
        st.stop()

    q = cycle.get("question")
    concept_id = cycle["curriculum_decision"]["concept_id"]
    concept_display = CONCEPT_DISPLAY.get(concept_id, concept_id)

    st.markdown(f"**Concept:** {concept_display}")

    if q:
        st.markdown(f"### {q['question']}")
        st.markdown("")

        # MCQ answer
        option_key = "submit_answer_radio"
        selected = st.radio(
            "Your answer:",
            options=list(q.get("options", {}).keys()),
            format_func=lambda k: f"**{k}:** {q['options'][k]}",
            key=option_key,
            index=None
        )

        if st.button("🚀 Submit Answer", type="primary", disabled=(selected is None)):
            with st.spinner("Evaluating your answer..."):
                result = orch.submit_assessment_answer(
                    st.session_state.student_id,
                    q["question_id"],
                    selected
                )

            # Feedback display
            classification = result.get("classification", "incorrect")
            feedback = result.get("feedback", "")

            if classification == "correct":
                st.markdown(
                    f'<div class="feedback-correct">✅ <strong>Correct!</strong><br>{feedback}</div>',
                    unsafe_allow_html=True
                )
            elif classification == "partially_correct":
                st.markdown(
                    f'<div class="feedback-partial">⚠️ <strong>Partially Correct</strong><br>{feedback}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="feedback-incorrect">❌ <strong>Incorrect</strong><br>{feedback}</div>',
                    unsafe_allow_html=True
                )

            # BKT update
            bkt = result.get("bkt_update", {})
            if bkt:
                col1, col2, col3 = st.columns(3)
                col1.metric("Prior Mastery", f"{bkt.get('prior_mastery', 0):.0%}")
                col2.metric("Updated Mastery", f"{bkt.get('mastery', 0):.0%}",
                            delta=f"{bkt.get('mastery', 0) - bkt.get('prior_mastery', 0):+.0%}")
                col3.metric("BKT Outcome", "✅ Correct" if bkt.get("outcome") == 1 else "❌ Incorrect")

            _get_dashboard.clear()

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📚 Continue Learning", type="primary"):
                    st.session_state.current_cycle = None
                    st.session_state.page = "learning"
                    st.rerun()
            with col2:
                if st.button("📊 View Dashboard"):
                    st.session_state.page = "dashboard"
                    st.rerun()


# ---------------------------------------------------------------------------
# ADAPTIVE RECOMMENDATIONS PAGE
# ---------------------------------------------------------------------------
elif st.session_state.page == "recommendations":
    st.markdown('<div class="hero-title">🎯 Adaptive Recommendations</div>', unsafe_allow_html=True)

    if not orch:
        st.error("Orchestrator not loaded.")
        st.stop()

    with st.spinner("Analyzing your learning profile..."):
        dash = _get_dashboard(st.session_state.student_id)

    if not dash or "error" in dash:
        st.error("Could not load recommendations.")
        st.stop()

    mastery = dash.get("mastery_profile", {})
    recommendation = dash.get("curriculum_recommendation", {})

    if "concept_id" not in recommendation:
        st.info("Complete the Initial Assessment to receive personalized recommendations.")
        if st.button("📋 Go to Assessment"):
            st.session_state.page = "assessment"
            st.rerun()
        st.stop()

    # Main recommendation
    concept_id = recommendation["concept_id"]
    concept_display = CONCEPT_DISPLAY.get(concept_id, concept_id)
    mastery_score = mastery.get(concept_id, 0.0)
    activity = recommendation["activity"]

    st.markdown(f"## 🎯 Recommended: **{concept_display}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Activity", activity.capitalize())
    col2.metric("Your Mastery", f"{mastery_score:.0%}")
    col3.metric("Status", _mastery_label(mastery_score))

    st.markdown(f"**Reasoning:** {recommendation.get('reason', 'N/A')}")
    st.markdown(f"**Next Action:** `{recommendation.get('next_action', 'N/A')}`")

    st.divider()

    # All concepts overview
    st.markdown('<div class="section-header">📊 Full Concept Overview</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, concept in enumerate(DSA_CONCEPTS):
        score = mastery.get(concept, None)
        label = CONCEPT_DISPLAY.get(concept, concept)
        with cols[i % 3]:
            if score is not None:
                badge_class = _mastery_badge_class(score)
                st.markdown(
                    f'<span class="concept-badge {badge_class}">{label}: {score:.0%}</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<span class="concept-badge" style="background:#f1f5f9;color:#64748b">'
                    f'{label}: Not assessed</span>',
                    unsafe_allow_html=True
                )

    st.divider()

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Start Learning Session", type="primary"):
            st.session_state.page = "learning"
            st.rerun()
    with col2:
        if st.button("📋 Take Assessment", type="secondary"):
            st.session_state.page = "assessment"
            st.rerun()


# ---------------------------------------------------------------------------
# LEARNING HISTORY PAGE
# ---------------------------------------------------------------------------
elif st.session_state.page == "history":
    st.markdown('<div class="hero-title">📈 Learning History</div>', unsafe_allow_html=True)

    if not orch:
        st.error("Orchestrator not loaded.")
        st.stop()

    with st.spinner("Loading your history..."):
        dash = _get_dashboard(st.session_state.student_id)

    if not dash or "error" in dash:
        st.error("Could not load history.")
        st.stop()

    history = dash.get("learning_history", [])
    attempts = dash.get("attempts", [])

    # Attempts table
    st.markdown('<div class="section-header">📝 Answer Attempts</div>', unsafe_allow_html=True)

    if attempts:
        import pandas as pd
        att_df = pd.DataFrame(attempts)
        # Clean up display
        for col_name in ["concept_id", "classification"]:
            if col_name in att_df.columns:
                att_df[col_name] = att_df[col_name].fillna("").str.replace("_", " ").str.title()
        display_cols = [c for c in ["question_id", "concept_id", "classification", "outcome", "attempted_at"]
                        if c in att_df.columns]
        # Color coding
        def _color_row(row):
            if str(row.get("outcome")) == "1":
                return ["background-color: #d1fae5"] * len(row)
            elif row.get("classification", "").lower() == "partially correct":
                return ["background-color: #fef3c7"] * len(row)
            elif str(row.get("outcome")) == "0":
                return ["background-color: #fee2e2"] * len(row)
            return [""] * len(row)

        st.dataframe(
            att_df[display_cols].style.apply(_color_row, axis=1),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No answer attempts yet.")

    # Learning history timeline
    st.markdown('<div class="section-header">🗓️ Curriculum Decisions</div>', unsafe_allow_html=True)

    if history:
        for record in history[:10]:
            concept = CONCEPT_DISPLAY.get(record.get("concept_id", ""), record.get("concept_id", ""))
            activity = record.get("activity", "").capitalize()
            reason = record.get("reason", "")
            created = record.get("created_at", "")

            activity_icon = {
                "Revision": "📖",
                "Practice": "✏️",
                "Reassessment": "🔄",
                "Advance": "⬆️"
            }.get(activity, "📌")

            with st.expander(f"{activity_icon} {activity} — {concept} ({created})"):
                st.markdown(f"**Reason:** {reason}")
                st.markdown(f"**Next Action:** `{record.get('next_action', 'N/A')}`")
    else:
        st.info("No learning history yet.")

    # Mastery progression chart (if attempts exist)
    if attempts and len(attempts) > 1:
        st.markdown('<div class="section-header">📉 Mastery Progression</div>', unsafe_allow_html=True)
        st.info("Track individual concept mastery over time as you complete more sessions.")
