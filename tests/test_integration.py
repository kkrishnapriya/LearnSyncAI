"""
LearnSync AI — Integrated Test Suite
=====================================
Tests Member 1, Member 2, Member 3, and their integration.

Run with:
    cd LearnSyncAI
    python tests/test_integration.py

Or using pytest:
    pytest tests/test_integration.py -v
"""

import sys
import os
import unittest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    os.path.join(_ROOT, "data"),
    os.path.join(_ROOT, "Member_2"),
    os.path.join(_ROOT, "Member_3"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force offline mode for sandbox / CI environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ===========================================================================
# MEMBER 1 TESTS
# ===========================================================================

class TestMember1Assessment(unittest.TestCase):
    """Unit tests for Member 1: Assessment module."""

    def setUp(self):
        import assessment
        self.assessment = assessment

    def test_get_all_questions(self):
        questions = self.assessment.get_assessment_questions()
        self.assertIsInstance(questions, list)
        self.assertEqual(len(questions), 13, "Should have 13 questions (one per concept)")

    def test_get_questions_by_concept(self):
        questions = self.assessment.get_assessment_questions(concept_id="stacks")
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["concept_id"], "stacks")
        self.assertEqual(questions[0]["question_id"], "assess_stacks_01")

    def test_question_has_no_answer(self):
        """Correct answer must NOT be exposed to students."""
        questions = self.assessment.get_assessment_questions()
        for q in questions:
            self.assertNotIn("correct_answer", q)

    def test_evaluate_correct_answer(self):
        result = self.assessment.evaluate_assessment_answer("assess_arrays_01", "A")
        self.assertEqual(result["outcome"], 1)
        self.assertEqual(result["classification"], "correct")
        self.assertEqual(result["concept_id"], "arrays")

    def test_evaluate_incorrect_answer(self):
        result = self.assessment.evaluate_assessment_answer("assess_arrays_01", "B")
        self.assertEqual(result["outcome"], 0)
        self.assertEqual(result["classification"], "incorrect")

    def test_evaluate_by_text(self):
        """Should accept option text as well as letter."""
        result = self.assessment.evaluate_assessment_answer("assess_arrays_01", "O(1)")
        self.assertEqual(result["outcome"], 1)

    def test_evaluate_unknown_question(self):
        with self.assertRaises(ValueError):
            self.assessment.evaluate_assessment_answer("UNKNOWN_Q", "A")

    def test_evaluate_empty_answer(self):
        with self.assertRaises(ValueError):
            self.assessment.evaluate_assessment_answer("assess_arrays_01", "")

    def test_all_concepts_covered(self):
        all_concepts = {q["concept_id"] for q in self.assessment.get_assessment_questions()}
        expected = {
            "arrays", "linked_lists", "stacks", "queues", "recursion",
            "trees", "binary_search_trees", "heaps", "sorting",
            "hashing", "graphs", "bfs", "dfs"
        }
        self.assertEqual(all_concepts, expected)


class TestMember1BKT(unittest.TestCase):
    """Unit tests for Member 1: Bayesian Knowledge Tracing."""

    def setUp(self):
        import bkt_engine
        self.bkt = bkt_engine
        self.bkt.reset_bkt_state()

    def tearDown(self):
        self.bkt.reset_bkt_state()

    def test_initial_mastery(self):
        profile = self.bkt.get_mastery_profile("new_student")
        self.assertEqual(profile["mastery"], {})

    def test_update_bkt_correct(self):
        result = self.bkt.update_bkt("s1", "arrays", 1)
        self.assertIn("mastery", result)
        self.assertGreater(result["mastery"], self.bkt.P_INIT)

    def test_update_bkt_incorrect(self):
        result = self.bkt.update_bkt("s1", "arrays", 0)
        self.assertIn("mastery", result)
        # With incorrect answer, mastery stays near P_INIT but may differ

    def test_mastery_increases_with_correct_answers(self):
        for _ in range(5):
            self.bkt.update_bkt("s2", "stacks", 1)
        profile = self.bkt.get_mastery_profile("s2")
        self.assertGreater(profile["mastery"]["stacks"], self.bkt.P_INIT)

    def test_mastery_profile_format(self):
        self.bkt.update_bkt("s3", "trees", 1)
        profile = self.bkt.get_mastery_profile("s3")
        self.assertIn("student_id", profile)
        self.assertIn("mastery", profile)
        self.assertIsInstance(profile["mastery"], dict)

    def test_invalid_outcome(self):
        with self.assertRaises(ValueError):
            self.bkt.update_bkt("s1", "arrays", 2)

    def test_multiple_concepts(self):
        self.bkt.update_bkt("s4", "arrays", 1)
        self.bkt.update_bkt("s4", "stacks", 0)
        profile = self.bkt.get_mastery_profile("s4")
        self.assertIn("arrays", profile["mastery"])
        self.assertIn("stacks", profile["mastery"])


# ===========================================================================
# MEMBER 2 TESTS
# ===========================================================================

class TestMember2Evaluator(unittest.TestCase):
    """Unit tests for Member 2: Answer Evaluator."""

    @classmethod
    def setUpClass(cls):
        import evaluator
        cls.evaluator = evaluator
        evaluator._load_questions()

    def test_correct_answer(self):
        ans = "A stack operates on Last-In First-Out (LIFO) discipline. Push and pop both take O(1) time."
        result = self.evaluator.evaluate_answer("Q_STK_01", ans)
        self.assertEqual(result["classification"], "correct")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["bkt_outcome"], 1)
        self.assertEqual(result["concept_id"], "stacks")
        self.assertIsInstance(result["identified_misconceptions"], list)
        self.assertIsInstance(result["evidence"], list)

    def test_partially_correct_answer(self):
        ans = "A stack operates on LIFO where you push to the top and pop from the top."
        result = self.evaluator.evaluate_answer("Q_STK_01", ans)
        self.assertEqual(result["classification"], "partially_correct")
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["bkt_outcome"], 0)

    def test_misconception_detected(self):
        ans = "A stack is First-In First-Out (FIFO) like a supermarket queue."
        result = self.evaluator.evaluate_answer("Q_STK_01", ans)
        self.assertEqual(result["classification"], "incorrect")
        self.assertIn("Confuses LIFO with FIFO", result["identified_misconceptions"])

    def test_bkt_outcome_rule(self):
        self.assertEqual(self.evaluator.bkt_outcome_from_evaluation({"score": 1.0}), 1)
        self.assertEqual(self.evaluator.bkt_outcome_from_evaluation({"score": 0.70}), 1)
        self.assertEqual(self.evaluator.bkt_outcome_from_evaluation({"score": 0.69}), 0)
        self.assertEqual(self.evaluator.bkt_outcome_from_evaluation({"score": 0.5}), 0)
        self.assertEqual(self.evaluator.bkt_outcome_from_evaluation({"score": 0.0}), 0)

    def test_unknown_question_handled(self):
        result = self.evaluator.evaluate_answer("UNKNOWN_Q_999", "Some answer")
        self.assertEqual(result["classification"], "incorrect")
        self.assertEqual(result["bkt_outcome"], 0)
        self.assertIn("concept_id", result)

    def test_empty_answer_handled(self):
        result = self.evaluator.evaluate_answer("Q_STK_01", "")
        self.assertEqual(result["classification"], "incorrect")
        self.assertEqual(result["score"], 0.0)

    def test_negation_not_flagged(self):
        """FIFO mentioned in contrast with LIFO should NOT trigger FIFO misconception."""
        ans = "A stack operates on LIFO with O(1) push and pop, unlike a queue which is FIFO."
        result = self.evaluator.evaluate_answer("Q_STK_01", ans)
        self.assertNotIn("Confuses LIFO with FIFO", result["identified_misconceptions"])

    def test_output_schema(self):
        """Every field in the contract schema must be present."""
        result = self.evaluator.evaluate_answer("Q_STK_01", "LIFO means last in, first out. Push and pop are O(1).")
        for key in ["question_id", "concept_id", "classification", "score",
                    "bkt_outcome", "identified_misconceptions", "feedback", "evidence"]:
            self.assertIn(key, result, f"Missing field: {key}")


class TestMember2RAG(unittest.TestCase):
    """Unit tests for Member 2: RAG pipeline."""

    @classmethod
    def setUpClass(cls):
        import rag
        cls.rag = rag
        docs = rag.load_opendsa_content()
        cls.index, cls.docs = rag.build_vector_index(docs)

    def test_opendsa_loaded(self):
        self.assertGreater(len(self.docs), 0)

    def test_all_docs_have_required_fields(self):
        required = {"chunk_id", "concept_id", "title", "content", "source_url"}
        for doc in self.docs:
            for field in required:
                self.assertIn(field, doc)

    def test_retrieve_context_with_concept(self):
        results = self.rag.retrieve_context("What is LIFO in stacks?", concept_id="stacks", top_k=2)
        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["concept_id"], "stacks")
        self.assertIn("score", results[0])

    def test_retrieve_context_all_results_match_concept(self):
        results = self.rag.retrieve_context("recursion base case", concept_id="recursion", top_k=3)
        for r in results:
            self.assertEqual(r["concept_id"], "recursion")

    def test_empty_query_returns_empty(self):
        results = self.rag.retrieve_context("", concept_id="stacks")
        self.assertEqual(results, [])

    def test_generate_explanation_foundational(self):
        ctx = self.rag.retrieve_context("Explain stacks", concept_id="stacks", top_k=2)
        exp = self.rag.generate_explanation("Explain stacks", ctx, mastery_level=0.2)
        self.assertIn("Foundational Tier", exp)
        self.assertIn("OpenDSA Sources", exp)

    def test_generate_explanation_intermediate(self):
        ctx = self.rag.retrieve_context("Explain stacks", concept_id="stacks", top_k=2)
        exp = self.rag.generate_explanation("Explain stacks", ctx, mastery_level=0.55)
        self.assertIn("Intermediate Tier", exp)

    def test_generate_explanation_advanced(self):
        ctx = self.rag.retrieve_context("Explain stacks", concept_id="stacks", top_k=2)
        exp = self.rag.generate_explanation("Explain stacks", ctx, mastery_level=0.85)
        self.assertIn("Advanced Tier", exp)

    def test_generate_explanation_no_context(self):
        exp = self.rag.generate_explanation("What is X?", [], mastery_level=0.5)
        self.assertIn("Context Insufficient", exp)


# ===========================================================================
# MEMBER 3 TESTS
# ===========================================================================

class TestMember3Database(unittest.TestCase):
    """Unit tests for Member 3: Database layer (SQLite)."""

    @classmethod
    def setUpClass(cls):
        import database
        cls.db = database
        cls.test_student = "test_db_student_integration"
        database.create_student(cls.test_student)

    def test_create_student(self):
        result = self.db.create_student("test_new_student_xyz")
        self.assertTrue(result)

    def test_create_student_idempotent(self):
        """Calling create_student twice should not raise."""
        self.db.create_student("test_idempotent")
        result = self.db.create_student("test_idempotent")
        self.assertTrue(result)

    def test_save_and_load_mastery(self):
        mastery = {"arrays": 0.72, "stacks": 0.45}
        self.db.save_mastery(self.test_student, mastery)
        loaded = self.db.load_mastery(self.test_student)
        self.assertAlmostEqual(loaded["arrays"], 0.72, places=4)
        self.assertAlmostEqual(loaded["stacks"], 0.45, places=4)

    def test_save_mastery_wrapped_format(self):
        """Accepts {'student_id': ..., 'mastery': {...}} format."""
        wrapped = {"student_id": self.test_student, "mastery": {"queues": 0.33}}
        self.db.save_mastery(self.test_student, wrapped)
        loaded = self.db.load_mastery(self.test_student)
        self.assertAlmostEqual(loaded["queues"], 0.33, places=4)

    def test_save_attempt(self):
        evaluation = {
            "classification": "correct",
            "outcome": 1,
            "bkt_outcome": 1,
            "score": 1.0,
            "feedback": "Correct!"
        }
        result = self.db.save_attempt(
            self.test_student, "assess_arrays_01", "arrays", evaluation
        )
        self.assertTrue(result)

    def test_get_student_attempts(self):
        evaluation = {
            "classification": "correct",
            "outcome": 1,
            "bkt_outcome": 1,
            "score": 1.0,
            "feedback": "Correct!"
        }
        self.db.save_attempt(
            self.test_student, "assess_arrays_01", "arrays", evaluation
        )
        attempts = self.db.get_student_attempts(self.test_student)
        self.assertIsInstance(attempts, list)
        self.assertGreater(len(attempts), 0)

    def test_save_and_get_learning_history(self):
        history_id = self.db.save_learning_history(
            self.test_student, "arrays", "practice",
            "Student needs practice", "generate_question"
        )
        self.assertIsNotNone(history_id)
        history = self.db.get_student_history(self.test_student)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

    def test_invalid_mastery_score(self):
        with self.assertRaises(ValueError):
            self.db.save_mastery(self.test_student, {"arrays": 1.5})

    def test_invalid_classification(self):
        evaluation = {"classification": "unknown_class", "outcome": 1}
        with self.assertRaises(ValueError):
            self.db.save_attempt(self.test_student, "q1", "arrays", evaluation)


class TestMember3Curriculum(unittest.TestCase):
    """Unit tests for Member 3: Curriculum module."""

    @classmethod
    def setUpClass(cls):
        import curriculum
        cls.curriculum = curriculum

    def test_new_student_gets_first_concept(self):
        """A student with no mastery should be assigned an unassessed concept."""
        decision = self.curriculum.select_next_activity(
            "new_s", {}, self.curriculum.DSA_PREREQUISITES, []
        )
        self.assertIn("concept_id", decision)
        self.assertEqual(decision["activity"], "reassessment")

    def test_low_mastery_gets_revision(self):
        # When all concepts are assessed and arrays has lowest mastery (< 0.40)
        mastery = {c: 0.80 for c in self.curriculum.DSA_CONCEPTS}
        mastery["arrays"] = 0.20
        decision = self.curriculum.select_next_activity(
            "s", mastery, self.curriculum.DSA_PREREQUISITES, []
        )
        # arrays has low mastery; expect revision
        self.assertEqual(decision["concept_id"], "arrays")
        self.assertEqual(decision["activity"], "revision")

    def test_advanced_mastery_gets_advance(self):
        mastery = {c: 0.90 for c in self.curriculum.DSA_CONCEPTS}
        decision = self.curriculum.select_next_activity(
            "s", mastery, self.curriculum.DSA_PREREQUISITES, []
        )
        self.assertEqual(decision["activity"], "advance")

    def test_strict_prerequisites_respected(self):
        """BST should not be selectable if trees mastery < threshold."""
        mastery = {c: 0.90 for c in self.curriculum.DSA_CONCEPTS if c != "trees"}
        mastery["trees"] = 0.20  # Below STRICT_PREREQUISITE_THRESHOLD
        mastery["binary_search_trees"] = 0.10
        decision = self.curriculum.select_next_activity(
            "s", mastery, self.curriculum.DSA_PREREQUISITES, []
        )
        # BST should not be recommended when trees mastery is too low
        self.assertNotEqual(decision["concept_id"], "binary_search_trees")

    def test_decision_has_required_fields(self):
        decision = self.curriculum.select_next_activity(
            "s", {}, self.curriculum.DSA_PREREQUISITES, []
        )
        for field in ["student_id", "concept_id", "activity", "reason", "next_action"]:
            self.assertIn(field, decision)


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

class TestMember1Member2Integration(unittest.TestCase):
    """Integration: Member 1 + Member 2."""

    @classmethod
    def setUpClass(cls):
        import assessment
        import evaluator
        import bkt_engine
        cls.assessment = assessment
        cls.evaluator = evaluator
        cls.bkt = bkt_engine
        evaluator._load_questions()
        bkt_engine.reset_bkt_state()

    def tearDown(self):
        self.bkt.reset_bkt_state()

    def test_assessment_to_bkt_pipeline(self):
        """MCQ evaluation feeds directly into BKT update."""
        result = self.assessment.evaluate_assessment_answer("assess_stacks_01", "C")
        self.assertEqual(result["outcome"], 1)

        bkt_result = self.bkt.update_bkt("int_student", "stacks", result["outcome"])
        self.assertGreater(bkt_result["mastery"], self.bkt.P_INIT)

    def test_evaluator_bkt_outcome_feeds_bkt(self):
        """Short-answer evaluator's bkt_outcome feeds into BKT."""
        answer = "A stack operates on LIFO. Push and pop both take O(1) time."
        eval_result = self.evaluator.evaluate_answer("Q_STK_01", answer)
        bkt_input = eval_result["bkt_outcome"]
        bkt_result = self.bkt.update_bkt("int_student_2", "stacks", bkt_input)
        self.assertIn("mastery", bkt_result)
        self.assertEqual(bkt_result["outcome"], bkt_input)


class TestFullLearningCycle(unittest.TestCase):
    """End-to-end integration: complete learning cycle."""

    @classmethod
    def setUpClass(cls):
        import orchestrator
        import bkt_engine
        cls.orch = orchestrator
        cls.bkt = bkt_engine

    def setUp(self):
        self.student_id = "e2e_test_student_001"
        self.bkt.reset_bkt_state()

    def test_initial_cycle(self):
        result = self.orch.run_learning_cycle(self.student_id)
        self.assertIn("student_id", result)
        self.assertIn("curriculum_decision", result)
        self.assertIn("question", result)
        self.assertIn("explanation", result)
        self.assertIn("mastery_profile", result)
        self.assertIn("learning_history", result)

    def test_assessment_cycle(self):
        result = self.orch.run_assessment_cycle(self.student_id)
        self.assertIn("questions", result)
        self.assertEqual(len(result["questions"]), 13)

    def test_submit_assessment_answer(self):
        result = self.orch.submit_assessment_answer(
            self.student_id, "assess_arrays_01", "A"
        )
        self.assertEqual(result["classification"], "correct")
        self.assertEqual(result["outcome"], 1)
        self.assertIn("arrays", result["mastery_profile"])

    def test_mastery_persists_across_cycles(self):
        """Mastery should be restored from DB on session resume."""
        # Submit correct answer
        self.orch.submit_assessment_answer(self.student_id, "assess_arrays_01", "A")

        # Reset in-memory BKT state (simulating new session)
        self.bkt.reset_bkt_state()

        # New cycle should restore mastery from DB
        result = self.orch.run_learning_cycle(self.student_id)
        self.assertIn("arrays", result["mastery_profile"])
        self.assertGreater(result["mastery_profile"]["arrays"], 0.0)

    def test_dashboard_data(self):
        self.orch.submit_assessment_answer(self.student_id, "assess_stacks_01", "C")
        dash = self.orch.get_student_dashboard(self.student_id)
        self.assertIn("mastery_profile", dash)
        self.assertIn("learning_history", dash)
        self.assertIn("attempts", dash)
        self.assertIn("curriculum_recommendation", dash)
        self.assertGreater(len(dash["attempts"]), 0)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LearnSync AI — Integrated Test Suite")
    print("=" * 70)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestMember1Assessment,
        TestMember1BKT,
        TestMember2Evaluator,
        TestMember2RAG,
        TestMember3Database,
        TestMember3Curriculum,
        TestMember1Member2Integration,
        TestFullLearningCycle,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}, ERRORS: {len(result.errors)}")
    print("=" * 70)
