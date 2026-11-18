"""
Unit and Integration Test Suite for LearnSync AI - Member 2
============================================================
Verifies all contract requirements for:
    - RAG Content Agent (rag.py)
    - Rule-based Answer Evaluation Agent (evaluator.py)
    - Data schemas in opendsa_chunks.csv and short_answer_dataset.csv
"""

import unittest
from rag import load_opendsa_content, build_vector_index, retrieve_context, generate_explanation, VALID_CONCEPTS
from evaluator import evaluate_answer, bkt_outcome_from_evaluation, _load_questions


class TestMember2RAGAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_opendsa_content()
        cls.index, cls.active_docs = build_vector_index(cls.docs)

    def test_opendsa_content_loaded(self):
        self.assertGreater(len(self.docs), 0, "OpenDSA chunks should not be empty.")
        for doc in self.docs:
            self.assertIn("chunk_id", doc)
            self.assertIn("concept_id", doc)
            self.assertIn("title", doc)
            self.assertIn("content", doc)
            self.assertIn("source_url", doc)
            self.assertIn(doc["concept_id"], VALID_CONCEPTS, f"Invalid concept_id: {doc['concept_id']}")

    def test_retrieve_context_with_concept_filter(self):
        results = retrieve_context("What is LIFO in stacks?", concept_id="stacks", top_k=2)
        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["concept_id"], "stacks")
        self.assertTrue(results[0]["source_url"].startswith("http"))
        self.assertIn("score", results[0])

    def test_retrieve_context_empty_query(self):
        results = retrieve_context("", concept_id="stacks")
        self.assertEqual(results, [])

    def test_generate_explanation_mastery_levels(self):
        context = retrieve_context("Explain stacks", concept_id="stacks", top_k=2)
        exp_low = generate_explanation("Explain stacks", context, mastery_level=0.2)
        self.assertIn("Foundational Tier", exp_low)
        self.assertIn("OpenDSA Sources", exp_low)

        exp_med = generate_explanation("Explain stacks", context, mastery_level=0.55)
        self.assertIn("Intermediate Tier", exp_med)

        exp_high = generate_explanation("Explain stacks", context, mastery_level=0.85)
        self.assertIn("Advanced Tier", exp_high)


class TestMember2AnswerEvaluationAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _load_questions()

    def test_evaluate_correct_answer(self):
        ans = "A stack operates on Last-In First-Out (LIFO) discipline. Push and pop both take O(1) time."
        res = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res["classification"], "correct")
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["bkt_outcome"], 1)
        self.assertEqual(res["identified_misconceptions"], [])
        self.assertEqual(res["concept_id"], "stacks")

    def test_evaluate_partially_correct_answer(self):
        ans = "A stack operates on LIFO where you push to the top and pop from the top."
        res = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res["classification"], "partially_correct")
        self.assertEqual(res["score"], 0.5)
        self.assertEqual(res["bkt_outcome"], 0)

    def test_evaluate_misconception_detection(self):
        ans = "A stack is First-In First-Out (FIFO) like a supermarket queue. Push is O(1) and pop is O(n)."
        res = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res["classification"], "incorrect")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["bkt_outcome"], 0)
        self.assertIn("Confuses LIFO with FIFO", res["identified_misconceptions"])

    def test_bkt_conversion_rule(self):
        # Explicit test for Part A & B requirement
        self.assertEqual(bkt_outcome_from_evaluation({"score": 1.0}), 1)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.75}), 1)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.5}), 0)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.0}), 0)

    def test_unknown_question_handling(self):
        res = evaluate_answer("UNKNOWN_Q_999", "Some answer")
        self.assertEqual(res["classification"], "incorrect")
        self.assertEqual(res["bkt_outcome"], 0)


if __name__ == "__main__":
    unittest.main()
