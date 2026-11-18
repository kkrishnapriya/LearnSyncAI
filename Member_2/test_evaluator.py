"""
LearnSync AI - Member 2: Rule-Based Answer Evaluation Test Suite & Benchmark (test_evaluator.py)
================================================================================================
This module delivers the comprehensive test suite and quantitative benchmark for the
rule-based Answer Evaluation Agent (evaluator.py) in accordance with the LearnSync AI
Implementation Contract.

Classification Methodology:
    The evaluator is a deterministic, rule-based classifier using regex rubrics,
    invariant checks, and clause-bounded negation handling.
    Dense SentenceTransformer similarity is logged as supporting evidence only
    and does NOT affect rule-based classification or scoring.

Key Sections:
    1. 15 Comprehensive Test Categories (Unit and Integration Tests)
       - Category 6 features at least 8 adversarial negation/contrast tests across 4 concepts
         in both directions (negated wrong term must not trigger; earlier negation must not
         suppress a subsequent genuine error).
    2. 117-Row Quantitative Benchmark against short_answer_dataset.csv:
       - 3-Class Confusion Matrix (correct, partially_correct, incorrect)
       - Precision, Recall, F1 (per-class, macro-averaged, and weighted)
       - Sub-cohort metrics: reviewed_reference (39) vs synthetic_demo (78)
       - Exact-label misconception evaluation:
           * Omission detection (partially_correct rows) evaluated separately with TP, FN, FP,
             precision, recall, F1, and question IDs
           * Critical misconception detection (incorrect rows) evaluated separately with TP, FN, FP,
             precision, recall, F1, and question IDs
           * Transparent reporting of label mismatches between dataset strings and rule names
       - Binary BKT observation mapping verification
       - Execution latencies (mean, median, min, max, total elapsed)
"""

import os
import csv
import time
import unittest
from typing import Dict, List, Any

import evaluator
from evaluator import (
    evaluate_answer,
    bkt_outcome_from_evaluation,
    _load_questions,
    VALID_CONCEPTS,
    _QUESTION_BANK,
)


class TestEvaluatorComprehensive(unittest.TestCase):
    """
    15-Category Test Suite covering all contract specifications, edge cases,
    robustness constraints, and pedagogical invariants for the rule-based evaluator.
    """

    @classmethod
    def setUpClass(cls):
        _load_questions()

    # -------------------------------------------------------------------------
    # Category 1: Correct answers evaluation
    # -------------------------------------------------------------------------
    def test_category_01_correct_answers_evaluation(self):
        """Verifies that an exemplary answer receives score 1.0, bkt_outcome 1, and no misconceptions."""
        ans = (
            "A stack is a Last-In First-Out (LIFO) linear data structure where elements are "
            "inserted and removed exclusively from the top. Push and pop operations operate in O(1) time."
        )
        res = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res["classification"], "correct")
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["bkt_outcome"], 1)
        self.assertEqual(res["identified_misconceptions"], [])
        self.assertEqual(res["concept_id"], "stacks")
        self.assertTrue(any("All required conceptual invariants verified" in ev for ev in res["evidence"]))

    # -------------------------------------------------------------------------
    # Category 2: Partially correct answers evaluation
    # -------------------------------------------------------------------------
    def test_category_02_partially_correct_answers_evaluation(self):
        """Verifies that an answer missing a required invariant receives score 0.5, bkt_outcome 0, and omission note."""
        ans = "A stack operates on LIFO where you push to the top and pop from the top."
        res = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res["classification"], "partially_correct")
        self.assertEqual(res["score"], 0.5)
        self.assertEqual(res["bkt_outcome"], 0)
        self.assertGreater(len(res["identified_misconceptions"]), 0)
        self.assertIn("Omission of O(1) time complexity specification", res["identified_misconceptions"])
        self.assertTrue(any("Omission:" in ev for ev in res["evidence"]))

    # -------------------------------------------------------------------------
    # Category 3: Incorrect answers evaluation
    # -------------------------------------------------------------------------
    def test_category_03_incorrect_answers_evaluation(self):
        """Verifies that an answer exhibiting a critical misconception receives score 0.0, bkt_outcome 0, and diagnosis."""
        ans = (
            "A stack is First-In First-Out (FIFO) like a supermarket checkout queue. "
            "Push takes O(1) time and pop takes O(n) time."
        )
        res = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res["classification"], "incorrect")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["bkt_outcome"], 0)
        self.assertIn("Confuses LIFO with FIFO", res["identified_misconceptions"])
        self.assertTrue(any("Detected conceptual error:" in ev for ev in res["evidence"]))

    # -------------------------------------------------------------------------
    # Category 4: All 13 supported DSA concepts
    # -------------------------------------------------------------------------
    def test_category_04_all_13_supported_dsa_concepts(self):
        """Verifies that evaluation succeeds and correctly identifies all 13 supported DSA concepts."""
        sample_questions = {
            "arrays": "Q_ARR_01",
            "linked_lists": "Q_LL_01",
            "stacks": "Q_STK_01",
            "queues": "Q_QUE_01",
            "recursion": "Q_REC_01",
            "trees": "Q_TRE_01",
            "binary_search_trees": "Q_BST_01",
            "heaps": "Q_HEA_01",
            "sorting": "Q_SRT_01",
            "hashing": "Q_HAS_01",
            "graphs": "Q_GRP_01",
            "bfs": "Q_BFS_01",
            "dfs": "Q_DFS_01",
        }
        for concept, qid in sample_questions.items():
            self.assertIn(concept, VALID_CONCEPTS)
            res = evaluate_answer(qid, "This is a basic answer attempting to explain the concept.")
            self.assertEqual(
                res["concept_id"],
                concept,
                f"Mismatch for question {qid}: expected {concept}, got {res['concept_id']}",
            )
            self.assertIn(res["classification"], {"correct", "partially_correct", "incorrect"})
            self.assertIn(res["bkt_outcome"], {0, 1})

    # -------------------------------------------------------------------------
    # Category 5: Misconception categories in the dataset
    # -------------------------------------------------------------------------
    def test_category_05_misconception_categories_in_dataset(self):
        """Verifies explicit detection of known pedagogical misconceptions across different concepts."""
        cases = [
            ("Q_QUE_03", "A queue operates under Last-In First-Out (LIFO) order where the last item added is popped first.", "Confuses FIFO with LIFO"),
            ("Q_BST_03", "A BST is any binary tree where search is always strictly O(log n) regardless of insertion order.", "Assumes BST search is always guaranteed O(log n) regardless of tree shape"),
            ("Q_HEA_03", "In a max-heap, all elements are fully sorted from left to right like an array, taking O(1) insertion.", "Confuses heap order with total sorted order"),
            ("Q_HAS_03", "Hash tables guarantee O(1) search time in all cases because hash codes are always unique.", "Believes hash tables guarantee worst-case O(1) and collisions never occur"),
            ("Q_BFS_03", "BFS uses a LIFO stack and finds shortest paths in graphs with negative weighted edges.", "Confuses BFS queue with DFS stack and Dijkstra/Bellman-Ford weights"),
        ]
        for qid, ans, expected_misc in cases:
            res = evaluate_answer(qid, ans)
            self.assertEqual(res["classification"], "incorrect")
            self.assertEqual(res["score"], 0.0)
            self.assertIn(
                expected_misc,
                res["identified_misconceptions"],
                f"Failed to identify '{expected_misc}' in question {qid}",
            )

    # -------------------------------------------------------------------------
    # Category 6: Negation and contrast cases (8 adversarial tests across 4 concepts)
    # -------------------------------------------------------------------------
    def test_category_06_negation_and_contrast_cases(self):
        """
        Adversarial negation and contrast tests covering both directions across at least 4 concepts:
          Direction A: Negated wrong term must NOT trigger false-positive misconception.
          Direction B: Earlier negation in a preceding sentence/clause must NOT suppress a genuine later error.
        """
        # Direction A: Negated wrong terms must not trigger false positives
        direction_a_cases = [
            ("Q_STK_01", "A stack follows LIFO discipline with O(1) push and pop. It does not use FIFO ordering."),
            ("Q_QUE_01", "A queue operates on FIFO where elements are added at rear and removed at front in O(1) time. It is not LIFO."),
            ("Q_BFS_01", "BFS explores level by level using a FIFO queue to find unweighted shortest paths. It does not use a stack or priority queue."),
            ("Q_BST_01", "In a binary search tree, the left subtree is less than root and right subtree is greater. Balanced trees achieve O(log n) search, but a skewed tree degrades to O(n) worst case. It is not always balanced."),
        ]
        for qid, ans in direction_a_cases:
            res = evaluate_answer(qid, ans)
            self.assertEqual(
                res["classification"],
                "correct",
                f"Direction A failed: Negated wrong term falsely triggered misconception in {qid}: {res['identified_misconceptions']}",
            )
            self.assertEqual(res["score"], 1.0)
            self.assertEqual(res["bkt_outcome"], 1)
            self.assertEqual(res["identified_misconceptions"], [])

        # Direction B: Earlier negation must not suppress a genuine later error
        direction_b_cases = [
            (
                "Q_STK_01",
                "A stack follows LIFO. It is not a queue. However, the first inserted element is removed first. Push and pop are O(1).",
                "Confuses LIFO with FIFO",
            ),
            (
                "Q_QUE_01",
                "A queue is not a stack. However, elements are popped in last-in first-out (LIFO) order. Enqueue and dequeue are O(1).",
                "Confuses FIFO with LIFO",
            ),
            (
                "Q_DFS_01",
                "DFS does not use recursion without reason. But it uses a FIFO queue to visit all neighbors at distance 1 level by level.",
                "Confuses DFS depth exploration with BFS level exploration",
            ),
            (
                "Q_BST_01",
                "A BST does not store elements randomly. Yet, search is always strictly O(log n) regardless of insertion order.",
                "Assumes BST search is always guaranteed O(log n) regardless of tree shape",
            ),
        ]
        for qid, ans, exp_misc in direction_b_cases:
            res = evaluate_answer(qid, ans)
            self.assertEqual(
                res["classification"],
                "incorrect",
                f"Direction B failed: Earlier negation improperly suppressed subsequent error in {qid}: {res['feedback']}",
            )
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["bkt_outcome"], 0)
            self.assertIn(
                exp_misc,
                res["identified_misconceptions"],
                f"Direction B failed to identify '{exp_misc}' in {qid}",
            )

    # -------------------------------------------------------------------------
    # Category 7: keyword-preserving variations
    # -------------------------------------------------------------------------
    def test_category_07_valid_paraphrased_answers(self):
        """Verifies that correct answers using alternative phrasing and synonyms pass invariant validation."""
        paraphrases = [
            ("Q_STK_01", "Stacks utilize the last-in-first-out paradigm, executing insertion and removal at constant time O(1)."),
            ("Q_ARR_01", "Arrays store items in contiguous memory allowing O(1) direct offset indexing, but inserting at index 0 requires shifting all elements in O(n) time."),
            ("Q_LL_01", "A singly linked list uses nodes with data and next pointers, allowing O(1) head insertion by redirecting references, but finding an element requires O(n) linear traversal."),
        ]
        for qid, ans in paraphrases:
            res = evaluate_answer(qid, ans)
            self.assertEqual(
                res["classification"],
                "correct",
                f"Valid paraphrase rejected for {qid}: {res['feedback']}",
            )
            self.assertEqual(res["score"], 1.0)
            self.assertEqual(res["bkt_outcome"], 1)

    # -------------------------------------------------------------------------
    # Category 8: Correct keywords in incorrect explanations (Keyword Stuffing)
    # -------------------------------------------------------------------------
    def test_category_08_correct_keywords_in_incorrect_explanations(self):
        """Verifies defense against keyword stuffing where buzzwords are present but the explanation is flawed."""
        # Answer contains "LIFO", "push", "pop", "O(1)", but asserts FIFO mechanism
        stuffed_ans = (
            "A stack uses LIFO and push and pop in O(1) time, but it functions as First-In First-Out "
            "where the first element inserted is always the first one popped."
        )
        res = evaluate_answer("Q_STK_01", stuffed_ans)
        self.assertEqual(res["classification"], "incorrect")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["bkt_outcome"], 0)
        self.assertIn("Confuses LIFO with FIFO", res["identified_misconceptions"])

    # -------------------------------------------------------------------------
    # Category 9: Blank and whitespace answers
    # -------------------------------------------------------------------------
    def test_category_09_blank_and_whitespace_answers(self):
        """Verifies that empty or whitespace-only answers return score 0.0 without errors."""
        blank_inputs = ["", "   ", "\t\t\n\n", "   \r\n   "]
        for blank in blank_inputs:
            res = evaluate_answer("Q_STK_01", blank)
            self.assertEqual(res["classification"], "incorrect")
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["bkt_outcome"], 0)
            self.assertIn("Blank or empty submission", res["identified_misconceptions"])
            self.assertIn("No answer was provided", res["feedback"])

    # -------------------------------------------------------------------------
    # Category 10: Unknown question IDs
    # -------------------------------------------------------------------------
    def test_category_10_unknown_question_ids(self):
        """Verifies graceful handling when an unregistered question_id is provided."""
        res = evaluate_answer("NON_EXISTENT_QUESTION_ID_9999", "Some arbitrary answer text.")
        self.assertEqual(res["classification"], "incorrect")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["bkt_outcome"], 0)
        self.assertIn("not recognized", res["feedback"])
        self.assertIn(res["concept_id"], {"unmapped", "unknown"})

    # -------------------------------------------------------------------------
    # Category 11: Invalid input types
    # -------------------------------------------------------------------------
    def test_category_11_invalid_input_types(self):
        """Verifies robustness against non-string input types for student_answer and question_id."""
        invalid_answers = [None, 12345, 99.9, ["list", "of", "words"], {"key": "val"}]
        for invalid_ans in invalid_answers:
            res = evaluate_answer("Q_STK_01", invalid_ans)  # type: ignore
            self.assertEqual(res["classification"], "incorrect")
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["bkt_outcome"], 0)

        invalid_qids = [None, 123, ["qid"]]
        for invalid_qid in invalid_qids:
            res = evaluate_answer(invalid_qid, "Some answer")  # type: ignore
            self.assertEqual(res["classification"], "incorrect")
            self.assertEqual(res["score"], 0.0)
            self.assertEqual(res["bkt_outcome"], 0)

    # -------------------------------------------------------------------------
    # Category 12: BKT threshold boundaries
    # -------------------------------------------------------------------------
    def test_category_12_bkt_threshold_boundaries(self):
        """Verifies strict adherence to the BKT binary conversion rule at exact boundary values."""
        # Default threshold = 0.70
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.69}), 0)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.699999}), 0)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.70}), 1)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.700001}), 1)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.71}), 1)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.0}), 0)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 1.0}), 1)

        # Custom threshold parameter
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.75}, threshold=0.80), 0)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.80}, threshold=0.80), 1)
        self.assertEqual(bkt_outcome_from_evaluation({"score": 0.85}, threshold=0.80), 1)

        # Non-dict or invalid evaluations
        self.assertEqual(bkt_outcome_from_evaluation(None), 0)
        self.assertEqual(bkt_outcome_from_evaluation({}), 0)
        self.assertEqual(bkt_outcome_from_evaluation({"score": "invalid"}), 0)

    # -------------------------------------------------------------------------
    # Category 13: Missing or malformed dataset handling
    # -------------------------------------------------------------------------
    def test_category_13_missing_or_malformed_dataset_handling(self):
        """Verifies that _load_questions handles a non-existent or inaccessible filepath safely."""
        loaded = _load_questions("this_file_does_not_exist_at_all.csv")
        self.assertIsInstance(loaded, dict)

    # -------------------------------------------------------------------------
    # Category 14: Output schema and score validity
    # -------------------------------------------------------------------------
    def test_category_14_output_schema_and_score_validity(self):
        """Verifies that every evaluation returns the 7 required contract keys with valid types and bounds."""
        res = evaluate_answer("Q_STK_01", "A stack operates on LIFO.")
        required_keys = {
            "concept_id",
            "classification",
            "score",
            "bkt_outcome",
            "identified_misconceptions",
            "feedback",
            "evidence",
        }
        self.assertEqual(set(res.keys()), required_keys)
        self.assertIsInstance(res["concept_id"], str)
        self.assertIn(res["classification"], {"correct", "partially_correct", "incorrect"})
        self.assertIsInstance(res["score"], float)
        self.assertTrue(0.0 <= res["score"] <= 1.0)
        self.assertIsInstance(res["bkt_outcome"], int)
        self.assertIn(res["bkt_outcome"], {0, 1})
        self.assertIsInstance(res["identified_misconceptions"], list)
        self.assertIsInstance(res["feedback"], str)
        self.assertIsInstance(res["evidence"], list)

    # -------------------------------------------------------------------------
    # Category 15: Repeated evaluation and caching
    # -------------------------------------------------------------------------
    def test_category_15_repeated_evaluation_and_caching(self):
        """Verifies deterministic idempotency across multiple sequential evaluations."""
        ans = "A stack operates on LIFO discipline with O(1) push and pop."
        res1 = evaluate_answer("Q_STK_01", ans)
        res2 = evaluate_answer("Q_STK_01", ans)
        self.assertEqual(res1["classification"], res2["classification"])
        self.assertEqual(res1["score"], res2["score"])
        self.assertEqual(res1["bkt_outcome"], res2["bkt_outcome"])
        self.assertEqual(res1["identified_misconceptions"], res2["identified_misconceptions"])
        self.assertEqual(res1["feedback"], res2["feedback"])


# =============================================================================
# Automated 117-Row Quantitative Benchmark Pipeline
# =============================================================================
def run_quantitative_benchmark(dataset_path: str = "short_answer_dataset.csv") -> Dict[str, Any]:
    """
    Executes the full 117-row evaluation pipeline on short_answer_dataset.csv.

    Evaluation Standards:
        - 3-Class Confusion Matrix (correct, partially_correct, incorrect)
        - Exact-label misconception and omission comparison:
            * Omissions (partially_correct rows) evaluated separately
            * Critical misconceptions (incorrect rows) evaluated separately
            * False positives and false negatives listed by question ID
            * Mismatches between dataset labels and rule names reported transparently
    """
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Benchmark dataset not found at: {dataset_path}")

    with open(dataset_path, mode="r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    classes = ["correct", "partially_correct", "incorrect"]
    cm = {act: {pred: 0 for pred in classes} for act in classes}
    sub_cm = {
        "reviewed_reference": {act: {pred: 0 for pred in classes} for act in classes},
        "synthetic_demo": {act: {pred: 0 for pred in classes} for act in classes},
    }

    bkt_matches = 0
    latencies = []

    # Exact-label tracking structures
    omiss_tp, omiss_fn, omiss_fp = 0, 0, 0
    omiss_fn_details, omiss_fp_details = [], []

    crit_tp, crit_fn, crit_fp = 0, 0, 0
    crit_fn_details, crit_fp_details = [], []

    label_mismatches = []

    print("=" * 115)
    print("LearnSync AI - Phase 4 Quantitative Benchmark (short_answer_dataset.csv, 117 Rows)")
    print("=" * 115)
    print("Evaluator Type: Rule-Based Classifier (regex rubrics, invariant checks, negation handling)")
    print("Note: Dense SentenceTransformer similarity is logged as supporting evidence only.\n")

    start_total = time.perf_counter()

    for r in records:
        qid = r["question_id"].strip()
        actual_cls = r["ground_truth_classification"].strip()
        source = r.get("data_source", "synthetic_demo").strip()
        expected_misc = r.get("expected_misconceptions", "").strip()
        has_expected = expected_misc not in ("", "None", "none", None)

        t0 = time.perf_counter()
        res = evaluate_answer(qid, r["student_answer"])
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        pred_cls = res["classification"]
        score = res["score"]
        bkt = res["bkt_outcome"]
        pred_miscs = res["identified_misconceptions"]

        # 3-Class Confusion matrix
        cm[actual_cls][pred_cls] += 1
        if source in sub_cm:
            sub_cm[source][actual_cls][pred_cls] += 1

        # BKT binary mapping rule check
        expected_bkt = 1 if score >= 0.70 else 0
        if bkt == expected_bkt:
            bkt_matches += 1

        # Exact-label omission detection (partially_correct rows)
        if actual_cls == "partially_correct":
            if has_expected:
                if expected_misc in pred_miscs:
                    omiss_tp += 1
                else:
                    omiss_fn += 1
                    omiss_fn_details.append((qid, expected_misc))
                    label_mismatches.append({
                        "question_id": qid,
                        "category": "omission",
                        "expected_label": expected_misc,
                        "rule_names": pred_miscs,
                    })
            for p in pred_miscs:
                if p != expected_misc:
                    omiss_fp += 1
                    omiss_fp_details.append((qid, p))

        # Exact-label critical misconception detection (incorrect rows)
        elif actual_cls == "incorrect":
            if has_expected:
                if expected_misc in pred_miscs:
                    crit_tp += 1
                else:
                    crit_fn += 1
                    crit_fn_details.append((qid, expected_misc))
                    label_mismatches.append({
                        "question_id": qid,
                        "category": "critical_misconception",
                        "expected_label": expected_misc,
                        "rule_names": pred_miscs,
                    })
            for p in pred_miscs:
                if p != expected_misc:
                    crit_fp += 1
                    crit_fp_details.append((qid, p))

    total_time = (time.perf_counter() - start_total) * 1000.0

    # -------------------------------------------------------------------------
    # Metric Calculations
    # -------------------------------------------------------------------------
    total_samples = len(records)
    total_correct_predictions = sum(cm[c][c] for c in classes)
    overall_accuracy = (total_correct_predictions / total_samples) * 100.0

    metrics_per_class = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = sum(cm[c].values())
        metrics_per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }

    macro_prec = sum(m["precision"] for m in metrics_per_class.values()) / len(classes)
    macro_rec = sum(m["recall"] for m in metrics_per_class.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / len(classes)

    weighted_prec = sum(m["precision"] * m["support"] for m in metrics_per_class.values()) / total_samples
    weighted_rec = sum(m["recall"] * m["support"] for m in metrics_per_class.values()) / total_samples
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics_per_class.values()) / total_samples

    # Sub-cohort accuracies
    sub_metrics = {}
    for src in ["reviewed_reference", "synthetic_demo"]:
        src_total = sum(sum(sub_cm[src][c].values()) for c in classes)
        src_correct = sum(sub_cm[src][c][c] for c in classes)
        sub_metrics[src] = {
            "total": src_total,
            "correct": src_correct,
            "accuracy": (src_correct / src_total * 100.0) if src_total > 0 else 0.0,
        }

    # Exact-label omission metrics
    prec_omiss = (omiss_tp / (omiss_tp + omiss_fp)) if (omiss_tp + omiss_fp) > 0 else 0.0
    rec_omiss = (omiss_tp / (omiss_tp + omiss_fn)) if (omiss_tp + omiss_fn) > 0 else 0.0
    f1_omiss = (2 * prec_omiss * rec_omiss / (prec_omiss + rec_omiss)) if (prec_omiss + rec_omiss) > 0 else 0.0

    # Exact-label critical misconception metrics
    prec_crit = (crit_tp / (crit_tp + crit_fp)) if (crit_tp + crit_fp) > 0 else 0.0
    rec_crit = (crit_tp / (crit_tp + crit_fn)) if (crit_tp + crit_fn) > 0 else 0.0
    f1_crit = (2 * prec_crit * rec_crit / (prec_crit + rec_crit)) if (prec_crit + rec_crit) > 0 else 0.0

    # -------------------------------------------------------------------------
    # Display Benchmark Output
    # -------------------------------------------------------------------------
    print(f"Total Rows Evaluated: {total_samples}")
    print(f"Overall Classification Accuracy: {total_correct_predictions}/{total_samples} ({overall_accuracy:.2f}%)\n")

    print("Confusion Matrix:")
    HDR = "Actual \\ Predicted"
    print(f"{HDR:<22} | {'correct':<10} | {'partially_correct':<18} | {'incorrect':<10}")
    print("-" * 70)
    for act in classes:
        print(f"{act:<22} | {cm[act]['correct']:<10} | {cm[act]['partially_correct']:<18} | {cm[act]['incorrect']:<10}")
    print()

    print("Per-Class Classification Metrics:")
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)
    for c in classes:
        m = metrics_per_class[c]
        print(f"{c:<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['support']:<8}")
    print("-" * 70)
    print(f"{'Macro Average':<20} | {macro_prec:<10.4f} | {macro_rec:<10.4f} | {macro_f1:<10.4f} | {total_samples:<8}")
    print(f"{'Weighted Average':<20} | {weighted_prec:<10.4f} | {weighted_rec:<10.4f} | {weighted_f1:<10.4f} | {total_samples:<8}")
    print()

    print("Sub-Cohort Performance Analysis:")
    for src, sm in sub_metrics.items():
        print(f" - {src:<20}: {sm['correct']}/{sm['total']} ({sm['accuracy']:.2f}% accuracy)")
    print()

    print("=" * 115)
    print("Exact-Label Diagnostic Metrics (Evaluated Separately)")
    print("=" * 115)
    print("A. Omission Detection (partially_correct rows, 39 support):")
    print(f" - Exact Matches (TP):      {omiss_tp}")
    print(f" - Missing Exact Match (FN):{omiss_fn}")
    print(f" - Mismatched Labels (FP):  {omiss_fp}")
    print(f" - Exact-Label Precision:   {prec_omiss:.4f}")
    print(f" - Exact-Label Recall:      {rec_omiss:.4f}")
    print(f" - Exact-Label F1-Score:    {f1_omiss:.4f}")
    if omiss_fn_details:
        print(" - Omission False Negatives by Question ID:")
        for q, exp in omiss_fn_details:
            print(f"     * {q}: expected exact label '{exp}'")
    if omiss_fp_details:
        print(" - Omission False Positives (Differing Rule Names) by Question ID:")
        for q, pred in omiss_fp_details:
            print(f"     * {q}: rule emitted '{pred}'")
    print()

    print("B. Critical Misconception Detection (incorrect rows, 39 support):")
    print(f" - Exact Matches (TP):      {crit_tp}")
    print(f" - Missing Exact Match (FN):{crit_fn}")
    print(f" - Mismatched Labels (FP):  {crit_fp}")
    print(f" - Exact-Label Precision:   {prec_crit:.4f}")
    print(f" - Exact-Label Recall:      {rec_crit:.4f}")
    print(f" - Exact-Label F1-Score:    {f1_crit:.4f}")
    if crit_fn_details:
        print(" - Critical Misconception False Negatives by Question ID:")
        for q, exp in crit_fn_details:
            print(f"     * {q}: expected exact label '{exp}'")
    if crit_fp_details:
        print(" - Critical Misconception False Positives (Differing Rule Names) by Question ID:")
        for q, pred in crit_fp_details:
            print(f"     * {q}: rule emitted '{pred}'")
    print()

    print(f"C. Label String Mismatches between Dataset and Rule Names ({len(label_mismatches)} total):")
    print("    (Note: Rules are preserved as authored; mismatches are transparently reported below)")
    for mm in label_mismatches:
        print(f"    * {mm['question_id']} ({mm['category']}):")
        print(f"        Dataset Label : '{mm['expected_label']}'")
        print(f"        Rule Output   : {mm['rule_names']}")
    print()

    print("BKT Alignment Verification:")
    print(f" - Rule Compliance: {bkt_matches}/{total_samples} ({(bkt_matches / total_samples) * 100.0:.2f}%)")
    print(" - Rule: score >= 0.70 => bkt_outcome = 1; score < 0.70 => bkt_outcome = 0")
    print(" - Status: Provisional rule pending approval from Member 1 and Member 3")
    print()

    print("Execution Performance:")
    mean_lat = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat) // 2]
    min_lat = min(latencies)
    max_lat = max(latencies)
    print(f" - Total Benchmark Elapsed Time: {total_time:.2f} ms ({total_time / 1000.0:.2f} s)")
    print(f" - Mean Latency per Answer:      {mean_lat:.2f} ms")
    print(f" - Median Latency per Answer:    {median_lat:.2f} ms")
    print(f" - Min Latency per Answer:       {min_lat:.2f} ms")
    print(f" - Max Latency per Answer:       {max_lat:.2f} ms")
    print("=" * 115)
    print()

    return {
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": cm,
        "metrics_per_class": metrics_per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "sub_metrics": sub_metrics,
        "omiss_exact": {"tp": omiss_tp, "fn": omiss_fn, "fp": omiss_fp, "precision": prec_omiss, "recall": rec_omiss, "f1": f1_omiss},
        "crit_exact": {"tp": crit_tp, "fn": crit_fn, "fp": crit_fp, "precision": prec_crit, "recall": rec_crit, "f1": f1_crit},
        "label_mismatches": label_mismatches,
        "bkt_matches": bkt_matches,
        "mean_latency_ms": mean_lat,
    }


if __name__ == "__main__":
    # Execute quantitative benchmark first
    run_quantitative_benchmark()

    # Execute all 15 test categories via unittest
    print("=" * 115)
    print("Executing 15-Category Test Suite (test_evaluator.py)")
    print("=" * 115)
    unittest.main()
