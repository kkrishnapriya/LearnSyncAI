"""
LearnSync AI - Member 2: Held-Out Answer Evaluation Pipeline (run_heldout_eval.py)
==================================================================================
This script executes a strict, out-of-sample empirical evaluation of the
rule-based Answer Evaluation Agent (evaluator.py) on heldout_answers.csv (26 rows).

The 26 held-out records were NOT used during rubric or rule construction and
serve as an unbiased generalization test on unobserved student phrasing.

Requirements:
    - Reads heldout_answers.csv (columns: question_id, student_answer, your_label, expected_misconception)
    - Runs evaluate_answer() on each row
    - Computes and prints:
        * Overall classification accuracy
        * 3-Class Confusion Matrix (correct, partially_correct, incorrect)
        * Per-class precision, recall, F1, and support
        * Macro and weighted averages
        * Detailed list of misclassified rows with question IDs and diagnostics
    - Does NOT modify any rules in evaluator.py
"""

import os
import csv
import time
from typing import Dict, List, Any

from evaluator import evaluate_answer, bkt_outcome_from_evaluation


def evaluate_heldout_dataset(csv_path: str = "heldout_answers.csv") -> Dict[str, Any]:
    """
    Evaluates the held-out benchmark dataset without modifying any rules.
    """
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Held-out dataset not found at: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    classes = ["correct", "partially_correct", "incorrect"]
    cm = {act: {pred: 0 for pred in classes} for act in classes}
    misclassified = []
    latencies = []

    print("=" * 115)
    print(f"LearnSync AI - Held-Out Benchmark Evaluation ({os.path.basename(csv_path)}, {len(records)} Rows)")
    print("=" * 115)
    print("Note: The 26 held-out examples were not used during rule development.")
    print("This evaluation measures out-of-sample generalization of the rule-based classifier.\n")

    t_start = time.perf_counter()

    for idx, r in enumerate(records, 1):
        qid = r["question_id"].strip()
        ans = r["student_answer"].strip()
        act = r["your_label"].strip()
        exp_misc = r.get("expected_misconception", "").strip()

        t0 = time.perf_counter()
        res = evaluate_answer(qid, ans)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        pred = res["classification"]
        cm[act][pred] += 1

        if pred != act:
            misclassified.append({
                "row_index": idx,
                "question_id": qid,
                "student_answer": ans,
                "actual_label": act,
                "predicted_label": pred,
                "score": res["score"],
                "bkt_outcome": res["bkt_outcome"],
                "expected_misconception": exp_misc,
                "identified_misconceptions": res["identified_misconceptions"],
                "feedback": res["feedback"],
            })

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    total_samples = len(records)
    total_correct = sum(cm[c][c] for c in classes)
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    # Per-class metrics
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

    # Print results
    print(f"Total Held-Out Samples: {total_samples}")
    print(f"Overall Classification Accuracy: {total_correct}/{total_samples} ({accuracy:.2f}%)\n")

    print("3-Class Confusion Matrix:")
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

    print(f"Total Misclassified Rows: {len(misclassified)} / {total_samples}")
    print("-" * 115)
    for m in misclassified:
        print(f"[Row {m['row_index']:02d}] {m['question_id']} | Actual: '{m['actual_label']}' | Predicted: '{m['predicted_label']}' (Score: {m['score']})")
        print(f"  Student Answer: \"{m['student_answer']}\"")
        if m["expected_misconception"]:
            print(f"  Expected Misconception: {m['expected_misconception']}")
        print(f"  Identified Misconceptions: {m['identified_misconceptions']}")
        print(f"  Feedback: {m['feedback']}")
        print()

    print("Execution Latency Summary:")
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"  Total Elapsed Time: {total_time_ms:.2f} ms ({total_time_ms/1000.0:.2f} s)")
    print(f"  Mean Latency per Answer: {mean_lat:.2f} ms")
    print("=" * 115)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "metrics_per_class": metrics_per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "misclassified": misclassified,
    }


if __name__ == "__main__":
    evaluate_heldout_dataset()
