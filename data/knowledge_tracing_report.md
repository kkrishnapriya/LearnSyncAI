# LearnSync AI — Knowledge Tracing Report

## Member 1: Assessment and Knowledge Tracing

## 1\. Objective

The objective of this module is to analyse student programming submissions, map programming problems to DSA concepts, construct chronological student learning sequences, assess student knowledge using objective questions, and estimate concept mastery using Bayesian Knowledge Tracing (BKT).

## 2\. Dataset Overview

The project uses the Zenodo programming-submission dataset.

Files inspected:

- java\_submissions.csv  
- distribution\_per\_question.csv  
- tst\_questions.csv

Key dataset observations:

- Raw Java submission records: 16,074  
- Cleaned interactions: 15,474  
- Unique students: 270  
- Unique programming questions: 99  
- Empty-result records excluded during preprocessing: 600

Submission outcomes:

- Successful submissions are represented as 1\.  
- Failed submissions are represented as 0\.  
- Empty-result records are excluded from binary BKT observations.

Missing timestamps were handled using the available alternative timestamp fields during preprocessing.

## 3\. Concept Mapping

Programming questions were mapped to the project's agreed DSA concept IDs.

The mapping includes:

- concept\_id  
- mapping\_reason  
- confidence

Seven questions remain unmapped because their primary DSA concept could not be assigned confidently.

Important mapping corrections:

- hash\_table\_chaining \-\> hashing  
- queue\_jumper \-\> arrays

Unmapped questions are retained in the preprocessed dataset but excluded from concept-specific BKT sequences.

## 4\. Preprocessing and Student Sequences

The final preprocessed dataset contains:

- 15,474 interactions  
- 270 unique students  
- 99 unique questions

Interactions are grouped by student and concept, then ordered chronologically by timestamp.

Final sequence statistics:

- Student-concept sequences: 1,491  
- Unique students with mapped observations: 270  
- Mapped observations used by BKT: 14,390

## 5\. Assessment Module

The assessment module provides 13 objective multiple-choice questions, covering the agreed DSA concepts.

Implemented functions:

- get\_assessment\_questions(concept\_id=None)  
- evaluate\_assessment\_answer(question\_id, answer)

The evaluator returns:

- question\_id  
- concept\_id  
- binary outcome (0 or 1\)  
- correctness classification  
- feedback and explanation

The correct answer is not exposed by the question-retrieval function.

The current question bank is a prototype with one question per concept, not a comprehensive assessment bank.

## 6\. Bayesian Knowledge Tracing

BKT estimates the probability that a student has mastered a concept.

Initial baseline parameters:

- Initial mastery (P\_INIT): 0.20  
- Learning probability (P\_LEARN): 0.10  
- Guess probability (P\_GUESS): 0.20  
- Slip probability (P\_SLIP): 0.10

These are baseline assumptions and have not been fitted or optimized against a held-out dataset.

Implemented functions:

- update\_bkt(student\_id, concept\_id, outcome)  
- get\_mastery\_profile(student\_id)  
- predict\_correct\_probability(...)  
- evaluate\_bkt(sequences)  
- reset\_bkt\_state()

The mastery profile follows the shared project contract:

{ "student\_id": "student\_001", "mastery": { "arrays": 0.72, "linked\_lists": 0.45 } }

Mastery estimates are bounded between 0 and 1\.

## 7\. BKT Evaluation

Evaluation was performed by processing each student's mapped interactions in chronological order. For each observation, the model predicts the probability of a correct response before updating its mastery estimate using the observed outcome.

Final evaluation results:

| Metric | Result |
| :---- | ----: |
| Observations | 14,390 |
| Accuracy | 0.6833 |
| Brier score | 0.2228 |
| Log loss | 0.6498 |

These are sequential one-step-ahead results on the available interaction sequences. They should not be interpreted as held-out test performance or evidence of generalization to unseen students.

## 8\. Example Mastery Profile

Example student: user\_0001

| Concept | Estimated mastery |
| :---- | ----: |
| Arrays | 0.1143 |
| Recursion | 0.8118 |
| Sorting | 0.9706 |
| Stacks | 0.9559 |
| Queues | 0.5765 |
| Hashing | 0.1143 |
| Binary Search Trees | 0.9997 |
| BFS | 0.5765 |
| Heaps | 0.7980 |

The profile illustrates how mastery estimates are maintained separately for each student and DSA concept.

## 9\. Limitations

1. BKT parameters are baseline assumptions, not learned parameters.  
2. Evaluation uses sequential historical observations and does not use a held-out test set.  
3. Programming submission outcomes are proxies for correctness and do not directly prove conceptual understanding.  
4. Repeated submissions may be correlated.  
5. Seven questions remain unmapped.  
6. The assessment bank currently contains only one question per concept.  
7. The current BKT implementation maintains state in memory; persistent storage is handled by the integration layer.

## 10\. Conclusion

The Member 1 module provides a working data preprocessing pipeline, documented problem-to-concept mappings, chronological student-concept sequences, an objective assessment bank, and a functional BKT engine.

The module produces per-student, per-concept mastery estimates in the shared project format and has been evaluated on 14,390 mapped observations.