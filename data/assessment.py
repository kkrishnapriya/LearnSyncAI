
"""
LearnSync AI - Assessment Agent (Member 1)

Provides DSA multiple-choice assessment questions and
evaluates answers using deterministic answer matching.
"""

QUESTION_BANK = [
    {
        "question_id": "assess_arrays_01",
        "concept_id": "arrays",
        "question": "What is the time complexity of accessing an element using its index in an array?",
        "options": {
            "A": "O(1)",
            "B": "O(n)",
            "C": "O(log n)",
            "D": "O(n^2)"
        },
        "correct_answer": "A",
        "explanation": "Array elements can be accessed directly using their index."
    },
    {
        "question_id": "assess_linked_lists_01",
        "concept_id": "linked_lists",
        "question": "In a singly linked list, what does each node normally contain?",
        "options": {
            "A": "Only an array index",
            "B": "Data and a reference to the next node",
            "C": "Two child nodes",
            "D": "A priority value only"
        },
        "correct_answer": "B",
        "explanation": "A singly linked-list node stores data and a link to the next node."
    },
    {
        "question_id": "assess_stacks_01",
        "concept_id": "stacks",
        "question": "Which principle does a stack follow?",
        "options": {
            "A": "FIFO",
            "B": "Random access",
            "C": "LIFO",
            "D": "Priority-based access"
        },
        "correct_answer": "C",
        "explanation": "A stack follows Last In, First Out."
    },
    {
        "question_id": "assess_queues_01",
        "concept_id": "queues",
        "question": "Which principle does a standard queue follow?",
        "options": {
            "A": "LIFO",
            "B": "FIFO",
            "C": "Random access",
            "D": "Sorted access"
        },
        "correct_answer": "B",
        "explanation": "A standard queue follows First In, First Out."
    },
    {
        "question_id": "assess_recursion_01",
        "concept_id": "recursion",
        "question": "What is essential to prevent infinite recursive calls?",
        "options": {
            "A": "A hash function",
            "B": "A base case",
            "C": "A sorting operation",
            "D": "A queue"
        },
        "correct_answer": "B",
        "explanation": "A base case provides a stopping condition for recursion."
    },
    {
        "question_id": "assess_trees_01",
        "concept_id": "trees",
        "question": "In a tree data structure, what is the topmost node called?",
        "options": {
            "A": "Leaf",
            "B": "Sibling",
            "C": "Root",
            "D": "Edge"
        },
        "correct_answer": "C",
        "explanation": "The root is the topmost node of a tree."
    },
    {
        "question_id": "assess_bst_01",
        "concept_id": "binary_search_trees",
        "question": "In a binary search tree, values in the left subtree of a node are generally:",
        "options": {
            "A": "Greater than the node",
            "B": "Smaller than the node",
            "C": "Always equal to the node",
            "D": "Stored randomly"
        },
        "correct_answer": "B",
        "explanation": "In the standard BST ordering, left-subtree values are smaller than the node."
    },
    {
        "question_id": "assess_heaps_01",
        "concept_id": "heaps",
        "question": "In a max-heap, which element is at the root?",
        "options": {
            "A": "The smallest element",
            "B": "The largest element",
            "C": "The oldest element",
            "D": "A random element"
        },
        "correct_answer": "B",
        "explanation": "The maximum element is stored at the root of a max-heap."
    },
    {
        "question_id": "assess_sorting_01",
        "concept_id": "sorting",
        "question": "Which sorting algorithm repeatedly selects the minimum element from the unsorted portion?",
        "options": {
            "A": "Selection sort",
            "B": "Merge sort",
            "C": "Quick sort",
            "D": "Heap sort"
        },
        "correct_answer": "A",
        "explanation": "Selection sort repeatedly selects the minimum remaining element."
    },
    {
        "question_id": "assess_hashing_01",
        "concept_id": "hashing",
        "question": "What is the main purpose of a hash function in a hash table?",
        "options": {
            "A": "To sort all stored values",
            "B": "To map a key to an index or table position",
            "C": "To reverse a linked list",
            "D": "To traverse a graph"
        },
        "correct_answer": "B",
        "explanation": "A hash function maps a key to a table position."
    },
    {
        "question_id": "assess_graphs_01",
        "concept_id": "graphs",
        "question": "What does an edge represent in a graph?",
        "options": {
            "A": "A connection between vertices",
            "B": "The root of a tree",
            "C": "An array index",
            "D": "A sorting operation"
        },
        "correct_answer": "A",
        "explanation": "An edge represents a connection between vertices."
    },
    {
        "question_id": "assess_bfs_01",
        "concept_id": "bfs",
        "question": "Which data structure is commonly used to manage the frontier in Breadth-First Search?",
        "options": {
            "A": "Stack",
            "B": "Queue",
            "C": "Heap",
            "D": "Hash function"
        },
        "correct_answer": "B",
        "explanation": "BFS uses a queue to explore vertices level by level."
    },
    {
        "question_id": "assess_dfs_01",
        "concept_id": "dfs",
        "question": "Which data structure is commonly used to implement iterative Depth-First Search?",
        "options": {
            "A": "Queue",
            "B": "Stack",
            "C": "Priority queue",
            "D": "Sorted array"
        },
        "correct_answer": "B",
        "explanation": "Iterative DFS uses a stack to explore deeper vertices first."
    }
]


def get_assessment_questions(concept_id=None):
    """
    Return assessment questions, optionally filtered by concept_id.
    Correct answers are not exposed to the student.
    """
    questions = QUESTION_BANK

    if concept_id is not None:
        questions = [
            q for q in questions
            if q["concept_id"] == concept_id
        ]

    return [
        {
            "question_id": q["question_id"],
            "concept_id": q["concept_id"],
            "question": q["question"],
            "options": q["options"]
        }
        for q in questions
    ]


def evaluate_assessment_answer(question_id, answer):
    """
    Evaluate an MCQ response.
    Accepts an option letter (A/B/C/D) or the exact option text.
    Returns a binary outcome suitable for BKT.
    """
    question = next(
        (q for q in QUESTION_BANK if q["question_id"] == question_id),
        None
    )

    if question is None:
        raise ValueError(f"Unknown question_id: {question_id}")

    if answer is None or not str(answer).strip():
        raise ValueError("Answer cannot be empty.")

    submitted = str(answer).strip().casefold()

    correct_option = question["correct_answer"]
    correct_text = question["options"][correct_option]

    valid_answers = {
        correct_option.casefold(),
        correct_text.casefold()
    }

    outcome = int(submitted in valid_answers)

    return {
        "question_id": question_id,
        "concept_id": question["concept_id"],
        "outcome": outcome,
        "classification": "correct" if outcome == 1 else "incorrect",
        "feedback": (
            "Correct! " + question["explanation"]
            if outcome == 1
            else "Incorrect. " + question["explanation"]
        )
    }
