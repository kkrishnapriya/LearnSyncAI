"""
LearnSync AI - Member 2: Answer Evaluation Agent (evaluator.py)
==============================================================
This module implements the rule-based answer evaluation and misconception
identification system for student natural language responses, adhering strictly
to the LearnSync AI Hybrid Agentic Implementation Contract.

Classification Methodology:
    The evaluator is a deterministic, rule-based classifier utilizing:
        1. Canonical question rubrics and regular expressions.
        2. Pedagogical invariant coverage verification.
        3. Clause-aware negation and contrast detection.
    Dense SentenceTransformer cosine similarity is computed and logged as
    supporting evidence only and does NOT influence the rule-based classification
    or score assignment.

Contract Required Public Interfaces:
    1. evaluate_answer(question_id, student_answer)
    2. bkt_outcome_from_evaluation(evaluation, threshold=0.70)

Standard Answer-Evaluation Output Schema:
    {
        "question_id": str,              # original input ID, preserved exactly (e.g. 'Q_ARR_01', 'Q_STACK_01')
        "concept_id": str,
        "classification": "correct" | "partially_correct" | "incorrect",
        "score": float (0.0 to 1.0),
        "bkt_outcome": int (0 or 1),
        "identified_misconceptions": list[str],
        "feedback": str,
        "evidence": list[str]
    }

BKT Conversion Rule (Contract Part A & B):
    BKT observations require a strict binary input {0, 1}.
    A score is converted according to an explicit, documented rule:
        - outcome = 1 IF score >= 0.70 (Student demonstrated sufficient mastery)
        - outcome = 0 IF score <  0.70  (Student needs reinforcement before progressing)
    Approval status: Member 1 has approved bkt_outcome as an additional evaluator output
    field and has approved the configurable 0.70 threshold for curriculum decisions.
    Partial credit (score 0.5) maps to bkt_outcome = 0. Member 3 has agreed to accept
    bkt_outcome and pass it to update_bkt(); Member 3 will finalize its use in the
    curriculum planner.
"""

import os
import re
import csv
from typing import Dict, List, Any, Optional, Tuple

# Validated DSA concepts according to the LearnSync AI Implementation Contract
VALID_CONCEPTS = {
    "arrays", "linked_lists", "stacks", "queues", "recursion",
    "trees", "binary_search_trees", "heaps", "sorting", "hashing",
    "graphs", "bfs", "dfs"
}

# Reference question dictionary populated from short_answer_dataset.csv
_QUESTION_BANK: Dict[str, Dict[str, Any]] = {}

# Backward-compatibility alias map (e.g. legacy Q_STACK_01 -> canonical Q_STK_01)
_ALIAS_MAP = {
    "Q_STACK_01": "Q_STK_01",
    "Q_STACK_02": "Q_STK_02",
    "Q_STACK_03": "Q_STK_03",
}

# Lazy-loaded SentenceTransformer for supporting evidence logging
_EVAL_MODEL = None


def _get_eval_model():
    """
    Lazily loads SentenceTransformer for supporting evidence logging only.
    Dense similarity is logged as supporting evidence and does not affect classification.
    """
    global _EVAL_MODEL
    if _EVAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EVAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EVAL_MODEL = False
    return _EVAL_MODEL if _EVAL_MODEL is not False else None


def _is_negated_or_contrasted(match_span: Tuple[int, int], text: str) -> bool:
    """
    Detects whether a matched keyword or phrase is preceded by a negation operator
    or used within a contrastive clause (e.g., 'unlike a queue which is FIFO', 'not LIFO').

    Clause-bounded scope:
        Negation applies ONLY within the same clause or sentence.
        Lookback terminates at sentence boundaries ('.', ';', '!', '?', newline)
        or clause-reversal conjunctions ('however', 'but', 'although', 'yet').
    """
    start, _ = match_span
    text_before = text[:start]

    # Delimiters: sentence terminators and clause-reversal conjunctions
    delim_pattern = re.compile(r"[.;!?\n]|\b(?:however|but|although|yet)\b", re.IGNORECASE)
    matches = list(delim_pattern.finditer(text_before))
    if matches:
        last_delim_end = matches[-1].end()
        clause_prefix = text_before[last_delim_end:start].lower()
    else:
        clause_prefix = text_before[max(0, start - 100):start].lower()

    negation_patterns = [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bdoesn't\b", r"\bdoes not\b",
        r"\bisn't\b", r"\bis not\b", r"\bcannot\b", r"\bwithout\b",
        r"\bunlike\b", r"\bin contrast to\b", r"\bas opposed to\b",
        r"\brather than\b", r"\binstead of\b", r"\bwhereas\b", r"\bcontrasted with\b"
    ]
    for neg in negation_patterns:
        if re.search(neg, clause_prefix):
            return True
    return False


# -----------------------------------------------------------------------------
# Concept & Question-Specific Evaluation Rubrics (39 canonical DSA questions)
# -----------------------------------------------------------------------------
QUESTION_RUBRICS = {
    # -------------------------------------------------------------------------
    # ARRAYS
    # -------------------------------------------------------------------------
    ("arrays", 1): {
        "omission": "Omission of element shifting mechanism",
        "omission_fb": "Your answer correctly notes contiguous memory and O(1) random access, but omits the O(n) element shifting required when inserting at index 0.",
        "invariants": [
            (r"contiguous|direct|memory offset|formula|address", "contiguous memory layout / addressing"),
            (r"shift|mov(?:ing|e|es)?|displac", "element shifting mechanism for index 0 insertion")
        ],
        "critical": [
            {
                "name": "Believes array indexing requires sequential traversal",
                "patterns": [r"step(?:ping)? through|start at index 0|o\(k\)|step through each slot|sequential(?:ly)? until|scan sequentially"],
                "feedback": "Arrays occupy contiguous memory allowing direct arithmetic offset calculation (base + index * size) in O(1) time without sequential stepping."
            }
        ]
    },
    ("arrays", 2): {
        "omission": "Omission of doubling strategy and amortized complexity",
        "omission_fb": "Your answer explains reallocation and copying, but omits the doubling capacity strategy and the resulting amortized O(1) insertion time.",
        "invariants": [
            (r"allocat|new (?:bigger|larger|memory)|cop(?:y|ies|ying)", "reallocation and copying"),
            (r"doubl|twice|2x", "doubling capacity strategy"),
            (r"amortiz|averag", "amortized O(1) complexity")
        ],
        "critical": [
            {
                "name": "Assumes in-place contiguous memory expansion without reallocation",
                "patterns": [r"without (?:needing )?(?:any )?reallocation|automatically claims|next adjacent memory addresses in ram|claims the next adjacent"],
                "feedback": "Fixed memory blocks cannot simply claim adjacent RAM; dynamic arrays must allocate a new, larger memory block and copy elements over."
            }
        ]
    },
    ("arrays", 3): {
        "omission": "Omission of CPU cache line prefetching explanation",
        "omission_fb": "Your answer notes that arrays have sequential memory, but omits the CPU cache line prefetching mechanism that loads adjacent elements simultaneously.",
        "invariants": [
            (r"contiguous|sequential|right next to each other", "contiguous / sequential memory"),
            (r"cache line|prefetch|pulls? (?:nearby|adjacent)|spatial (?:cache )?locality", "CPU cache line loading / spatial locality")
        ],
        "critical": [
            {
                "name": "Believes memory hardware cannot traverse non-contiguous pointers",
                "patterns": [r"cannot (?:follow|traverse) pointers|only read continuous memory blocks|ram can only read"],
                "feedback": "Hardware memory can follow arbitrary pointer addresses; linked lists are slower because non-contiguous node addresses incur frequent CPU cache misses."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # LINKED LISTS
    # -------------------------------------------------------------------------
    ("linked_lists", 1): {
        "omission": "Omission of formal O(n) access complexity comparison",
        "omission_fb": "Your answer correctly notes pointer updates for head insertion, but omits the formal O(n) access complexity comparison against arrays.",
        "invariants": [
            (r"o\(1\)|constant", "O(1) head insertion"),
            (r"o\(n\)|linear|traverse|node by node", "O(n) sequential traversal to index")
        ],
        "critical": [
            {
                "name": "Assumes linked list nodes occupy contiguous memory enabling pointer arithmetic",
                "patterns": [r"directly adding the index|pointer arithmetic|o\(1\) random index access|index number to the head"],
                "feedback": "Linked list nodes are scattered dynamically across heap memory and connected by pointers, so direct index arithmetic is impossible."
            }
        ]
    },
    ("linked_lists", 2): {
        "omission": "Omission of O(1) deletion advantage and memory overhead analysis",
        "omission_fb": "Your answer notes bidirectional traversal, but omits the O(1) node deletion advantage (given a pointer) and the higher memory overhead from two pointers per node.",
        "invariants": [
            (r"prev|backward|both directions", "bidirectional traversal"),
            (r"delet|remove|o\(1\)", "O(1) deletion given pointer"),
            (r"overhead|extra (?:memory|space)|more (?:total )?memory|trade-off", "memory overhead of two pointers")
        ],
        "critical": [
            {
                "name": "Believes bidirectional pointers reduce node memory footprint",
                "patterns": [r"less (?:total )?memory|compress the data|reduce(?:s)? node memory"],
                "feedback": "Storing two pointer references (next and prev) strictly increases the memory overhead per node compared to singly linked lists."
            }
        ]
    },
    ("linked_lists", 3): {
        "omission": "Omission of predecessor next-pointer bypass mechanics",
        "omission_fb": "Your answer notes disconnecting pointers, but omits the specific mechanism of updating the predecessor's next pointer to bypass the deleted node.",
        "invariants": [
            (r"predecessor|previous node|node before", "predecessor node reference"),
            (r"next pointer|bypass|point to.*next", "updating next pointer to bypass")
        ],
        "critical": [
            {
                "name": "Assumes deleting a node triggers automatic contiguous physical contraction",
                "patterns": [r"automatically pulls? (?:adjacent )?nodes together|pulls adjacent nodes|physical contraction|do not need the predecessor"],
                "feedback": "Deleting a node in a linked list does not shift memory; you must manually update the predecessor's next pointer to point to the successor."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # STACKS
    # -------------------------------------------------------------------------
    ("stacks", 1): {
        "omission": "Omission of O(1) time complexity specification",
        "omission_fb": "Your answer captures the LIFO discipline, but omits the explicit O(1) constant time complexity for push and pop operations.",
        "invariants": [
            (r"lifo|last[- ]in[- ]first[- ]out", "LIFO discipline"),
            (r"o\(1\)|constant time", "O(1) push and pop complexities")
        ],
        "critical": [
            {
                "name": "Confuses LIFO with FIFO",
                "patterns": [
                    r"first[- ]in[- ]first[- ]out",
                    r"\bfifo\b",
                    r"supermarket queue",
                    r"\b(?:first|earliest|oldest)\s+(?:inserted|added|pushed|in)?\s*(?:element|item|node|entry)?\s*(?:is|gets)?\s*(?:the\s+)?(?:removed|popped|served|out|deleted)\s+first\b",
                    r"\b(?:first|earliest|oldest)\s+(?:element|item|node|entry)\s+(?:is|gets)?\s*(?:removed|popped|served|deleted)\s+first\b",
                    r"\bremove[sd]?\s+(?:the\s+)?(?:first|oldest|earliest)\s+(?:inserted|added|pushed)?\s*(?:element|item|node|entry)?\s+first\b",
                    r"\b(?:first|earliest|oldest)\s+(?:element|item|entry|node)\s+(?:that\s+was\s+)?(?:inserted|added|pushed)\s+is\s+(?:the\s+)?(?:first\s+)?(?:removed|popped|served|deleted)\b",
                ],
                "feedback": "A stack strictly operates under the Last-In First-Out (LIFO) rule; FIFO governs queues."
            }
        ]
    },
    ("stacks", 2): {
        "omission": "Omission of index mechanics and formal capacity bound",
        "omission_fb": "Your answer explains that an array stack pushes and tracks the top, but omits specific top-index arithmetic (+ 1) and formal capacity bounds causing overflow.",
        "invariants": [
            (r"top\s*\+\s*1|increment|integer index", "top index increment / integer index mechanics"),
            (r"capacity|maximum capacity|fixed[- ]size|limit|allocated slots|filled all", "formal array capacity bound")
        ],
        "critical": [
            {
                "name": "Believes stack top is located at array index 0 requiring shifting",
                "patterns": [r"inserts? at index 0 and shifts|index 0 requiring shifting|top is (?:located )?at array index 0"],
                "feedback": "Array stacks place the top at the highest occupied index so push and pop take O(1) time without shifting elements."
            }
        ]
    },
    ("stacks", 3): {
        "omission": "Omission of LIFO matching order justification",
        "omission_fb": "Your answer describes pushing opening and popping closing brackets, but omits the explicit justification that the most recently opened bracket must be closed first (LIFO).",
        "invariants": [
            (r"push.*open.*pop|opening.*closing", "pushing open and popping on closing"),
            (r"lifo|most recent|last opened|inner.*first", "LIFO matching order justification")
        ],
        "critical": [
            {
                "name": "Confuses nested bracket matching order with FIFO queueing",
                "patterns": [r"fifo order|first bracket opened must match the first bracket closed"],
                "feedback": "Parenthesis matching requires LIFO ordering because the innermost (most recently opened) bracket must be closed first."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # QUEUES
    # -------------------------------------------------------------------------
    ("queues", 1): {
        "omission": "Omission of FIFO acronym and O(1) time complexity",
        "omission_fb": "Your answer notes insertion at rear and removal at front, but omits the explicit FIFO designation and O(1) operational complexities.",
        "invariants": [
            (r"fifo|first[- ]in[- ]first[- ]out", "FIFO discipline"),
            (r"o\(1\)|constant time", "O(1) enqueue and dequeue complexities")
        ],
        "critical": [
            {
                "name": "Confuses FIFO with LIFO",
                "patterns": [r"last[- ]in[- ]first[- ]out|\blifo\b|popped from the top"],
                "feedback": "A queue strictly follows First-In First-Out (FIFO) ordering; Last-In First-Out (LIFO) is the rule for stacks."
            }
        ]
    },
    ("queues", 2): {
        "omission": "Omission of modulo arithmetic and O(n) shifting avoidance rationale",
        "omission_fb": "Your answer notes that circular queues wrap around, but omits the modulo arithmetic index formula and the avoidance of O(n) array element shifting.",
        "invariants": [
            (r"empty slots|reused|unusable", "empty slots at front of linear array"),
            (r"modulo|%|formula|shift|avoid.*shifting|o\(n\)", "modulo arithmetic or shifting avoidance")
        ],
        "critical": [
            {
                "name": "Believes circular queue allows arbitrary middle deletion",
                "patterns": [r"removing elements from the middle|arbitrary middle deletion|middle index"],
                "feedback": "A circular queue is strictly a FIFO structure restricted to front dequeue and rear enqueue; it does not allow arbitrary middle deletions."
            }
        ]
    },
    ("queues", 3): {
        "omission": "Omission of priority-based dequeue rule and binary heap backend",
        "omission_fb": "Your answer contrasts standard queues with priority queues, but omits the priority-based removal rule and binary heap backend.",
        "invariants": [
            (r"arrival order|fifo|first in, first out", "FIFO arrival order for standard queues"),
            (r"highest priority|priority key", "highest priority removal"),
            (r"heap|o\(log n\)", "binary heap implementation / O(log n)")
        ],
        "critical": [
            {
                "name": "Believes priority queue enforces identical priority across all elements",
                "patterns": [r"exact same priority value|forced to have the exact same|enforces? fifo order among elements with identical keys|strictly orders elements alphabetically"],
                "feedback": "A priority queue allows arbitrary distinct priority keys and removes the highest/lowest priority element, not forcing identical priorities."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # RECURSION
    # -------------------------------------------------------------------------
    ("recursion", 1): {
        "omission": "Omission of shrinking recursive step specification",
        "omission_fb": "Your answer correctly emphasizes the base case, but omits the requirement that the recursive step must strictly shrink the input toward the base case.",
        "invariants": [
            (r"base case", "terminating base case"),
            (r"shrink|reduc|smaller|converge|step toward", "shrinking recursive step toward base case")
        ],
        "critical": [
            {
                "name": "Confuses iteration with recursion",
                "patterns": [r"only requires a while loop|while loop and an accumulator|for loop"],
                "feedback": "Recursion relies on self-referential function calls pushing call stack frames, not iterative while/for loops."
            }
        ]
    },
    ("recursion", 2): {
        "omission": "Omission of stack frame allocation and auxiliary space bound",
        "omission_fb": "Your answer notes recursion and stack overflow, but omits the formal activation record allocation and O(depth) auxiliary memory complexity.",
        "invariants": [
            (r"call stack|stack frames?", "call stack / stack frames allocation"),
            (r"shrinks? toward|termination condition|base case", "shrinking toward base case / termination")
        ],
        "critical": [
            {
                "name": "Confuses system call stack memory with persistent disk storage",
                "patterns": [r"disk storage space|writing return values to the hard drive|hard drive|stored in heap memory|heap-allocated"],
                "feedback": "StackOverflowError is caused by exhausting system RAM allocated for the call stack, not disk storage space."
            }
        ]
    },
    ("recursion", 3): {
        "omission": "Omission of stack frame reuse mechanics and O(1) space optimization",
        "omission_fb": "Your answer defines tail recursion as the final call, but omits the compiler stack frame reuse optimization that achieves O(1) auxiliary space.",
        "invariants": [
            (r"last (?:operation|statement|call)|final operation", "final operation tail call definition"),
            (r"reuse.*(?:stack )?frame|turn.*iteration|o\(1\) space", "compiler stack frame reuse / O(1) space")
        ],
        "critical": [
            {
                "name": "Confuses tail recursion with initial recursive calls",
                "patterns": [r"calls itself at the beginning of the function body|beginning of the function body before any variables"],
                "feedback": "Tail recursion occurs strictly when the recursive call is the FINAL action of the function, not at the beginning."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # TREES
    # -------------------------------------------------------------------------
    ("trees", 1): {
        "omission": "Omission of acyclic constraint and root-leaf reachability property",
        "omission_fb": "Your answer states that a tree has n - 1 edges, but omits the essential acyclic property and root-to-leaf path connectivity.",
        "invariants": [
            (r"n\s*-\s*1\s*edges|one less edge than (?:the number of )?nodes", "n - 1 edge count"),
            (r"acyclic|no cycles|connected|hierarchical", "connected and acyclic constraint")
        ],
        "critical": [
            {
                "name": "Confuses tree edge count with complete graph edge count",
                "patterns": [r"n\s*\*\s*\(\s*n\s*-\s*1\s*\)\s*/\s*2|complete graph|complete edges"],
                "feedback": "A tree with n vertices contains exactly n - 1 edges. The formula n(n - 1)/2 calculates edge count in a complete graph."
            }
        ]
    },
    ("trees", 2): {
        "omission": "Omission of postorder traversal mechanics and application",
        "omission_fb": "Your answer explains preorder and inorder traversals, but omits postorder traversal mechanics (Left-Right-Node) and bottom-up applications.",
        "invariants": [
            (r"preorder.*(?:root|node).*first|preorder visits node", "preorder traversal mechanics"),
            (r"inorder.*left.*(?:node|root).*right|inorder.*sorted", "inorder traversal mechanics"),
            (r"postorder|left.*right.*node|bottom-up|delet", "postorder traversal mechanics")
        ],
        "critical": [
            {
                "name": "Confuses inorder tree traversal with breadth-first queue traversal",
                "patterns": [r"inorder traversal visits all nodes level by level|level by level starting from the root using a fifo queue|preorder.*(?:yields|produces) sorted"],
                "feedback": "Inorder traversal is a depth-first traversal (Left-Node-Right), not a level-by-level BFS queue traversal."
            }
        ]
    },
    ("trees", 3): {
        "omission": "Omission of exact leaf invariant formula L = I + 1",
        "omission_fb": "Your answer notes leaf and internal nodes, but omits the exact Full Binary Tree Theorem formula L = I + 1.",
        "invariants": [
            (r"l\s*=\s*i\s*\+\s*1|one more leaf than internal|leaves = internal \+ 1|leaf nodes equals the number of internal nodes plus one", "leaf invariant L = I + 1")
        ],
        "critical": [
            {
                "name": "Assumes all binary trees have identical leaf and internal node counts",
                "patterns": [r"must always have an equal number of leaves and internal nodes|every binary tree has equal numbers of leaves and internal nodes|identical leaf and internal node counts"],
                "feedback": "Full binary trees always satisfy L = I + 1; leaf count strictly exceeds internal node count by one."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # BINARY SEARCH TREES
    # -------------------------------------------------------------------------
    ("binary_search_trees", 1): {
        "omission": "Omission of skewed tree degradation scenario and balanced condition",
        "omission_fb": "Your answer defines the BST invariant, but omits the balanced tree condition (O(log n)) versus the degenerate skewed worst-case (O(n)).",
        "invariants": [
            (r"left.*(?:less|smaller).*right.*greater", "BST ordering comparison rule"),
            (r"o\(n\).*skewed|skewed.*o\(n\)", "skewed tree O(n) worst case"),
            (r"o\(log n\).*balanced|balanced.*o\(log n\)", "balanced tree O(log n) performance")
        ],
        "critical": [
            {
                "name": "Assumes BST search is always guaranteed O(log n) regardless of tree shape",
                "patterns": [r"always strictly o\(log n\)|guaranteed o\(log n\) regardless|never degrades|only (?:checks?|needs to check) immediate children"],
                "feedback": "BST search degrades to O(n) when keys are inserted in sorted order, forming a skewed linked list."
            }
        ]
    },
    ("binary_search_trees", 2): {
        "omission": "Omission of inorder predecessor alternative and recursive deletion mechanics",
        "omission_fb": "Your answer correctly identifies the inorder successor replacement, but omits the symmetric inorder predecessor alternative and recursive deletion mechanics.",
        "invariants": [
            (r"inorder successor|smallest in (?:the )?right", "inorder successor replacement"),
            (r"inorder predecessor|largest in (?:the )?left|recursive", "inorder predecessor alternative / recursive deletion")
        ],
        "critical": [
            {
                "name": "Believes deleting a node with two children deletes both subtrees",
                "patterns": [r"automatically deletes both of its entire subtrees|promotes both children|splits into two independent trees"],
                "feedback": "Deleting a node with two children never deletes subtrees; it replaces the node's key with its inorder predecessor or successor."
            }
        ]
    },
    ("binary_search_trees", 3): {
        "omission": "Omission of formal height h = n derivation and O(n) operation impact",
        "omission_fb": "Your answer describes right-child chaining, but omits the formal tree height h = n derivation and the resulting O(n) search and insert complexity.",
        "invariants": [
            (r"right child|straight line|chain", "right-child degenerate chain"),
            (r"height (?:of )?n|h\s*=\s*n|degrade to o\(n\)|o\(n\)", "height n / O(n) search and insert degradation")
        ],
        "critical": [
            {
                "name": "Believes pre-sorted keys represent the ideal BST insertion order with O(1) search",
                "patterns": [r"ideal case for a bst|immediately find any element in o\(1\)|pre-sorted keys is the ideal"],
                "feedback": "Pre-sorted keys are the WORST case for an unaugmented BST, degenerating it into a linked list taking O(n) time."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # HEAPS
    # -------------------------------------------------------------------------
    ("heaps", 1): {
        "omission": "Omission of complete binary tree structural property",
        "omission_fb": "Your answer captures the heap-order property, but omits the complete binary tree structural invariant required for compact array storage.",
        "invariants": [
            (r"heap[- ]order|parent.*child|max-heap|min-heap|greater than or equal to its children", "heap-order property"),
            (r"root", "maximum key at root"),
            (r"o\(log n\).*bubbl|bubbl.*o\(log n\)", "O(log n) bubbling up insertion")
        ],
        "critical": [
            {
                "name": "Confuses heap order with total sorted order",
                "patterns": [r"fully sorted from left to right|sorted like an array|all elements are sorted"],
                "feedback": "A heap only enforces a partial ordering (parent vs child); siblings are not ordered, so heap elements are not totally sorted."
            }
        ]
    },
    ("heaps", 2): {
        "omission": "Omission of parent index arithmetic formula and gapless completeness justification",
        "omission_fb": "Your answer gives child index formulas, but omits the parent index arithmetic (i - 1)//2 and the gapless completeness property of heaps.",
        "invariants": [
            (r"2i\s*\+\s*1|2i\s*\+\s*2", "child index arithmetic"),
            (r"\(i\s*-\s*1\)\s*//\s*2", "parent index arithmetic"),
            (r"complete tree|no gaps|no missing slots", "complete tree / no gaps property")
        ],
        "critical": [
            {
                "name": "Believes heap array stores explicit memory pointers in each slot",
                "patterns": [r"stores pointer addresses inside each array slot|locate children in memory using pointers|heap array indexing requires pointer"],
                "feedback": "Array-backed heaps compute child and parent locations strictly using direct index arithmetic without pointer storage."
            }
        ]
    },
    ("heaps", 3): {
        "omission": "Omission of geometric summation bound explanation",
        "omission_fb": "Your answer identifies linear O(n) heap construction, but omits the geometric summation explanation showing that lower nodes have smaller heights.",
        "invariants": [
            (r"bottom[- ]up|sift(?:ing)?[- ]down|bottom to top", "bottom-up processing / sift-down"),
            (r"o\(n\)|linear", "linear O(n) construction time"),
            (r"sum|height|lower nodes|geometric|most nodes are near the leaves", "geometric summation / node height distribution")
        ],
        "critical": [
            {
                "name": "Believes any complete binary tree automatically satisfies the heap invariant without swaps",
                "patterns": [r"automatically valid heaps when stored in an array|takes o\(1\) constant time because all complete binary trees|always requires o\(n log n\)"],
                "feedback": "Complete binary trees require bottom-up sift-down swaps to establish the heap order property; they are not automatically valid heaps."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # SORTING
    # -------------------------------------------------------------------------
    ("sorting", 1): {
        "omission": "Omission of Quicksort worst-case pivot degradation scenario",
        "omission_fb": "Your answer notes the Omega(n log n) comparison sorting lower bound and Merge Sort's guarantee, but omits Quicksort's worst-case O(n^2) degradation.",
        "invariants": [
            (r"omega\(n log n\)", "Omega(n log n) lower bound"),
            (r"merge sort.*o\(n log n\)", "Merge Sort O(n log n) in all cases"),
            (r"quicksort.*o\(n\^2\)|worst case.*pivot", "Quicksort O(n^2) worst case pivot degradation")
        ],
        "critical": [
            {
                "name": "Confuses sub-linear complexity with sorting runtime and inverts complexities",
                "patterns": [r"quicksort is always faster with o\(log n\)|o\(log n\) total runtime|sort in o\(n\) using comparisons"],
                "feedback": "Comparison-based sorting cannot run in O(log n) or O(n); information theory establishes a strict lower bound of Omega(n log n)."
            }
        ]
    },
    ("sorting", 2): {
        "omission": "Omission of inner loop early-termination mechanism",
        "omission_fb": "Your answer notes that Insertion Sort runs in O(n) time on nearly-sorted data, but omits the early-termination condition of the inner loop.",
        "invariants": [
            (r"adaptive|terminates early", "adaptive behavior / early termination"),
            (r"smaller value|correct place", "stops when hitting smaller element"),
            (r"o\(n\)|linear", "O(n) linear time on nearly sorted data")
        ],
        "critical": [
            {
                "name": "Believes insertion sort lacks early-exit adaptive behavior",
                "patterns": [r"takes o\(n\^2\) time regardless of input order|must compare every element with every other element|insertion sort is o\(n log n\) in all cases"],
                "feedback": "Insertion Sort's inner loop stops immediately once a smaller element is encountered, giving adaptive linear O(n) time on nearly-sorted data."
            }
        ]
    },
    ("sorting", 3): {
        "omission": "Omission of recursive stack space bound for Quicksort",
        "omission_fb": "Your answer compares Merge Sort auxiliary space with in-place Quicksort, but omits the O(log n) auxiliary stack space required for Quicksort's recursion.",
        "invariants": [
            (r"merge sort.*o\(n\) (?:extra|auxiliary) memory", "Merge Sort O(n) auxiliary space"),
            (r"quicksort.*in[- ]place", "Quicksort in-place sorting"),
            (r"o\(log n\) stack|stack space for recursion", "Quicksort O(log n) recursion stack space")
        ],
        "critical": [
            {
                "name": "Inverts auxiliary space requirements between merge sort and quicksort",
                "patterns": [r"merge sort uses o\(1\) space|quicksort creates o\(n\^2\) temporary duplicate arrays"],
                "feedback": "Merge Sort requires O(n) auxiliary memory for merging, whereas Quicksort operates in-place using O(log n) stack space."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # HASHING
    # -------------------------------------------------------------------------
    ("hashing", 1): {
        "omission": "Omission of worst-case collision degradation scenario",
        "omission_fb": "Your answer explains average O(1) hash lookup, but omits the worst-case O(n) performance degradation when collisions saturate bucket chains.",
        "invariants": [
            (r"o\(1\)|constant|average|expected", "expected / average O(1) lookup"),
            (r"o\(n\)|linear|worst|collision|degrad", "worst-case O(n) collision degradation")
        ],
        "critical": [
            {
                "name": "Believes hash tables guarantee worst-case O(1) and collisions never occur",
                "patterns": [r"guarantees? o\(1\) (?:search )?time in all cases|hash codes are always unique|collisions never occur"],
                "feedback": "Hash table O(1) performance is strictly average-case; collisions can degrade performance to linear O(n) in the worst case."
            }
        ]
    },
    ("hashing", 2): {
        "omission": "Omission of load factor alpha tolerance and search cost O(1 + alpha)",
        "omission_fb": "Your answer describes separate chaining bucket lists, but omits the load factor alpha tolerance (> 1.0) and O(1 + alpha) search cost.",
        "invariants": [
            (r"linked list|bucket list|chain", "bucket linked lists for collisions"),
            (r"load factor|alpha|o\(1\s*\+\s*alpha\)|greater than 1|saturation", "load factor alpha tolerance / search cost")
        ],
        "critical": [
            {
                "name": "Believes separate chaining overwrites and destroys collided entries",
                "patterns": [r"discarding older keys|overwrites? (?:and destroys? )?collided entries|overwrites the old key"],
                "feedback": "Separate chaining preserves all keys by appending collided entries into a linked list at that bucket, never discarding existing data."
            }
        ]
    },
    ("hashing", 3): {
        "omission": "Omission of linear probing mechanism and step-size causation",
        "omission_fb": "Your answer notes that clustering occurs from collisions, but omits the step-by-step linear probing mechanism causing continuous block clustering.",
        "invariants": [
            (r"linear probing|\(index\s*\+\s*i\)\s*%\s*m|next (?:available )?slot|step size", "linear probing mechanism"),
            (r"primary clustering|clusters?|consecutive|adjacent slots", "primary clustering explanation")
        ],
        "critical": [
            {
                "name": "Confuses open addressing clustering with separate chaining bucket lists",
                "patterns": [r"clustering only occurs in separate chaining|primary clustering only occurs in separate chaining"],
                "feedback": "Primary clustering is a flaw exclusive to open addressing linear probing; separate chaining uses independent bucket lists and does not exhibit primary clustering."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # GRAPHS
    # -------------------------------------------------------------------------
    ("graphs", 1): {
        "omission": "Omission of sparse graph condition |E| << |V|^2 and formal space bounds",
        "omission_fb": "Your answer notes that adjacency lists use less memory, but omits the formal sparse graph condition |E| << |V|^2 and exact space bounds.",
        "invariants": [
            (r"o\(\|?[vV]\|?\s*\+\s*\|?[eE]\|?\)|list.*less memory|sparse", "adjacency list space efficiency for sparse graphs"),
            (r"o\(\|?[vV]\|?\^2\)|matrix.*v\^2|dense|\|e\|\s*<<\s*\|v\|\^2", "adjacency matrix O(|V|^2) space / sparse condition")
        ],
        "critical": [
            {
                "name": "Believes adjacency matrix is space-efficient for sparse graphs",
                "patterns": [r"matrices are always faster and use less space for sparse|matrix is compact for sparse|matrix uses less space for sparse"],
                "feedback": "Adjacency matrices allocate O(|V|^2) space regardless of edge count, making them highly wasteful for sparse graphs."
            }
        ]
    },
    ("graphs", 2): {
        "omission": "Omission of visited tracking loop and O(V + E) complexity bound",
        "omission_fb": "Your answer explains running graph traversals to find components, but omits the outer visited tracking loop and the formal O(|V| + |E|) time complexity.",
        "invariants": [
            (r"bfs|dfs|traversal|search", "graph traversal mechanism (BFS/DFS)"),
            (r"visited|o\(\|?[vV]\|?\s*\+\s*\|?[eE]\|?\)|loop.*unvisited", "visited tracking loop / O(V + E) bound")
        ],
        "critical": [
            {
                "name": "Confuses connected component count with Euler characteristic",
                "patterns": [r"number of vertices minus the number of edges|\|v\|\s*-\s*\|e\||euler characteristic"],
                "feedback": "Connected components are calculated via graph traversals in O(|V| + |E|) time, not by computing |V| - |E| (which is the Euler characteristic)."
            }
        ]
    },
    ("graphs", 3): {
        "omission": "Omission of acyclic constraint and formal prerequisite definition u -> v",
        "omission_fb": "Your answer notes ordering by directed arrows, but omits the strict directed acyclic graph (DAG) constraint and the formal dependency definition.",
        "invariants": [
            (r"acyclic|no (?:directed )?cycles|dag", "DAG / acyclic graph constraint"),
            (r"linear order|dependencies|u.*v|prerequisite|in-degree", "linear dependency ordering definition")
        ],
        "critical": [
            {
                "name": "Believes topological sort works on cyclic graphs and orders by edge weight",
                "patterns": [r"sort any graph, including cyclic|highest edge weight first|cyclic graphs, by placing vertices with the highest edge weight"],
                "feedback": "Topological sorting is strictly impossible on graphs with directed cycles, and is determined by dependency arrows, not edge weights."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # BFS
    # -------------------------------------------------------------------------
    ("bfs", 1): {
        "omission": "Omission of unweighted shortest path guarantee",
        "omission_fb": "Your answer notes that BFS uses a queue to traverse level by level, but omits the mathematical shortest path guarantee on unweighted graphs.",
        "invariants": [
            (r"queue|fifo", "FIFO queue data structure"),
            (r"shortest path|unweighted|minimum (?:number of )?edges", "shortest path guarantee on unweighted graphs")
        ],
        "critical": [
            {
                "name": "Confuses BFS queue with DFS stack and Dijkstra/Bellman-Ford weights",
                "patterns": [r"uses a lifo stack|stack and finds shortest|negative weighted edges"],
                "feedback": "BFS uses a FIFO queue (not a stack) and guarantees shortest paths ONLY on unweighted graphs, not weighted graphs."
            }
        ]
    },
    ("bfs", 2): {
        "omission": "Omission of infinite loop and queue overflow failure modes",
        "omission_fb": "Your answer notes that a visited set prevents repeating work, but omits the specific failure modes: infinite loops and queue memory exhaustion.",
        "invariants": [
            (r"track|seen|mark|visited", "tracking visited vertices"),
            (r"infinite loop|loop forever|queue overflow|re-enqueue|memory exhaustion", "infinite loop and memory overflow failure modes")
        ],
        "critical": [
            {
                "name": "Assumes queue data structure inherently filters duplicate vertex visits",
                "patterns": [r"visited set is not needed in bfs|queue automatically prevents nodes from being visited twice"],
                "feedback": "A queue simply buffers elements in arrival order; it does not check for duplicates, making an explicit visited set mandatory."
            }
        ]
    },
    ("bfs", 3): {
        "omission": "Omission of conflict detection condition (edge between identical colors / odd cycle)",
        "omission_fb": "Your answer describes assigning two alternating colors, but omits the conflict detection condition: an edge connecting two vertices of identical color indicates an odd cycle.",
        "invariants": [
            (r"alternat|two colors?|color", "alternating 2-coloring process"),
            (r"conflict|same color|identical color|odd cycle", "conflict detection condition / odd cycle")
        ],
        "critical": [
            {
                "name": "Confuses bipartite 2-coloring with Eulerian circuit degree conditions",
                "patterns": [r"every vertex has an even degree|eulerian circuit"],
                "feedback": "Bipartiteness requires 2-colorability without odd cycles; even vertex degrees govern Eulerian circuits, not bipartiteness."
            }
        ]
    },

    # -------------------------------------------------------------------------
    # DFS
    # -------------------------------------------------------------------------
    ("dfs", 1): {
        "omission": "Omission of LIFO stack mechanism and O(V + E) complexity",
        "omission_fb": "Your answer explains deep exploration and backtracking, but omits the LIFO stack mechanism (or call stack) and the formal O(|V| + |E|) time complexity.",
        "invariants": [
            (r"deep|backtrack", "deep branch exploration and backtracking"),
            (r"lifo|stack|call stack|o\(\|?[vV]\|?\s*\+\s*\|?[eE]\|?\)", "LIFO stack mechanism / O(V + E) complexity")
        ],
        "critical": [
            {
                "name": "Confuses DFS depth exploration with BFS level exploration",
                "patterns": [r"uses a fifo queue to visit all neighbors at distance 1|level by level"],
                "feedback": "DFS explores deep paths using a LIFO stack; level-by-level exploration using a FIFO queue is the defining property of BFS."
            }
        ]
    },
    ("dfs", 2): {
        "omission": "Omission of forward/cross edge categories and ancestor definition",
        "omission_fb": "Your answer identifies tree and back edges, but omits forward and cross edges and the formal ancestor definition of back edges.",
        "invariants": [
            (r"back edge.*cycle|cycle.*back edge", "back edge cycle detection"),
            (r"forward|cross|ancestor|four (?:edge )?types", "forward/cross edges and ancestor definition")
        ],
        "critical": [
            {
                "name": "Confuses forward edges with back edges in cycle detection",
                "patterns": [r"forward edge indicates a cycle|forward edges with back edges in cycle detection"],
                "feedback": "Forward edges lead to already-explored descendants and do not form cycles; ONLY back edges (pointing to ancestors) confirm cycles."
            }
        ]
    },
    ("dfs", 3): {
        "omission": "Omission of active recursion stack tracking vs completed visited state",
        "omission_fb": "Your answer notes arrow directions in directed graphs, but omits active recursion stack tracking (3-state coloring) versus completed visited states.",
        "invariants": [
            (r"recursion (?:call )?stack|active stack|3-state|currently visiting|ancestor", "active recursion stack tracking / 3-state system"),
            (r"undirected|completed|cross edge|already visited", "distinguishing active from completed visited states")
        ],
        "critical": [
            {
                "name": "Assumes cross edges in directed graphs constitute cycles",
                "patterns": [r"any edge pointing to any already-visited vertex guarantees a cycle|cross edges in directed graphs constitute cycles"],
                "feedback": "In directed graphs, an edge to a completed vertex is a cross edge (not a cycle); cycles require an edge to a node currently on the recursion stack."
            }
        ]
    }
}


def _load_questions(csv_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Loads reference questions from short_answer_dataset.csv into the assessment bank.
    Validates required columns, concepts, and score ranges.
    """
    global _QUESTION_BANK
    if _QUESTION_BANK and csv_path is None:
        return _QUESTION_BANK

    target = csv_path
    if target is None:
        default_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "short_answer_dataset.csv")
        target = default_p if os.path.exists(default_p) else "short_answer_dataset.csv"

    if os.path.exists(target):
        with open(target, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row["question_id"].strip()
                concept = row.get("concept_id", "unmapped").strip().lower()
                if concept not in VALID_CONCEPTS and concept != "unmapped":
                    concept = "unmapped"

                _QUESTION_BANK[qid] = {
                    "question_id": qid,
                    "concept_id": concept,
                    "question_text": row.get("question_text", "").strip(),
                    "reference_answer": row.get("reference_answer", "").strip(),
                    "data_source": row.get("data_source", "synthetic_demo").strip()
                }

    # Populate alias mapping (e.g. Q_STACK_01 -> Q_STK_01)
    for alias_id, target_id in _ALIAS_MAP.items():
        if target_id in _QUESTION_BANK and alias_id not in _QUESTION_BANK:
            aliased_item = dict(_QUESTION_BANK[target_id])
            aliased_item["question_id"] = alias_id
            _QUESTION_BANK[alias_id] = aliased_item

    # Fallback default if CSV is completely unavailable
    if not _QUESTION_BANK:
        _QUESTION_BANK["Q_STK_01"] = {
            "question_id": "Q_STK_01",
            "concept_id": "stacks",
            "question_text": "What principle governs stack operations, and what are the time complexities of push and pop?",
            "reference_answer": "A stack operates on the Last-In First-Out (LIFO) discipline. Push and pop both execute in O(1) constant time.",
            "data_source": "reviewed_reference"
        }
        _QUESTION_BANK["Q_STACK_01"] = dict(_QUESTION_BANK["Q_STK_01"])

    return _QUESTION_BANK


def bkt_outcome_from_evaluation(evaluation: Dict[str, Any], threshold: float = 0.70) -> int:
    """
    Maps an answer evaluation score to a binary BKT observation {0, 1}
    under the agreed LearnSync AI contract rule:
        - Outcome = 1 if score >= threshold (default 0.70)
        - Outcome = 0 if score < threshold (default 0.70)

    Parameters:
        evaluation (Dict[str, Any]): Evaluator output containing a 'score' key in [0.0, 1.0].
        threshold (float): Mastery cutoff threshold (provisional default: 0.70).

    Returns:
        int: Strict binary observation (0 or 1). Never fractional.
    """
    if not isinstance(evaluation, dict):
        return 0
    try:
        score = float(evaluation.get("score", 0.0))
    except (ValueError, TypeError):
        score = 0.0
    return 1 if score >= threshold else 0


def evaluate_answer(question_id: str, student_answer: str) -> Dict[str, Any]:
    """
    Evaluates a student's natural-language answer to a DSA assessment question.

    Detects conceptual correctness, identifies specific DSA misconceptions and omissions,
    assigns a score in [0.0, 1.0], derives binary BKT outcome, and produces targeted feedback.

    Parameters:
        question_id (str): Unique identifier of the question (e.g. 'Q_STK_01', 'Q_STACK_01').
        student_answer (str): Natural language text written by the student.

    Returns:
        Dict[str, Any]: Standard answer-evaluation output adhering to Contract Part B:
            - question_id (str): original input ID, preserved exactly (e.g. 'Q_ARR_01', 'Q_STACK_01')
            - concept_id (str)
            - classification ("correct" | "partially_correct" | "incorrect")
            - score (float: 1.0, 0.5, or 0.0)
            - bkt_outcome (int: 1 or 0)
            - identified_misconceptions (list of str)
            - feedback (str)
            - evidence (list of str)
    """
    _load_questions()

    # Input validation and defensive programming
    if not isinstance(question_id, str):
        return {
            "question_id": str(question_id),
            "concept_id": "unmapped",
            "classification": "incorrect",
            "score": 0.0,
            "bkt_outcome": 0,
            "identified_misconceptions": ["Invalid question_id type"],
            "feedback": f"Question identifier must be a string, got {type(question_id).__name__}.",
            "evidence": ["Question ID type validation failed."]
        }

    # Resolve aliases (e.g., Q_STACK_01 -> Q_STK_01)
    resolved_qid = _ALIAS_MAP.get(question_id, question_id)

    if resolved_qid not in _QUESTION_BANK:
        return {
            "question_id": question_id,
            "concept_id": "unmapped",
            "classification": "incorrect",
            "score": 0.0,
            "bkt_outcome": 0,
            "identified_misconceptions": [f"Unknown question_id: '{question_id}'"],
            "feedback": f"Question identifier '{question_id}' was not recognized in the assessment bank.",
            "evidence": [f"Question ID '{question_id}' not found in loaded question bank."]
        }

    q_info = _QUESTION_BANK[resolved_qid]
    concept_id = q_info.get("concept_id", "unmapped")
    reference_answer = q_info.get("reference_answer", "")

    if not isinstance(student_answer, str):
        if student_answer is None:
            student_answer_clean = ""
        else:
            return {
                "question_id": question_id,
                "concept_id": concept_id,
                "classification": "incorrect",
                "score": 0.0,
                "bkt_outcome": 0,
                "identified_misconceptions": ["Invalid input type for student answer"],
                "feedback": f"Expected string for student_answer, got {type(student_answer).__name__}.",
                "evidence": ["Input type validation failed."]
            }
    else:
        student_answer_clean = student_answer.strip()

    # Handle blank answer
    if not student_answer_clean:
        return {
            "question_id": question_id,
            "concept_id": concept_id,
            "classification": "incorrect",
            "score": 0.0,
            "bkt_outcome": 0,
            "identified_misconceptions": ["Blank or empty submission"],
            "feedback": "No answer was provided. Please attempt to explain the concept.",
            "evidence": ["Input text was empty or whitespace."]
        }

    # Determine group index from question_id (e.g. Q_ARR_05 -> num=5 -> group_idx=2)
    match_num = re.search(r'_(\d+)$', resolved_qid)
    group_idx = 1
    if match_num:
        num = int(match_num.group(1))
        group_idx = (num - 1) // 3 + 1

    rubric = QUESTION_RUBRICS.get((concept_id, group_idx))

    # Dense similarity measurement via SentenceTransformers (supporting evidence only; does not affect classification)
    dense_sim = None
    model = _get_eval_model()
    if model and reference_answer:
        try:
            import numpy as np
            embs = model.encode([student_answer_clean, reference_answer], normalize_embeddings=True)
            dense_sim = float(np.dot(embs[0], embs[1]))
        except Exception:
            dense_sim = None

    # Step 1: Detect critical misconceptions with negation & contrast defense
    detected_critical = []
    evidence = []

    if rubric:
        for misc in rubric.get("critical", []):
            for pat in misc["patterns"]:
                m = re.search(pat, student_answer_clean, re.IGNORECASE)
                if m:
                    # Check if keyword/phrase is negated or contrasted
                    if not _is_negated_or_contrasted(m.span(), student_answer_clean):
                        detected_critical.append(misc)
                        evidence.append(f"Detected conceptual error: '{m.group(0)}'")
                        break

    if dense_sim is not None:
        evidence.append(f"Supporting evidence (cosine similarity to reference): {dense_sim:.4f}")

    # Step 2: If critical misconception detected -> INCORRECT
    if detected_critical:
        classification = "incorrect"
        score = 0.0
        feedback = " ".join(c["feedback"] for c in detected_critical)
        identified_misconceptions = [c["name"] for c in detected_critical]

    # Step 3: Check invariant coverage
    elif rubric:
        missing_invariants = []
        for pat, desc in rubric.get("invariants", []):
            if not re.search(pat, student_answer_clean, re.IGNORECASE):
                missing_invariants.append(desc)

        if missing_invariants:
            classification = "partially_correct"
            score = 0.5
            feedback = rubric["omission_fb"]
            identified_misconceptions = [rubric["omission"]]
            evidence.extend([f"Omission: Missing {m}" for m in missing_invariants])
        else:
            classification = "correct"
            score = 1.0
            feedback = f"Excellent! Your answer accurately captures the core principles and operational mechanics of {concept_id.replace('_', ' ')}."
            identified_misconceptions = []
            evidence.append("All required conceptual invariants verified.")

    else:
        # Fallback when no specific rubric is registered
        classification = "partially_correct"
        score = 0.5
        feedback = f"Your answer touches on {concept_id.replace('_', ' ')}, but lacked specific validation rubrics."
        identified_misconceptions = []

    bkt_outcome = bkt_outcome_from_evaluation({"score": score})

    return {
        "question_id": question_id,
        "concept_id": concept_id,
        "classification": classification,
        "score": float(score),
        "bkt_outcome": int(bkt_outcome),
        "identified_misconceptions": identified_misconceptions,
        "feedback": feedback,
        "evidence": evidence
    }


# Self-test when executed directly
if __name__ == "__main__":
    print("=" * 70)
    print("LearnSync AI - Member 2: Answer Evaluation Agent Self-Verification")
    print("=" * 70)

    _load_questions()
    print(f"[OK] Loaded {len(_QUESTION_BANK)} questions into assessment bank.")

    # Test Case 1: Correct Answer
    ans_correct = "A stack operates on Last-In First-Out (LIFO) discipline. Push and pop both take O(1) time."
    res1 = evaluate_answer("Q_STK_01", ans_correct)
    print("\n--- Test Case 1: Correct Answer ---")
    print(res1)
    assert res1["classification"] == "correct"
    assert res1["score"] == 1.0
    assert res1["bkt_outcome"] == 1

    # Test Case 2: Partially Correct Answer (Omission)
    ans_partial = "A stack operates on LIFO where you push to the top and pop from the top."
    res2 = evaluate_answer("Q_STK_01", ans_partial)
    print("\n--- Test Case 2: Partially Correct Answer ---")
    print(res2)
    assert res2["classification"] == "partially_correct"
    assert res2["score"] == 0.5
    assert res2["bkt_outcome"] == 0

    # Test Case 3: Incorrect Answer with Misconception
    ans_misc = "A stack is First-In First-Out (FIFO) like a supermarket queue. Push is O(1) and pop is O(n)."
    res3 = evaluate_answer("Q_STK_01", ans_misc)
    print("\n--- Test Case 3: Misconception Detected ---")
    print(res3)
    assert res3["classification"] == "incorrect"
    assert "Confuses LIFO with FIFO" in res3["identified_misconceptions"]
    assert res3["bkt_outcome"] == 0

    # Test Case 4: Negation / Contrast Case (Should NOT trigger FIFO misconception)
    ans_contrast = "A stack operates on LIFO with O(1) push and pop, unlike a queue which is FIFO."
    res4 = evaluate_answer("Q_STK_01", ans_contrast)
    print("\n--- Test Case 4: Contrast Case (FIFO mentioned in contrast) ---")
    print(res4)
    assert res4["classification"] == "correct"
    assert res4["identified_misconceptions"] == []

    print("\n[SUCCESS] All self-tests passed successfully!")
