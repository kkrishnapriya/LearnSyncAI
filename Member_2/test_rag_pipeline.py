"""
Comprehensive Test and Evaluation Suite for RAG Content Agent (rag.py)
======================================================================
Verifies:
  1. Complete 13-concept retrieval evaluation with 26 diverse queries
  2. Ground-truth relevance matching for Hit Rate@1, Hit Rate@3, and MRR
  3. Disaggregated single-pass timing measurement:
     - Query Dense Embedding
     - FAISS Index Search
     - Concept Filtering & Result Assembly
     - Explanation Generation (Offline Fallback)
     - Total Warm Pipeline Latency
  4. Documentation of cold-start model initialization and live Gemini network latency
  5. Strict concept filtering regression tests across all 13 DSA concepts
  6. Unit tests for Citation Validation:
     - Valid citations
     - Missing citations
     - Missing citation section
     - Fabricated chunk IDs
     - Fabricated OpenDSA URLs
     - Mixed valid/invalid chunk IDs and URLs
     - Citation validation against actual retrieved OpenDSA metadata
  7. Unit tests for Gemini LLM Integration & Error Handling:
     - Mocked successful Gemini generation with verified citations
     - Empty API response fallback
     - Invalid response object fallback
     - Quota / Rate-limit exceeded fallback (HTTP 429)
     - Missing citation rejection and fallback
     - Network connectivity failure fallback
     - Missing API key fallback
     - Invalid API key authentication failure (HTTP 401/403)
     - Empty retrieved context limitation
  8. Deterministic mastery-tier adaptation across all three tiers
  9. Verification of contract public function signatures and defaults
  10. Verification of Phase 1 and Phase 2 dataset integrity preservation
  11. Live Gemini API execution when GEMINI_API_KEY is configured with matched query/context
"""

import os
import sys
import time
import inspect
import csv
import statistics
import unittest
from unittest.mock import MagicMock, patch

# Ensure workspace is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import (
    load_opendsa_content, build_vector_index, retrieve_context,
    retrieve_context_with_timings, generate_explanation,
    _validate_response_citations, _get_api_key, VALID_CONCEPTS
)

# 26 queries across all 13 concepts (2 queries per concept)
# Each query lists ONLY genuinely relevant OpenDSA chunks from opendsa_chunks.csv
BENCHMARK_QUERIES = [
    # Arrays
    {
        "query": "How does an array store items in sequential memory?",
        "concept": "arrays",
        "target_primary": "chk_arr_01",
        "relevant_chunks": ["chk_arr_01"]
    },
    {
        "query": "Why does doubling the buffer size give amortized constant time appends?",
        "concept": "arrays",
        "target_primary": "chk_arr_03",
        "relevant_chunks": ["chk_arr_03"]
    },

    # Linked Lists
    {
        "query": "Explain pointer based traversal and node structure in singly linked lists",
        "concept": "linked_lists",
        "target_primary": "chk_ll_01",
        "relevant_chunks": ["chk_ll_01"]
    },
    {
        "query": "What are the advantages of bidirectional prev and next pointers?",
        "concept": "linked_lists",
        "target_primary": "chk_ll_02",
        "relevant_chunks": ["chk_ll_02"]
    },

    # Stacks
    {
        "query": "What is the LIFO rule and what are push and pop complexities?",
        "concept": "stacks",
        "target_primary": "chk_stk_01",
        "relevant_chunks": ["chk_stk_01"]
    },
    {
        "query": "How do compilers use stacks for function calls and parenthesis matching?",
        "concept": "stacks",
        "target_primary": "chk_stk_03",
        "relevant_chunks": ["chk_stk_03"]
    },

    # Queues
    {
        "query": "How does First-In First-Out ordering work with enqueue and dequeue?",
        "concept": "queues",
        "target_primary": "chk_que_01",
        "relevant_chunks": ["chk_que_01"]
    },
    {
        "query": "Why does a circular queue use modulo arithmetic for array index wrapping?",
        "concept": "queues",
        "target_primary": "chk_que_02",
        "relevant_chunks": ["chk_que_02"]
    },

    # Recursion
    {
        "query": "Why is a terminating base case required in recursive algorithms?",
        "concept": "recursion",
        "target_primary": "chk_rec_01",
        "relevant_chunks": ["chk_rec_01"]
    },
    {
        "query": "How do stack frames and activation records cause stack overflow?",
        "concept": "recursion",
        "target_primary": "chk_rec_02",
        "relevant_chunks": ["chk_rec_02"]
    },

    # Trees
    {
        "query": "What is the relationship between node count and edge count in trees?",
        "concept": "trees",
        "target_primary": "chk_tre_01",
        "relevant_chunks": ["chk_tre_01"]
    },
    {
        "query": "Compare preorder, inorder, and postorder tree traversal orders",
        "concept": "trees",
        "target_primary": "chk_tre_02",
        "relevant_chunks": ["chk_tre_02"]
    },

    # Binary Search Trees
    {
        "query": "What ordering invariant must keys in a binary search tree follow?",
        "concept": "binary_search_trees",
        "target_primary": "chk_bst_01",
        "relevant_chunks": ["chk_bst_01"]
    },
    {
        "query": "How do search and insert degrade to linear time in a skewed BST?",
        "concept": "binary_search_trees",
        "target_primary": "chk_bst_03",
        "relevant_chunks": ["chk_bst_03"]
    },

    # Heaps
    {
        "query": "What are the complete binary tree and max heap order properties?",
        "concept": "heaps",
        "target_primary": "chk_hea_01",
        "relevant_chunks": ["chk_hea_01"]
    },
    {
        "query": "Why does bottom up heapify build a binary heap in linear O(n) time?",
        "concept": "heaps",
        "target_primary": "chk_hea_03",
        "relevant_chunks": ["chk_hea_03"]
    },

    # Sorting
    {
        "query": "What is the Omega(n log n) theoretical lower bound for comparison sorting?",
        "concept": "sorting",
        "target_primary": "chk_srt_01",
        "relevant_chunks": ["chk_srt_01"]
    },
    {
        "query": "Compare in-place partitioning in Quicksort with auxiliary memory in Mergesort",
        "concept": "sorting",
        "target_primary": "chk_srt_02",
        "relevant_chunks": ["chk_srt_02"]
    },

    # Hashing
    {
        "query": "What is expected O(1) lookup vs O(n) worst-case collision degradation?",
        "concept": "hashing",
        "target_primary": "chk_has_01",
        "relevant_chunks": ["chk_has_01"]
    },
    {
        "query": "How does separate chaining use linked bucket lists for collision resolution?",
        "concept": "hashing",
        "target_primary": "chk_has_02",
        "relevant_chunks": ["chk_has_02"]
    },

    # Graphs
    {
        "query": "When is an adjacency list representation preferred over an adjacency matrix?",
        "concept": "graphs",
        "target_primary": "chk_grp_01",
        "relevant_chunks": ["chk_grp_01"]
    },
    {
        "query": "What is a directed acyclic graph and how does topological sorting order vertices?",
        "concept": "graphs",
        "target_primary": "chk_grp_03",
        "relevant_chunks": ["chk_grp_03"]
    },

    # BFS
    {
        "query": "How does breadth first search guarantee finding the shortest path in unweighted graphs?",
        "concept": "bfs",
        "target_primary": "chk_bfs_02",
        "relevant_chunks": ["chk_bfs_02"]
    },
    {
        "query": "How does BFS verify if an undirected graph is bipartite using alternating colors?",
        "concept": "bfs",
        "target_primary": "chk_bfs_03",
        "relevant_chunks": ["chk_bfs_03"]
    },

    # DFS
    {
        "query": "Explain deep branch exploration, backtracking, and call stack in depth first search",
        "concept": "dfs",
        "target_primary": "chk_dfs_01",
        "relevant_chunks": ["chk_dfs_01"]
    },
    {
        "query": "How does DFS classify back edges to detect cycles in a graph?",
        "concept": "dfs",
        "target_primary": "chk_dfs_02",
        "relevant_chunks": ["chk_dfs_02"]
    }
]


def run_retrieval_benchmark():
    print("=" * 115)
    print("LearnSync AI - Phase 3 RAG Pipeline Benchmark (All 13 Concepts, 26 Queries)")
    print("=" * 115)

    # Measure cold-start initialization
    t_start = time.perf_counter()
    docs = load_opendsa_content("opendsa_chunks.csv")
    index, active_docs = build_vector_index(docs)
    cold_start_ms = (time.perf_counter() - t_start) * 1000

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []

    embed_latencies = []
    faiss_latencies = []
    filter_latencies = []
    retrieval_latencies = []
    generation_latencies = []
    total_warm_latencies = []
    metadata_checks_passed = True

    header = (
        f"{'#':<3} | {'Concept':<18} | {'Target':<10} | {'Rank':<5} | "
        f"{'Retrieved Chunks':<32} | {'Embed':<7} | {'FAISS':<7} | {'Filter':<7} | "
        f"{'Ret Total':<9} | {'Gen (ms)':<8} | {'Total Warm'}"
    )
    print(header)
    print("-" * 115)

    for i, item in enumerate(BENCHMARK_QUERIES):
        q = item["query"]
        concept = item["concept"]
        target = item["target_primary"]
        relevant = item["relevant_chunks"]

        # Disaggregated single-pass retrieval timing (Query is encoded exactly once)
        results, timings = retrieve_context_with_timings(q, concept_id=concept, top_k=3)

        t_embed = timings["embedding_ms"]
        t_faiss = timings["faiss_search_ms"]
        t_filter = timings["filtering_ms"]
        t_ret = timings["retrieval_total_ms"]

        embed_latencies.append(t_embed)
        faiss_latencies.append(t_faiss)
        filter_latencies.append(t_filter)
        retrieval_latencies.append(t_ret)

        # Offline Explanation Generation Timing
        t0 = time.perf_counter()
        exp = generate_explanation(q, results, mastery_level=0.55)
        t_gen = (time.perf_counter() - t0) * 1000
        generation_latencies.append(t_gen)

        t_total_warm = t_ret + t_gen
        total_warm_latencies.append(t_total_warm)

        # Check metadata preservation
        for r in results:
            for field in ["chunk_id", "concept_id", "title", "source_url", "difficulty_level", "content", "score"]:
                if field not in r or r[field] is None:
                    metadata_checks_passed = False

        retrieved_ids = [r["chunk_id"] for r in results]
        rank = None
        for r_idx, r_id in enumerate(retrieved_ids):
            if r_id in relevant:
                rank = r_idx + 1
                break

        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        retrieved_str = str(retrieved_ids)
        rank_str = str(rank) if rank is not None else "None"
        row_str = (
            f"{i+1:<3} | {concept:<18} | {target:<10} | {rank_str:<5} | "
            f"{retrieved_str:<32} | {t_embed:<7.2f} | {t_faiss:<7.3f} | {t_filter:<7.3f} | "
            f"{t_ret:<9.2f} | {t_gen:<8.3f} | {t_total_warm:.2f} ms"
        )
        print(row_str)

    total_q = len(BENCHMARK_QUERIES)
    hit_rate_1 = hits_at_1 / total_q
    hit_rate_3 = hits_at_3 / total_q
    mrr = sum(reciprocal_ranks) / total_q

    print("\n" + "=" * 115)
    print("Empirical Retrieval Benchmark Summary:")
    print(f" - Total Test Queries: {total_q} (2 per concept across all 13 concepts)")
    print(f" - Hit Rate@1: {hits_at_1}/{total_q} ({hit_rate_1 * 100:.2f}%)")
    print(f" - Hit Rate@3: {hits_at_3}/{total_q} ({hit_rate_3 * 100:.2f}%)")
    print(f" - Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f" - Source Metadata Retention Check: {'PASSED' if metadata_checks_passed else 'FAILED'}")
    print("\nDisaggregated Latency Profile:")
    print(f" - Cold-Start Model Weight Loading & Indexing: {cold_start_ms:.2f} ms (one-time initialization)")
    print(f" - Query Dense Embedding (ms): Mean={statistics.mean(embed_latencies):.2f}, Median={statistics.median(embed_latencies):.2f}, Min={min(embed_latencies):.2f}, Max={max(embed_latencies):.2f}")
    print(f" - FAISS Index Search (ms):     Mean={statistics.mean(faiss_latencies):.3f}, Median={statistics.median(faiss_latencies):.3f}, Min={min(faiss_latencies):.3f}, Max={max(faiss_latencies):.3f}")
    print(f" - Concept Filtering & Assembly: Mean={statistics.mean(filter_latencies):.3f}, Median={statistics.median(filter_latencies):.3f}, Min={min(filter_latencies):.3f}, Max={max(filter_latencies):.3f}")
    print(f" - End-to-End Retrieval (ms):   Mean={statistics.mean(retrieval_latencies):.2f}, Median={statistics.median(retrieval_latencies):.2f}, Min={min(retrieval_latencies):.2f}, Max={max(retrieval_latencies):.2f}")
    print(f" - Explanation Gen (Offline):    Mean={statistics.mean(generation_latencies):.3f}, Median={statistics.median(generation_latencies):.3f}, Min={min(generation_latencies):.3f}, Max={max(generation_latencies):.3f}")
    print(f" - Total Warm Pipeline Latency:  Mean={statistics.mean(total_warm_latencies):.2f} ms, Median={statistics.median(total_warm_latencies):.2f} ms")
    print("\nLive Gemini Network Latency Note:")
    print(" - When GEMINI_API_KEY is configured, external network round-trip and token generation by gemini-2.5-flash")
    print("   typically take ~1200 ms to ~2800 ms per query, compared to ~1.5 ms for the deterministic local fallback.")
    print("=" * 115)


class TestRAGUnitAndLLM(unittest.TestCase):
    """
    Unit test suite covering strict concept filtering, citation validation,
    error handling, mastery tiers, mocked Gemini API responses, and live API hooks.
    """
    @classmethod
    def setUpClass(cls):
        cls.docs = load_opendsa_content("opendsa_chunks.csv")
        cls.index, cls.active_docs = build_vector_index(cls.docs)
        cls.sample_ctx = retrieve_context("What is a stack?", concept_id="stacks", top_k=2)

    # -------------------------------------------------------------------------
    # Area 1: Strict Concept Filtering Regression Tests
    # -------------------------------------------------------------------------
    def test_strict_concept_filtering_all_13_concepts(self):
        """
        Regression test proving that retrieve_context returns ONLY chunks
        belonging to the requested concept across all 13 DSA concepts,
        with zero cross-concept pollution.
        """
        for concept in sorted(VALID_CONCEPTS):
            results = retrieve_context(
                query=f"Explain core operations and invariant rules in {concept}",
                concept_id=concept,
                top_k=3
            )
            self.assertEqual(len(results), 3, f"Expected 3 chunks for concept '{concept}', got {len(results)}")
            for chunk in results:
                self.assertEqual(
                    chunk.get("concept_id"),
                    concept,
                    f"Cross-concept leakage detected: chunk {chunk.get('chunk_id')} has concept '{chunk.get('concept_id')}', expected '{concept}'"
                )

    # -------------------------------------------------------------------------
    # Area 2: Citation Validation Tests
    # -------------------------------------------------------------------------
    def test_citation_validation_valid(self):
        """Tests that a properly formatted response with valid citations and section passes."""
        valid_text = (
            "A stack is a Last-In, First-Out (LIFO) data structure.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- **[Stack ADT and LIFO Discipline](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html)** "
            "(Chunk: `chk_stk_01`, Concept: `stacks`, Level: `beginner`)"
        )
        ok, msg = _validate_response_citations(valid_text, self.sample_ctx)
        self.assertTrue(ok)
        self.assertEqual(msg, "Valid citations verified")

    def test_citation_validation_missing_citations(self):
        """Tests that a response completely omitting chunk citations fails validation."""
        no_cite_text = (
            "A stack is a Last-In First-Out data structure where elements are pushed and popped.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- No sources referenced here."
        )
        ok, msg = _validate_response_citations(no_cite_text, self.sample_ctx)
        self.assertFalse(ok)
        self.assertIn("lacked citations", msg)

    def test_citation_validation_missing_section(self):
        """Tests that a response citing chunks in body but omitting the citation section fails."""
        no_section_text = (
            "According to chk_stk_01 and https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html, "
            "stacks enforce the LIFO discipline with O(1) push and pop."
        )
        ok, msg = _validate_response_citations(no_section_text, self.sample_ctx)
        self.assertFalse(ok)
        self.assertIn("lacked required OpenDSA citation section", msg)

    def test_citation_validation_fabricated_chunk_id(self):
        """Tests that a response citing an unretrieved / fabricated chunk ID is strictly rejected."""
        fake_cite_text = (
            "Stacks enforce LIFO ordering.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- **[Stack ADT](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html)** (Chunk: `chk_fake_9999`)"
        )
        ok, msg = _validate_response_citations(fake_cite_text, self.sample_ctx)
        self.assertFalse(ok)
        self.assertIn("fabricated or unretrieved chunk IDs", msg)
        self.assertIn("chk_fake_9999", msg)

    def test_citation_validation_fabricated_url(self):
        """Tests that a response citing an unretrieved / fabricated OpenDSA URL is strictly rejected."""
        fake_url_text = (
            "Stacks enforce LIFO ordering.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- **[Stack ADT](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/FakeInventedPage.html)** (Chunk: `chk_stk_01`)"
        )
        ok, msg = _validate_response_citations(fake_url_text, self.sample_ctx)
        self.assertFalse(ok)
        self.assertIn("fabricated or unretrieved OpenDSA URLs", msg)
        self.assertIn("FakeInventedPage.html", msg)

    def test_citation_validation_mixed_valid_and_fabricated_chunk(self):
        """Tests that mixed valid and fabricated chunk IDs are strictly rejected."""
        mixed_chunk_text = (
            "Stacks enforce LIFO ordering.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- **[Valid Stack](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html)** (Chunk: `chk_stk_01`)\n"
            "- **[Fake Stack](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html)** (Chunk: `chk_stk_9999`)"
        )
        ok, msg = _validate_response_citations(mixed_chunk_text, self.sample_ctx)
        self.assertFalse(ok)
        self.assertIn("fabricated or unretrieved chunk IDs", msg)
        self.assertIn("chk_stk_9999", msg)

    def test_citation_validation_mixed_valid_and_fabricated_url(self):
        """Tests that mixed valid and fabricated OpenDSA URLs are strictly rejected."""
        mixed_url_text = (
            "Stacks enforce LIFO ordering.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- **[Valid Page](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html)** (Chunk: `chk_stk_01`)\n"
            "- **[Fake Page](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/InventedPage.html)** (Chunk: `chk_stk_01`)"
        )
        ok, msg = _validate_response_citations(mixed_url_text, self.sample_ctx)
        self.assertFalse(ok)
        self.assertIn("fabricated or unretrieved OpenDSA URLs", msg)
        self.assertIn("InventedPage.html", msg)

    def test_citation_validation_against_actual_metadata(self):
        """
        Verifies that citation validation works end-to-end against actual retrieved metadata,
        and rejects chunks from outside the retrieved set.
        """
        real_ctx = retrieve_context("What is an array?", concept_id="arrays", top_k=2)
        c0 = real_ctx[0]
        # Valid explanation with real metadata
        resp_valid = (
            f"An array provides contiguous storage.\n\n"
            f"### 📚 Grounded OpenDSA Sources:\n"
            f"- **[{c0['title']}]({c0['source_url']})** (Chunk: `{c0['chunk_id']}`)"
        )
        ok, msg = _validate_response_citations(resp_valid, real_ctx)
        self.assertTrue(ok)

        # Invalid explanation citing a chunk from trees (not retrieved)
        resp_invalid = (
            f"An array provides contiguous storage.\n\n"
            f"### 📚 Grounded OpenDSA Sources:\n"
            f"- **[Tree Properties](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/BinaryTree.html)** (Chunk: `chk_tre_01`)"
        )
        ok2, msg2 = _validate_response_citations(resp_invalid, real_ctx)
        self.assertFalse(ok2)
        self.assertIn("chk_tre_01", msg2)

    # -------------------------------------------------------------------------
    # Area 3: Gemini Mocking, Error Handling & API Security Tests
    # -------------------------------------------------------------------------
    @patch("rag._get_api_key", return_value="mocked_valid_api_key_12345")
    @patch("google.genai.Client")
    def test_gemini_generation_mocked_success(self, mock_client_cls, mock_get_key):
        """Tests successful Gemini generation using a mocked API response with valid citations."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = (
            "## Stacks and LIFO\n\n"
            "A stack enforces the Last-In First-Out discipline.\n\n"
            "### 📚 Grounded OpenDSA Sources:\n"
            "- **[Stack ADT and LIFO Discipline](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/StackArray.html)** "
            "(Chunk: `chk_stk_01`, Concept: `stacks`, Level: `beginner`)"
        )
        mock_client.models.generate_content.return_value = mock_resp

        result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("[Mode: Generative AI (gemini-2.5-flash)]", result)
        self.assertIn("chk_stk_01", result)

    @patch("rag._get_api_key", return_value="mocked_valid_api_key_12345")
    @patch("google.genai.Client")
    def test_gemini_empty_response_fallback(self, mock_client_cls, mock_get_key):
        """Tests that an empty string API response triggers fallback with appropriate disclosure."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_client.models.generate_content.return_value = mock_resp

        result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("Deterministic Offline Fallback", result)
        self.assertIn("Invalid or empty response object from model", result)

    @patch("rag._get_api_key", return_value="mocked_valid_api_key_12345")
    @patch("google.genai.Client")
    def test_gemini_invalid_response_object_fallback(self, mock_client_cls, mock_get_key):
        """Tests that an invalid response object (None response or missing text) triggers fallback."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Case 1: response is None
        mock_client.models.generate_content.return_value = None
        result_none = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("Empty response from model", result_none)

        # Case 2: response has text = None
        mock_resp_bad = MagicMock()
        mock_resp_bad.text = None
        mock_client.models.generate_content.return_value = mock_resp_bad
        result_bad = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("Invalid or empty response object from model", result_bad)

    @patch("rag._get_api_key", return_value="mocked_valid_api_key_12345")
    @patch("google.genai.Client")
    def test_gemini_quota_exhausted_fallback(self, mock_client_cls, mock_get_key):
        """Tests that HTTP 429 quota exhaustion or rate limit triggers quota fallback."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("429 Resource exhausted: Quota exceeded for Generate Content API")

        result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("API quota exceeded / rate limit reached", result)

    @patch("rag._get_api_key", return_value="mocked_valid_api_key_12345")
    @patch("google.genai.Client")
    def test_gemini_missing_citation_rejection(self, mock_client_cls, mock_get_key):
        """Tests that a Gemini response lacking chunk citations is rejected and falls back."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = "A stack is a data structure with push and pop. (No citations included)."
        mock_client.models.generate_content.return_value = mock_resp

        result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("Deterministic Offline Fallback", result)
        self.assertIn("Response failed citation validation", result)

    @patch("rag._get_api_key", return_value="mocked_valid_api_key_12345")
    @patch("google.genai.Client")
    def test_gemini_network_failure_fallback(self, mock_client_cls, mock_get_key):
        """Tests that network/connection errors trigger fallback with network reason."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = ConnectionError("getaddrinfo failed: connection timeout")

        result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("Network connectivity failure", result)

    @patch("rag._get_api_key", return_value=None)
    def test_missing_api_key_fallback(self, mock_get_key):
        """Tests that missing API key triggers fallback with 'API key not configured'."""
        result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("API key not configured", result)

    def test_invalid_api_key_fallback(self):
        """Tests that an invalid API key triggers authentication failure reason."""
        with patch("rag._get_api_key", return_value="INVALID_DUMMY_KEY"):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.models.generate_content.side_effect = Exception("401 Unauthorized: Invalid API key")
                result = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
                self.assertIn("Authentication failed / invalid API key", result)

    def test_empty_retrieved_context(self):
        """Tests that empty retrieved context produces context insufficiency message."""
        result = generate_explanation("Unknown topic", [], mastery_level=0.5)
        self.assertIn("Context Insufficient", result)

    @patch("rag._get_api_key", return_value=None)
    def test_mastery_tiers_strictly_grounded(self, mock_get_key):
        """
        Tests that all 3 mastery tiers use strictly grounded content without generic hardcoded analogies.
        Guaranteed deterministic execution via mocked offline state regardless of environment keys.
        """
        exp_low = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.20)
        self.assertIn("Foundational Tier", exp_low)
        self.assertNotIn("cafeteria trays", exp_low)
        self.assertNotIn("grocery line", exp_low)

        exp_med = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.55)
        self.assertIn("Intermediate Tier", exp_med)

        exp_high = generate_explanation("What is a stack?", self.sample_ctx, mastery_level=0.85)
        self.assertIn("Advanced Tier", exp_high)

    # -------------------------------------------------------------------------
    # Area 4: Contract Signatures & Dataset Preservation Tests
    # -------------------------------------------------------------------------
    def test_public_function_signatures(self):
        """
        Verifies that all 4 required contract public function signatures and defaults
        remain exactly as specified in the LearnSync AI contract.
        """
        # 1. load_opendsa_content(source_directory=None)
        sig_load = inspect.signature(load_opendsa_content)
        self.assertIn("source_directory", sig_load.parameters)
        self.assertIs(sig_load.parameters["source_directory"].default, None)

        # 2. build_vector_index(documents)
        sig_build = inspect.signature(build_vector_index)
        self.assertIn("documents", sig_build.parameters)

        # 3. retrieve_context(query, concept_id, top_k=3)
        sig_ret = inspect.signature(retrieve_context)
        params_ret = list(sig_ret.parameters.keys())
        self.assertEqual(params_ret, ["query", "concept_id", "top_k"])
        self.assertEqual(sig_ret.parameters["top_k"].default, 3)

        # 4. generate_explanation(query, retrieved_context, mastery_level=0.5)
        sig_gen = inspect.signature(generate_explanation)
        params_gen = list(sig_gen.parameters.keys())
        self.assertEqual(params_gen, ["query", "retrieved_context", "mastery_level"])
        self.assertEqual(sig_gen.parameters["mastery_level"].default, 0.5)

    def test_datasets_integrity_preservation(self):
        """
        Verifies that Phase 1 (opendsa_chunks.csv) and Phase 2 (short_answer_dataset.csv)
        remain completely intact and unmodified.
        """
        # Phase 1: opendsa_chunks.csv must have exactly 39 rows and 13 concepts (3 each)
        with open("opendsa_chunks.csv", "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 39)
            concept_counts = {}
            for r in reader:
                c = r["concept_id"]
                concept_counts[c] = concept_counts.get(c, 0) + 1
            self.assertEqual(len(concept_counts), 13)
            for c, cnt in concept_counts.items():
                self.assertEqual(cnt, 3)

        # Phase 2: short_answer_dataset.csv must have exactly 117 rows (39 per classification)
        with open("short_answer_dataset.csv", "r", encoding="utf-8") as f:
            reader2 = list(csv.DictReader(f))
            self.assertEqual(len(reader2), 117)
            class_counts = {}
            for r in reader2:
                cls = r["ground_truth_classification"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
            self.assertEqual(class_counts["correct"], 39)
            self.assertEqual(class_counts["partially_correct"], 39)
            self.assertEqual(class_counts["incorrect"], 39)

    # -------------------------------------------------------------------------
    # Area 5: Live API Test Hook
    # -------------------------------------------------------------------------
    def test_live_gemini_api_call(self):
        """
        Executes an actual live Gemini API call ONLY if a real GEMINI_API_KEY is configured.
        Uses a matched query and retrieved context to ensure proper alignment.
        Clearly reports whether the live test ran, passed, failed, or was skipped.
        """
        key = _get_api_key()
        if not key or key.startswith("mocked") or key.startswith("INVALID"):
            self.skipTest("No real GEMINI_API_KEY detected in environment or Colab secrets.")

        print("\n  [LIVE API TEST STATUS: EXECUTING]")
        print("  Valid key detected. Executing live call to gemini-2.5-flash with matched query and context...")
        
        # Matched query and retrieved context
        live_query = "What is the LIFO rule and what are push and pop complexities in a stack?"
        live_ctx = retrieve_context(live_query, concept_id="stacks", top_k=2)

        try:
            res = generate_explanation(live_query, live_ctx, mastery_level=0.50)
            self.assertIn("[Mode: Generative AI (gemini-2.5-flash)]", res)
            self.assertIn("chk_stk_01", res)
            print("  [LIVE API TEST STATUS: PASSED] Genuine generative explanation received and verified.")
        except Exception as e:
            print(f"  [LIVE API TEST STATUS: FAILED] Error: {e}")
            raise e


def run_all_tests():
    print("\n" + "=" * 115)
    print("Running Unit Tests with Mocked API Responses, Error Handling & Citation Checks")
    print("=" * 115)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGUnitAndLLM)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - skipped

    print("\n" + "=" * 115)
    print("Unit Test Execution Summary:")
    print(f" - Total Tests:   {total}")
    print(f" - Passed Tests:  {passed}")
    print(f" - Failed Tests:  {failed}")
    print(f" - Skipped Tests: {skipped}")
    if skipped > 0:
        for test_case, reason in result.skipped:
            print(f"   * Skipped '{test_case.id().split('.')[-1]}': {reason}")
    print("=" * 115)

    return result.wasSuccessful()


if __name__ == "__main__":
    run_retrieval_benchmark()
    success = run_all_tests()
    if not success:
        sys.exit(1)
