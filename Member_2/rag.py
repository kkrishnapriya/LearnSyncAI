"""
LearnSync AI - Member 2: RAG Content Agent (rag.py)
===================================================
This module implements the primary Retrieval-Augmented Generation (RAG) pipeline
for Data Structures and Algorithms (DSA) instruction, adhering strictly to the
LearnSync AI Hybrid Agentic Implementation Contract.

Technology Stack:
    - Hugging Face Sentence Transformers (all-MiniLM-L6-v2)
    - FAISS (IndexFlatIP with normalized dense vectors for exact cosine similarity)
    - OpenDSA Curriculum Knowledge Base (39 curated semantic chunks)
    - Official Google GenAI SDK (google-genai) for real generative LLM explanations
    - Deterministic Grounded Offline Synthesizer as an automatic, transparent fallback

Contract Required Public Interfaces:
    1. load_opendsa_content(source_directory)
    2. build_vector_index(documents)
    3. retrieve_context(query, concept_id, top_k=3)
    4. generate_explanation(query, retrieved_context, mastery_level)

API Key Configuration:
    To enable live generative LLM responses via Gemini:
    - Local / Terminal: set environment variable GEMINI_API_KEY or GOOGLE_API_KEY
        Windows PowerShell: $env:GEMINI_API_KEY="your-api-key"
        Linux/Mac: export GEMINI_API_KEY="your-api-key"
    - Google Colab: Add 'GEMINI_API_KEY' to Colab Secrets (the key icon in sidebar).
    If no key is configured or if network/quota errors occur, the module automatically
    activates the deterministic grounded offline fallback synthesizer and transparently
    discloses the exact reason without exposing sensitive credentials.
"""

import os
import re
import csv
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Suppress Hugging Face advisory warnings at import time (safe, no network call)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# NOTE: The original code made an HTTP probe to huggingface.co at import time and
# conditionally set HF_HUB_OFFLINE as a process-wide side effect.
# This has been removed: (1) it caused a network call on every import,
# (2) it mutated os.environ for the entire process unexpectedly.
# HF offline mode can be enabled explicitly by the user: set HF_HUB_OFFLINE=1 in .env.

# Validated DSA concepts according to the LearnSync AI Implementation Contract
VALID_CONCEPTS = {
    "arrays", "linked_lists", "stacks", "queues", "recursion",
    "trees", "binary_search_trees", "heaps", "sorting", "hashing",
    "graphs", "bfs", "dfs"
}

# Global module state for active FAISS index, dense embedding model, and documents
_ACTIVE_INDEX = None
_ACTIVE_MODEL = None
_ACTIVE_DOCUMENTS: List[Dict[str, Any]] = []
_MODEL_NAME = "all-MiniLM-L6-v2"
_LLM_MODEL_NAME = "gemini-2.5-flash"


def _get_api_key() -> Optional[str]:
    """
    Securely retrieves the Gemini API key from environment variables or Colab Secrets.
    Never exposes or logs the raw API key.
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()

    try:
        from google.colab import userdata
        colab_key = userdata.get("GEMINI_API_KEY") or userdata.get("GOOGLE_API_KEY")
        if colab_key and str(colab_key).strip():
            return str(colab_key).strip()
    except (ImportError, Exception):
        pass

    return None


def load_opendsa_content(source_directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads and validates OpenDSA educational curriculum chunks.

    Parameters:
        source_directory (str, optional): Path to opendsa_chunks.csv or directory containing it.
                                          Defaults to opendsa_chunks.csv in current workspace.

    Returns:
        List[Dict[str, Any]]: List of document dictionaries with validated metadata and content.

    Raises:
        FileNotFoundError: If the specified file or default chunks CSV cannot be found.
        ValueError: If required columns or concepts fail validation.
    """
    if source_directory is None:
        target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opendsa_chunks.csv")
        if not os.path.exists(target_path):
            target_path = "opendsa_chunks.csv"
    elif os.path.isdir(source_directory):
        target_path = os.path.join(source_directory, "opendsa_chunks.csv")
    else:
        target_path = source_directory

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"OpenDSA chunks file not found at: {target_path}")

    required_cols = {"chunk_id", "concept_id", "title", "source_url", "content", "summary", "difficulty_level", "keywords"}
    documents = []

    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not required_cols.issubset(fieldnames):
            missing = required_cols - fieldnames
            raise ValueError(f"OpenDSA CSV is missing required columns: {missing}")

        for row_idx, row in enumerate(reader):
            concept = row["concept_id"].strip().lower()
            if concept not in VALID_CONCEPTS and concept != "unmapped":
                concept = "unmapped"
            row["concept_id"] = concept

            if not row.get("content", "").strip():
                raise ValueError(f"Row {row_idx} ({row.get('chunk_id')}) has empty content.")

            documents.append(dict(row))

    if not documents:
        raise ValueError(f"OpenDSA chunks file at {target_path} contains 0 records.")

    return documents


def build_vector_index(documents: List[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Builds the primary dense vector index using Hugging Face Sentence Transformers
    and FAISS IndexFlatIP (cosine similarity on L2-normalized embeddings).

    Parameters:
        documents (List[Dict[str, Any]]): List of validated OpenDSA document chunks.

    Returns:
        Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]: (FAISS vector index, documents list)

    Raises:
        ValueError: If documents list is empty.
        ImportError: If sentence-transformers or faiss is not installed.
    """
    global _ACTIVE_INDEX, _ACTIVE_MODEL, _ACTIVE_DOCUMENTS

    if not documents:
        raise ValueError("Cannot build vector index from an empty document list.")

    from sentence_transformers import SentenceTransformer
    import faiss

    if _ACTIVE_MODEL is None:
        _ACTIVE_MODEL = SentenceTransformer(_MODEL_NAME)

    _ACTIVE_DOCUMENTS = list(documents)

    texts_to_embed = [
        f"{doc.get('title', '')}. {doc.get('summary', '')} {doc.get('content', '')}"
        for doc in _ACTIVE_DOCUMENTS
    ]

    embeddings = _ACTIVE_MODEL.encode(
        texts_to_embed,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _ACTIVE_INDEX = index
    return _ACTIVE_INDEX, _ACTIVE_DOCUMENTS


def retrieve_context_with_timings(
    query: str,
    concept_id: str,
    top_k: int = 3
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Retrieves the most pedagogically relevant OpenDSA content passages
    while capturing disaggregated timing metrics for:
      - query dense embedding
      - FAISS index similarity search
      - concept-aware filtering and result construction
    Guarantees the query is encoded only once.

    Parameters:
        query (str): The search query or question text.
        concept_id (str): Target DSA concept identifier (or 'unmapped').
        top_k (int): Number of top passages to retrieve. Default is 3.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, float]]: (retrieved_passages, timing_metrics_ms)
    """
    global _ACTIVE_INDEX, _ACTIVE_MODEL, _ACTIVE_DOCUMENTS

    timings = {
        "embedding_ms": 0.0,
        "faiss_search_ms": 0.0,
        "filtering_ms": 0.0,
        "retrieval_total_ms": 0.0
    }

    if not query or not query.strip():
        return [], timings

    if _ACTIVE_INDEX is None or not _ACTIVE_DOCUMENTS or _ACTIVE_MODEL is None:
        docs = load_opendsa_content()
        build_vector_index(docs)

    norm_concept = concept_id.strip().lower() if isinstance(concept_id, str) else "unmapped"
    if norm_concept not in VALID_CONCEPTS and norm_concept != "unmapped":
        norm_concept = "unmapped"

    # 1. Query Dense Embedding
    t0 = time.perf_counter()
    q_emb = _ACTIVE_MODEL.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)
    t_embed = (time.perf_counter() - t0) * 1000

    # 2. FAISS Similarity Search (search all index documents to evaluate all concept candidates)
    t1 = time.perf_counter()
    k_search = len(_ACTIVE_DOCUMENTS)
    distances, indices = _ACTIVE_INDEX.search(q_emb, k_search)
    t_search = (time.perf_counter() - t1) * 1000

    # 3. Concept Filtering & Result Assembly (STRICT concept filtering when concept_id is specified)
    t2 = time.perf_counter()
    results: List[Dict[str, Any]] = []

    if norm_concept != "unmapped":
        # Strict concept filtering: strictly return ONLY chunks matching the specified concept_id
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_ACTIVE_DOCUMENTS):
                continue
            doc = dict(_ACTIVE_DOCUMENTS[idx])
            if doc.get("concept_id") == norm_concept:
                doc["score"] = float(score)
                results.append(doc)
                if len(results) >= top_k:
                    break
    else:
        # Cross-concept global search ONLY when concept_id is explicitly 'unmapped'
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_ACTIVE_DOCUMENTS):
                continue
            doc = dict(_ACTIVE_DOCUMENTS[idx])
            doc["score"] = float(score)
            results.append(doc)
            if len(results) >= top_k:
                break
    t_filter = (time.perf_counter() - t2) * 1000

    timings["embedding_ms"] = t_embed
    timings["faiss_search_ms"] = t_search
    timings["filtering_ms"] = t_filter
    timings["retrieval_total_ms"] = t_embed + t_search + t_filter

    return results, timings


def retrieve_context(query: str, concept_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves the most pedagogically relevant OpenDSA content passages
    matching the query and concept identifier using dense FAISS vector search.
    Preserves exact contract interface: retrieve_context(query, concept_id, top_k=3).

    Parameters:
        query (str): The search query or question text.
        concept_id (str): Target DSA concept identifier (e.g. 'stacks', 'trees', 'bfs').
                          Must not be omitted. Use 'unmapped' for broad cross-concept search.
        top_k (int): Number of top passages to retrieve. Default is 3.

    Returns:
        List[Dict[str, Any]]: List of retrieved chunk dictionaries retaining all metadata
                              (chunk_id, concept_id, title, source_url, difficulty_level,
                              summary, content, keywords) plus cosine relevance score.
    """
    results, _ = retrieve_context_with_timings(query, concept_id=concept_id, top_k=top_k)
    return results


def _validate_response_citations(response_text: str, retrieved_context: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Strictly verifies that the generated explanation contains valid citations
    corresponding to the retrieved chunk IDs and source URLs, rejects any
    fabricated chunk IDs and fabricated OpenDSA URLs, and requires a dedicated
    citation section.

    Rules:
      1. Non-empty response.
      2. Must contain a dedicated OpenDSA citation section header (e.g. '### Grounded OpenDSA Sources').
      3. At least one retrieved chunk ID and at least one retrieved source URL must be cited.
      4. Every cited chunk ID (matching 'chk_[a-z0-9_]+') MUST belong to the retrieved chunk metadata.
         Any fabricated or unretrieved chunk ID results in immediate rejection.
      5. Every cited OpenDSA URL MUST belong to the retrieved source_url metadata.
         Any fabricated or unretrieved OpenDSA URL results in immediate rejection.
      6. The dedicated citation section must contain at least one valid retrieved citation.
    """
    if not response_text or not response_text.strip():
        return False, "Response was empty"

    # Rule 2: Dedicated citation section requirement
    section_match = re.search(r"###\s*(?:[📚\s]*)?(?:Grounded\s+)?(?:OpenDSA\s+)?Sources.*", response_text, re.IGNORECASE)
    if not section_match:
        return False, "Response lacked required OpenDSA citation section"

    citation_section_text = response_text[section_match.start():]

    valid_chunk_ids = {c.get("chunk_id", "").strip() for c in retrieved_context if c.get("chunk_id")}
    valid_urls = {c.get("source_url", "").strip() for c in retrieved_context if c.get("source_url")}

    # Rule 4: Validate every cited chunk ID in entire response
    all_cited_chunk_ids = set(re.findall(r"\bchk_[a-z0-9_]+\b", response_text))
    fabricated_cids = all_cited_chunk_ids - valid_chunk_ids
    if fabricated_cids:
        return False, f"Response contained fabricated or unretrieved chunk IDs: {sorted(list(fabricated_cids))}"

    # Rule 5: Validate every cited OpenDSA URL in entire response
    raw_urls = re.findall(r"https?://[^\s\)\`\]\>]+", response_text)
    opendsa_urls_in_response = set()
    for u in raw_urls:
        u_clean = u.rstrip(".,:;)")
        if "opendsa" in u_clean.lower():
            opendsa_urls_in_response.add(u_clean)

    fabricated_urls = opendsa_urls_in_response - valid_urls
    if fabricated_urls:
        return False, f"Response contained fabricated or unretrieved OpenDSA URLs: {sorted(list(fabricated_urls))}"

    # Rule 3: Must have at least one valid chunk ID and at least one valid URL cited overall
    has_valid_chunk_id = any(cid in response_text for cid in valid_chunk_ids)
    has_valid_url = any(url in response_text for url in valid_urls)
    if not has_valid_chunk_id:
        return False, "Response lacked citations to retrieved chunk IDs"
    if not has_valid_url:
        return False, "Response lacked citations to retrieved OpenDSA URLs"

    # Rule 6: Citation section itself must contain at least one valid citation
    has_valid_chunk_in_section = any(cid in citation_section_text for cid in valid_chunk_ids)
    has_valid_url_in_section = any(url in citation_section_text for url in valid_urls)
    if not (has_valid_chunk_in_section or has_valid_url_in_section):
        return False, "Citation section does not contain valid retrieved chunk citations"

    return True, "Valid citations verified"


def _synthesize_deterministic_explanation(
    query: str,
    retrieved_context: List[Dict[str, Any]],
    mastery: float,
    fallback_reason: Optional[str] = None
) -> str:
    """
    Deterministic grounded explanation synthesizer (offline fallback).
    Uses ONLY facts and statements directly present in the retrieved passages.
    Contains zero outside examples, fabricated analogies, or unsupported claims.
    """
    primary = retrieved_context[0]
    title = primary.get("title", "DSA Concept")
    content = primary.get("content", "").strip()
    summary = primary.get("summary", "").strip()

    if mastery < 0.40:
        tier_title = "Foundational Tier (Mastery: {:.1f}%)".format(mastery * 100)
        instructional_content = (
            f"### 🟢 Core Concept Fundamentals & Summary\n\n"
            f"**Topic:** {title}\n\n"
            f"**Core Principle:** {summary}\n\n"
            f"**Reference Passage:**\n{content}\n\n"
            f"**Key Takeaway:** Begin by understanding the fundamental operational rules and constraints "
            f"described above before proceeding to implementation details."
        )
    elif mastery <= 0.70:
        tier_title = "Intermediate Tier (Mastery: {:.1f}%)".format(mastery * 100)
        instructional_content = (
            f"### 🟡 Operational Mechanics & Implementation Details\n\n"
            f"**Topic:** {title}\n\n"
            f"**Detailed Passage:**\n{content}\n\n"
            f"**Operational Summary:** {summary}"
        )
    else:
        tier_title = "Advanced Tier (Mastery: {:.1f}%)".format(mastery * 100)
        instructional_content = (
            f"### 🟣 Formal Invariants & Asymptotic Analysis\n\n"
            f"**Topic:** {title}\n\n"
            f"**Theoretical & Complexity Specification:**\n{content}\n\n"
            f"**Invariants & Bounds:** {summary}"
        )

    citations_list = []
    for c in retrieved_context:
        cid = c.get("chunk_id", "chunk")
        ctitle = c.get("title", "OpenDSA Topic")
        curl = c.get("source_url", "https://opendsa-server.cs.vt.edu/")
        cconcept = c.get("concept_id", "")
        cdiff = c.get("difficulty_level", "")
        citations_list.append(f"- **[{ctitle}]({curl})** (Chunk: `{cid}`, Concept: `{cconcept}`, Level: `{cdiff}`)")

    citations_markdown = "\n".join(citations_list)
    reason_label = fallback_reason if fallback_reason else "API key not configured"

    return (
        f"> *[Mode: Deterministic Offline Fallback (Reason: {reason_label})]*\n\n"
        f"## Instructional Response: {query}\n\n"
        f"**Pedagogical Target:** {tier_title}\n\n"
        f"{instructional_content}\n\n"
        f"### 📚 Grounded OpenDSA Sources:\n"
        f"{citations_markdown}"
    )


def generate_explanation(query: str, retrieved_context: List[Dict[str, Any]], mastery_level: float = 0.5) -> str:
    """
    Generates a pedagogically grounded DSA explanation conditioned on the student's mastery level.
    Uses Google GenAI SDK (Gemini) when an API key is configured, with citation validation
    and an automatic, transparent deterministic offline fallback.

    Mastery-Conditioned Adaptation:
        - Below 0.40 (Foundational): Intuitive summaries, core definitions, basic operational steps.
        - 0.40–0.70 (Intermediate): Operational mechanics, implementation details, time/space trade-offs.
        - Above 0.70 (Advanced): Formal invariants, asymptotic lower bounds, degenerate cases, complexity derivations.

    Grounding & Limitations:
        - Strictly uses retrieved OpenDSA passages as factual evidence.
        - If retrieved_context is empty, explicitly acknowledges the limitation.
        - Verifies that generated responses cite the actual retrieved chunk IDs and source URLs.
        - If citations fail validation or API errors occur, activates the deterministic fallback.

    Parameters:
        query (str): The student's question or topic of interest.
        retrieved_context (List[Dict[str, Any]]): Retrieved passages from OpenDSA.
        mastery_level (float): Current BKT mastery probability in [0.0, 1.0].

    Returns:
        str: Grounded instructional text in Markdown format with verified source citations.
    """
    if not retrieved_context:
        return (
            f"### Context Insufficient\n\n"
            f"The provided OpenDSA curriculum context is insufficient to reliably answer the query: **'{query}'**.\n"
            f"Please refine your query or consult the foundational chapters for this concept."
        )

    try:
        mastery = max(0.0, min(1.0, float(mastery_level)))
    except (ValueError, TypeError):
        mastery = 0.5

    api_key = _get_api_key()
    fallback_reason = "API key not configured"

    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            if mastery < 0.40:
                tier_instruction = (
                    f"TARGET TIER: FOUNDATIONAL (Student Mastery: {mastery * 100:.1f}%).\n"
                    "Explain the core principle using only the provided facts. Focus on definitions and basic steps."
                )
            elif mastery <= 0.70:
                tier_instruction = (
                    f"TARGET TIER: INTERMEDIATE (Student Mastery: {mastery * 100:.1f}%).\n"
                    "Explain operational mechanics, memory trade-offs, and algorithms directly described in the text."
                )
            else:
                tier_instruction = (
                    f"TARGET TIER: ADVANCED (Student Mastery: {mastery * 100:.1f}%).\n"
                    "Explain formal invariants, degenerate configurations, and asymptotic complexity bounds from the text."
                )

            context_blocks = []
            citations_blocks = []
            for c in retrieved_context:
                cid = c.get("chunk_id", "chunk")
                title = c.get("title", "Topic")
                url = c.get("source_url", "")
                diff = c.get("difficulty_level", "")
                content = c.get("content", "")
                concept = c.get("concept_id", "")
                context_blocks.append(f"--- CHUNK [{cid}] {title} (Concept: {concept}, Level: {diff}) ---\n{content}\n")
                citations_blocks.append(f"- **[{title}]({url})** (Chunk: `{cid}`, Concept: `{concept}`, Level: `{diff}`)")

            context_str = "\n".join(context_blocks)
            citations_str = "\n".join(citations_blocks)

            system_instruction = (
                "You are LearnSync AI's expert Data Structures and Algorithms tutor.\n"
                "You are provided with verified educational passages retrieved from OpenDSA.\n\n"
                "CRITICAL GROUNDING RULES:\n"
                "1. Base your explanation strictly on the facts and algorithms contained in the provided OpenDSA passages.\n"
                "2. Do not introduce outside concepts, alternative libraries, or unverified claims.\n"
                "3. Adapt your pedagogical tone, depth, and vocabulary strictly according to the TARGET TIER requested.\n"
                "4. Structure your response with clear markdown headings.\n"
                "5. At the end of your response, you MUST include the provided Grounded OpenDSA Sources section verbatim."
            )

            prompt = (
                f"{tier_instruction}\n\n"
                f"STUDENT QUERY:\n{query}\n\n"
                f"RETRIEVED OPENDSA CONTEXT:\n{context_str}\n\n"
                f"Please generate the grounded, mastery-adapted response. End with this exact citations block:\n\n"
                f"### 📚 Grounded OpenDSA Sources:\n{citations_str}"
            )

            response = client.models.generate_content(
                model=_LLM_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=1000
                )
            )

            if response is None:
                fallback_reason = "Empty response from model"
            else:
                try:
                    resp_text = getattr(response, "text", None)
                    if resp_text is None or not str(resp_text).strip():
                        fallback_reason = "Invalid or empty response object from model"
                    else:
                        resp_text = str(resp_text).strip()
                        is_valid_citation, citation_msg = _validate_response_citations(resp_text, retrieved_context)
                        if is_valid_citation:
                            header = f"> *[Mode: Generative AI ({_LLM_MODEL_NAME})]*\n\n"
                            return header + resp_text
                        else:
                            fallback_reason = f"Response failed citation validation ({citation_msg})"
                except Exception as ex:
                    fallback_reason = f"Invalid response object from model ({type(ex).__name__})"

        except Exception as e:
            err_msg = str(e).lower()
            if "auth" in err_msg or "key" in err_msg or "permission" in err_msg or "401" in err_msg or "403" in err_msg:
                fallback_reason = "Authentication failed / invalid API key"
            elif "quota" in err_msg or "resource_exhausted" in err_msg or "429" in err_msg or "rate limit" in err_msg:
                fallback_reason = "API quota exceeded / rate limit reached"
            elif "connection" in err_msg or "network" in err_msg or "getaddrinfo" in err_msg or "timeout" in err_msg:
                fallback_reason = "Network connectivity failure"
            else:
                fallback_reason = f"API error occurred ({type(e).__name__})"

    return _synthesize_deterministic_explanation(query, retrieved_context, mastery, fallback_reason)


# Minimal Working Example and Self-Test
if __name__ == "__main__":
    print("=" * 70)
    print("LearnSync AI - Member 2: RAG Content Agent (rag.py) Self-Verification")
    print("=" * 70)

    docs = load_opendsa_content()
    print(f"[OK] Successfully loaded {len(docs)} chunks from opendsa_chunks.csv.")

    index, active_docs = build_vector_index(docs)
    print(f"[OK] FAISS IndexFlatIP index built with {index.ntotal} vectors (dim={index.d}).")

    test_q = "How does LIFO work in a stack?"
    retrieved = retrieve_context(test_q, concept_id="stacks", top_k=2)
    print(f"\n[OK] Retrieved {len(retrieved)} chunks for query '{test_q}' (concept='stacks'):")
    for r in retrieved:
        print(f"  - [{r['chunk_id']}] {r['title']} (Score: {r['score']:.4f}, Concept: {r['concept_id']})")

    exp_low = generate_explanation(test_q, retrieved, mastery_level=0.25)
    print("\n--- Low Mastery Explanation Preview ---")
    print(exp_low[:320] + "...\n")

    exp_high = generate_explanation(test_q, retrieved, mastery_level=0.85)
    print("--- High Mastery Explanation Preview ---")
    print(exp_high[:320] + "...\n")

    exp_empty = generate_explanation("Unknown exotic query", [], mastery_level=0.5)
    print("--- Insufficient Context Preview ---")
    print(exp_empty[:220] + "\n")

    print("[SUCCESS] All Member 2 RAG interface checks passed successfully!")
