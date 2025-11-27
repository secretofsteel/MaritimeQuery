"""Query orchestration, confidence scoring, and reranking - WITH CONTEXT-AWARE CHAT."""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings as LlamaSettings
from google.genai import types

from .config import AppConfig
from .constants import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    MAX_CONTEXT_TURNS,
    CONTEXT_HISTORY_WINDOW,
    DEBUG_RAG,
    HIERARCHICAL_MAX_SECTIONS,
    HIERARCHICAL_MAX_DEPTH,
    HIERARCHICAL_MIN_CONTEXT_TOKENS,
    HIERARCHICAL_MAX_CONTEXT_TOKENS,
)
from .indexing import load_document_trees
from .logger import LOGGER
from .state import AppState

try:  # pragma: no cover - optional dependency
    import cohere
except Exception:  # pragma: no cover - optional dependency
    cohere = None

USE_RERANKER = True
cohere_client = None

if USE_RERANKER and cohere is not None:  # pragma: no cover - runtime configuration
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        try:
            cohere_client = cohere.ClientV2(cohere_api_key)
        except Exception as exc:
            LOGGER.warning("Failed to initialise Cohere client: %s", exc)
            cohere_client = None


def calculate_confidence(nodes: Sequence[NodeWithScore]) -> Tuple[int, str, str]:
    if not nodes:
        return 0, "LOW 🔴", "No relevant documents found."

    scores = [node.score for node in nodes if hasattr(node, "score") and node.score is not None]
    if not scores:
        return 50, "MEDIUM 🟡", "Relevance scores unavailable."

    min_score, max_score = min(scores), max(scores)
    if max_score > min_score:
        scores = [(score - min_score) / (max_score - min_score) for score in scores]
    else:
        scores = [1.0] * len(scores)

    top_score = scores[0]
    top_3_avg = sum(scores[:3]) / min(3, len(scores))
    top_2_avg = sum(scores[:2]) / min(2, len(scores))

    high_quality_count = sum(1 for score in scores[:10] if score > 0.7)
    high_quality_ratio = high_quality_count / min(10, len(scores))
    result_abundance = min(len(nodes) / 10, 1.0)

    confidence = (
        0.30 * top_score
        + 0.25 * top_3_avg
        + 0.20 * top_2_avg
        + 0.15 * high_quality_ratio
        + 0.10 * result_abundance
    )

    confidence_pct = int(confidence * 100)
    if confidence_pct >= CONFIDENCE_HIGH_THRESHOLD:
        level = "HIGH 🟢"
        note = "Based on clear documentation."
    elif confidence_pct >= CONFIDENCE_MEDIUM_THRESHOLD:
        level = "MEDIUM 🟡"
        note = "Based on available information. Verify if critical to operations."
    else:
        level = "LOW 🔴"
        note = "Limited information found. Contact relevant department for verification."

    return confidence_pct, level, note


def reciprocal_rank_fusion(vector_results: Sequence[NodeWithScore], bm25_results: Sequence[NodeWithScore], k: int = 60, top_k: int = 10) -> List[NodeWithScore]:
    scores: Dict[str, float] = {}
    node_map: Dict[str, NodeWithScore] = {}

    for rank, result in enumerate(vector_results, start=1):
        node_id = result.node.node_id
        scores[node_id] = scores.get(node_id, 0) + (1.0 / (k + rank))
        node_map[node_id] = result

    for rank, result in enumerate(bm25_results, start=1):
        node_id = result.node.node_id
        scores[node_id] = scores.get(node_id, 0) + (1.0 / (k + rank))
        node_map.setdefault(node_id, result)

    fused: List[NodeWithScore] = []
    for node_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        node = node_map[node_id]
        fused.append(NodeWithScore(node=node.node, score=score))
    return fused


def maximal_marginal_relevance(nodes: List[NodeWithScore], top_k: int = 20, lambda_param: float = 0.6) -> List[NodeWithScore]:
    if not nodes:
        return nodes
    selected = [nodes[0]]
    remaining = nodes[1:]
    while remaining and len(selected) < top_k:
        best_score = float("-inf")
        best_index = 0
        for index, candidate in enumerate(remaining):
            relevance = candidate.score
            max_similarity = max(selected_node.score * 0.9 for selected_node in selected)
            score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _expand_references(nodes: List[NodeWithScore], vector_retriever, top_k: int = 3) -> List[NodeWithScore]:
    referenced_docs = set()
    for node in nodes[:3]:
        references = node.node.metadata.get("references", "")
        if references:
            for part in references.split("|"):
                for ref in part.split(","):
                    clean_ref = ref.split(":")[-1].strip()
                    if clean_ref:
                        referenced_docs.add(clean_ref)

    expanded: List[NodeWithScore] = []
    for ref in referenced_docs:
        results = vector_retriever.retrieve(ref)
        if results and results[0].score > 0.7:
            candidate = results[0]
            candidate.score *= 0.6
            expanded.append(candidate)
    if expanded:
        nodes.extend(expanded[:top_k])
    return nodes


def _apply_section_score_adjustments(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    for node in nodes:
        section = node.node.metadata.get("section", "")
        if re.match(r"^[A-Z]\.\s+[A-Z]", section):
            node.score *= 1.5
        elif re.search(r"[A-Z]+:\s*\|", section):
            node.score *= 0.5
    return sorted(nodes, key=lambda n: n.score, reverse=True)


def _detect_scope(query: str) -> str:
    """
    Detect the scope/context of the query using simple keyword matching.
    
    Returns:
        "company" - Company-specific procedures/policies
        "regulatory" - IMO/SOLAS/MARPOL regulations
        "operational" - General ship operations
        "safety" - Safety/emergency procedures
        "general" - Default/unclear scope
    """
    query_lower = query.lower()
    
    # Company-specific indicators
    if any(keyword in query_lower for keyword in ["our company", "our policy", "our procedure", "company policy"]):
        return "company"
    
    # Regulatory indicators
    if any(keyword in query_lower for keyword in ["marpol", "solas", "imo", "mlc", "stcw", "regulation", "annex", "chapter"]):
        return "regulatory"
    
    # Safety/emergency indicators
    if any(keyword in query_lower for keyword in ["emergency", "drill", "fire", "abandon", "lifeboat", "evacuation", "alarm"]):
        return "safety"
    
    # Operational indicators
    if any(keyword in query_lower for keyword in ["operation", "procedure", "checklist", "maintenance", "inspection"]):
        return "operational"
    
    return "general"


def _extract_topic_keywords(query: str) -> Optional[str]:
    """
    Extract the BROAD operational context from a query using Gemini Flash Lite.
    Focus on umbrella topics, not specific subtasks.
    
    Returns:
        Umbrella topic string (e.g., "ice operations", "safety equipment")
        None if no clear topic detected
    """
    prompt = f"""Extract the MAIN OPERATIONAL CONTEXT from this query in 2-4 words.
Think about the BROADER TOPIC or operational area, not specific subtasks.

CRITICAL: Group related activities under the same umbrella topic.

Examples of GOOD umbrella topics:
"how should bridge watches be handled in ice waters?" → "ice operations"
"vessel breaking ice without icebreaker" → "ice operations"
"hull breach in ice" → "ice operations"
"how to survive if stuck in ice?" → "ice operations"

"what PPE for deck work?" → "safety equipment"
"which PPE is mandatory?" → "safety equipment"
"how to store PPE?" → "safety equipment"

"how to report a spill?" → "incident reporting"
"how to report an injury?" → "incident reporting"
"what form for near-miss?" → "incident reporting"

"what's the alcohol policy?" → "crew policies"
"how to handle positive test?" → "crew policies"

"ballast water procedure?" → "ballast operations"
"ballast discharge limits?" → "ballast operations"

BAD - too specific:
"bridge watch procedures" → Should be broader operational area
"hull breach response" → Should be part of "damage control" or parent operation

Query: "{query}"

Operational context (2-4 words, be broad):"""
    
    try:
        config = AppConfig.get()
        response = config.client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            ),
        )
        topic = response.text.strip().strip('"').strip("'")
        
        if topic == "[NO TOPIC]" or not topic:
            return None
        
        LOGGER.debug("Extracted topic: %s from query: %s", topic, query[:50])
        return topic
        
    except Exception as exc:
        LOGGER.warning("Topic extraction failed: %s", exc)
        return None


def _detect_doc_type_preference(query: str) -> Optional[str]:
    """
    Detect if user is asking for a specific document type.
    
    Returns:
        "Form" | "Checklist" | "Procedure" | "Policy" | "Manual" | "Regulation" | None
    """
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["form", "forms"]):
        return "Form"
    elif any(word in query_lower for word in ["checklist", "checklists", "check list"]):
        return "Checklist"
    elif any(word in query_lower for word in ["procedure", "procedures", "how to", "steps"]):
        return "Procedure"
    elif any(word in query_lower for word in ["policy", "policies"]):
        return "Policy"
    elif any(word in query_lower for word in ["manual", "manuals", "guide"]):
        return "Manual"
    elif any(word in query_lower for word in ["regulation", "regulations", "requirement"]):
        return "Regulation"
    
    return None


def _rescore_cached_chunks(cached_nodes: List[NodeWithScore], query: str, config: AppConfig) -> List[NodeWithScore]:
    """
    Re-score cached chunks against the new query to ensure relevance.
    Uses embedding similarity for accurate re-ranking.
    
    Args:
        cached_nodes: Previously retrieved chunks
        query: New query text
        config: App configuration for embedding model access
    
    Returns:
        Re-scored and re-sorted nodes
    """
    if not cached_nodes:
        if DEBUG_RAG:
            LOGGER.warning("DEBUG: _rescore_cached_chunks called with empty cached_nodes")
        return cached_nodes
    
    if DEBUG_RAG:
        LOGGER.info("DEBUG: _rescore_cached_chunks starting with %d nodes", len(cached_nodes))
    
    try:
        # Get query embedding using the same model as indexing
        from llama_index.core import Settings as LlamaSettings
        embed_model = LlamaSettings.embed_model
        
        query_embedding = embed_model.get_query_embedding(query)
        
        # Re-score each node based on cosine similarity with new query
        for node in cached_nodes:
            # Get node embedding (already exists in the node)
            if hasattr(node.node, 'embedding') and node.node.embedding:
                node_embedding = node.node.embedding
            else:
                # Fallback: get fresh embedding for node text
                node_embedding = embed_model.get_text_embedding(node.node.text[:512])
            
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
            )
            
            # Update score (blend old score with new similarity)
            # 70% new relevance, 30% original quality score
            node.score = 0.7 * float(similarity) + 0.3 * node.score
        
        # Re-sort by new scores
        rescored = sorted(cached_nodes, key=lambda n: n.score, reverse=True)
        LOGGER.info("✅ Re-scored %d cached chunks against new query", len(rescored))
        return rescored
        
    except Exception as exc:
        LOGGER.warning("Re-scoring failed, using original chunks: %s", exc)
        return cached_nodes


def _detect_topic_shift_with_gemini(query: str, last_query: str, last_answer_preview: str) -> bool:
    """
    Use Gemini Flash Lite to detect if the current query represents a topic shift.
    Uses STRICT criteria - only followups that are direct continuations.
    
    Returns:
        True if topic shifted (should reset context)
        False if direct followup/continuation
    """
    # Ask Gemini with STRICT criteria
    prompt = f"""Previous question: "{last_query}"
Previous answer preview: "{last_answer_preview}"

New question: "{query}"

CRITICAL: Is the new question a DIRECT CONTINUATION of the previous task/topic?

A FOLLOWUP means:
- Asking for more details about the same specific topic
- Asking "what about X?" where X is directly related to previous answer
- Refining/narrowing the previous question
- Asking next steps in the same procedure

DIFFERENT means:
- Completely new topic (even if vaguely related to maritime)
- New operational area
- Different procedure/policy/regulation
- Jumping to a new task

Examples:
Previous: "What is the ballast water procedure?"
New: "What about the discharge limits?" → FOLLOWUP (same procedure, drilling down)
New: "What are the PPE requirements?" → DIFFERENT (completely different topic)

Previous: "How do I handle ice navigation?"
New: "What if the hull gets breached in ice?" → FOLLOWUP (same operational context: ice)
New: "What about fire drills?" → DIFFERENT (different operational area)

Previous: "What's the alcohol policy?"
New: "How do I report a positive test?" → FOLLOWUP (same policy context)
New: "What PPE is required?" → DIFFERENT (different topic)

Previous: "What PPE for deck work?"
New: "Which PPE items are mandatory?" → FOLLOWUP (same topic: PPE requirements)
New: "How to report an injury?" → DIFFERENT (different topic: incident reporting)

Reply with ONLY ONE WORD (exactly as shown, all caps):
- FOLLOWUP (direct continuation of same task/topic)
- DIFFERENT (new topic/operational area)

Reply:"""
    
    try:
        config = AppConfig.get()
        response = config.client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        classification = response.text.strip().upper()
        
        # Strict parsing: only accept exact matches
        if "FOLLOWUP" in classification:
            LOGGER.info("Topic classification: FOLLOWUP (direct continuation)")
            return False
        elif "DIFFERENT" in classification:
            LOGGER.info("Topic classification: DIFFERENT (topic shift)")
            return True
        else:
            # Any other output = assume different (conservative)
            LOGGER.warning("Ambiguous classification: '%s', treating as DIFFERENT", classification)
            return True
        
    except Exception as exc:
        LOGGER.warning("Topic detection failed: %s, assuming DIFFERENT", exc)
        return True  # Conservative: assume shift on errors


def _build_conversation_history_context(app_state: AppState) -> str:
    """
    Build a concise conversation history for context.
    Uses only the last N exchanges (no separate summary).
    """
    if not app_state.query_history:
        return ""
    
    # Get last N exchanges
    recent_history = app_state.query_history[-CONTEXT_HISTORY_WINDOW:]
    
    history_parts = ["=== CONVERSATION HISTORY ==="]
    for idx, entry in enumerate(recent_history, 1):
        query = entry.get("query", "")
        answer = entry.get("answer", "")
        # Truncate long answers
        answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
        history_parts.append(f"\nTurn {idx}:")
        history_parts.append(f"User: {query}")
        history_parts.append(f"Assistant: {answer_preview}")
    
    history_parts.append("\n=== END HISTORY ===\n")
    return "\n".join(history_parts)


def _apply_doc_type_boost(nodes: List[NodeWithScore], preferred_doc_type: Optional[str]) -> List[NodeWithScore]:
    """
    Boost nodes that match the user's preferred document type.
    Uses metadata.doc_type, not filename patterns.
    """
    if not preferred_doc_type:
        return nodes
    
    for node in nodes:
        doc_type = node.node.metadata.get("doc_type", "")
        if doc_type == preferred_doc_type:
            node.score *= 1.3  # 30% boost for matching type
            LOGGER.debug("Boosted %s (type: %s)", node.node.metadata.get("source", "unknown"), doc_type)
    
    # Re-sort after boosting
    return sorted(nodes, key=lambda n: n.score, reverse=True)


def _classify_query_intent_llm(query: str, app_state: Optional[AppState] = None) -> str:
    """
    Classify query intent using LLM for more robust understanding.

    Args:
        query: User's input message
        app_state: Optional app state for conversation context

    Returns:
        "greeting" - Initial greeting (hi, hello)
        "goodbye" - Farewell (bye, see you)
        "thank_you" - Gratitude/acknowledgment (thanks, cool beans)
        "follow_up" - Follow-up question (what else?, tell me more)
        "clarification" - Asking about previous answer (what do you mean?)
        "new_query" - New question needing document search
        "chitchat" - Casual conversation not needing search
    """
    # Build context from recent conversation if available
    context_info = ""
    if app_state and app_state.query_history:
        last_entry = app_state.query_history[-1]
        last_query = last_entry.get("query", "")
        last_answer = last_entry.get("answer", "")[:150]
        context_info = f"""
PREVIOUS CONVERSATION:
User: {last_query}
Assistant: {last_answer}...

"""

    prompt = f"""{context_info}CURRENT USER MESSAGE: "{query}"

You are a query router for a maritime documentation assistant. Your job is to classify the user's intent.

CATEGORY DEFINITIONS:

1. **greeting** - Initial greeting or checking in
   Examples: "hi", "hello", "hey there", "good morning", "how are you"

2. **goodbye** - Ending conversation
   Examples: "bye", "goodbye", "see you later", "take care", "peace out"

3. **thank_you** - Expressing gratitude or acknowledgment (can be conventional or unconventional)
   Examples: "thanks", "thank you", "cool beans dude, good to know", "appreciate it", "perfect", "got it", "roger that"

4. **follow_up** - Wants more information about the same topic
   Examples: "what else?", "tell me more", "anything else?", "can you elaborate?", "what about other options?"

5. **clarification** - Confused about previous answer, asking for explanation
   Examples: "what do you mean?", "I don't understand", "can you explain that?", "unclear"

6. **new_query** - New question requiring document search (maritime procedures, forms, regulations, etc.)
   Examples: "what is the ballast procedure?", "which form for drug testing?", "how to handle ice navigation?"

7. **chitchat** - Casual conversation not requiring document search
   Examples: "how's the weather?", "what can you do?", "tell me a joke"

INSTRUCTIONS:
- If there's previous conversation, use it to detect follow-ups and clarifications
- "what else?" after a previous answer = **follow_up**, NOT greeting
- Unconventional acknowledgments like "cool beans" = **thank_you**, NOT new_query
- Questions about maritime topics = **new_query**
- Short messages can still be follow-ups if context suggests continuation

Classify this message into ONE category (lowercase): greeting, goodbye, thank_you, follow_up, clarification, new_query, or chitchat

Reply with ONLY the category name, nothing else."""

    try:
        config = AppConfig.get()
        response = config.client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

        classification = response.text.strip().lower()

        # Validate classification
        valid_intents = {"greeting", "goodbye", "thank_you", "follow_up", "clarification", "new_query", "chitchat"}

        # Extract the intent if it's in the response
        for intent in valid_intents:
            if intent in classification:
                LOGGER.info("LLM Query Router: %s -> %s", query[:50], intent)
                return intent

        # Fallback: if no valid intent found, default to new_query (conservative)
        LOGGER.warning("LLM router returned invalid classification '%s', defaulting to new_query", classification)
        return "new_query"

    except Exception as exc:
        LOGGER.warning("LLM intent classification failed: %s, falling back to heuristic", exc)
        # Fallback to simple heuristic
        return _classify_query_intent_heuristic(query)


def classify_retrieval_strategy(query: str) -> str:
    """
    Classify query to determine retrieval strategy.

    Uses Gemini Flash Lite to detect procedural queries that benefit from section-level retrieval.

    Args:
        query: User query text

    Returns:
        "chunk_level" - Specific facts, details (existing behavior)
        "section_level" - Procedural queries, how-to, procedures, steps
        "document_level" - Full document context needed (not implemented yet)
    """
    prompt = f"""Classify this maritime query into ONE retrieval strategy:

Query: "{query}"

STRATEGIES:

1. **chunk_level** - Specific facts, discrete details, single data points
   Examples:
   - "What is the discharge temperature limit?"
   - "What's the max ballast capacity?"
   - "Who is responsible for safety equipment?"
   - "What PPE is required for deck work?"

2. **section_level** - Procedural queries needing complete instructions/context
   Examples:
   - "What is the ballast discharge procedure?"
   - "How do I handle ice navigation?"
   - "What are the steps for drug testing?"
   - "What is the procedure for reporting incidents?"
   - "How to conduct fire drills?"

3. **document_level** - Needs full document overview (rare)
   Examples:
   - "Summarize the safety manual"
   - "What's covered in the ISM code?"

RULES:
- If query asks "how to", "procedure", "steps", "process" → section_level
- If query asks "what is [specific fact]" → chunk_level
- If query asks for summary/overview → document_level
- When in doubt, default to chunk_level

Reply with ONLY ONE WORD: chunk_level, section_level, or document_level"""

    try:
        config = AppConfig.get()
        response = config.client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

        classification = response.text.strip().lower()

        # Validate and extract
        if "section_level" in classification or "section-level" in classification:
            LOGGER.info("Query strategy: section_level (procedural)")
            return "section_level"
        elif "document_level" in classification or "document-level" in classification:
            LOGGER.info("Query strategy: document_level (overview)")
            return "document_level"
        else:
            # Default to chunk_level
            LOGGER.info("Query strategy: chunk_level (specific facts)")
            return "chunk_level"

    except Exception as exc:
        LOGGER.warning("Strategy classification failed: %s, defaulting to chunk_level", exc)
        return "chunk_level"


def _collect_section_recursively(
    section: Dict[str, Any],
    collected_chunks: List[NodeWithScore],
    node_map: Dict[str, NodeWithScore],
    depth: int = 0,
    max_depth: int = HIERARCHICAL_MAX_DEPTH
) -> None:
    """
    Recursively collect all chunks from a section and its subsections.

    Args:
        section: Section node from document tree
        collected_chunks: Output list to populate
        node_map: Map of node_id -> NodeWithScore for fast lookup
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
    """
    if depth > max_depth:
        LOGGER.warning("Max depth reached in section recursion")
        return

    # Collect chunks from this section
    chunk_ids = section.get("chunk_ids", [])
    for chunk_id in chunk_ids:
        if chunk_id in node_map:
            collected_chunks.append(node_map[chunk_id])

    # Recursively collect from children
    for child in section.get("children", []):
        _collect_section_recursively(
            child, collected_chunks, node_map, depth + 1, max_depth
        )


def retrieve_hierarchical(
    query: str,
    app_state: AppState,
    top_sections: int = HIERARCHICAL_MAX_SECTIONS
) -> List[NodeWithScore]:
    """
    Retrieve complete sections hierarchically for procedural queries.

    Process:
    1. Use existing hybrid_search to find top 5 relevant chunks
    2. Extract section_ids from those chunks
    3. Load document_trees.json
    4. For each section_id, fetch complete section (all chunks + subsections recursively)
    5. Return structured sections with hierarchy preserved

    Args:
        query: User query
        app_state: Application state with retrievers and nodes
        top_sections: Maximum number of sections to retrieve (token budget)

    Returns:
        List of NodeWithScore with complete section chunks
    """
    LOGGER.info("🔍 Starting hierarchical retrieval for query: %s", query[:50])

    # Early exit if hierarchical not available
    if not getattr(app_state, 'hierarchical_enabled', False):
        LOGGER.warning("Hierarchical retrieval not available, falling back to chunks")
        return []

    # Ensure retrievers are ready
    app_state.ensure_retrievers()
    vector_retriever = app_state.vector_retriever
    bm25_retriever = app_state.bm25_retriever

    if not vector_retriever or not bm25_retriever:
        LOGGER.error("Retrievers not initialized")
        return []

    # Step 1: Use hybrid search to find top relevant chunks
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)
    hybrid_results = reciprocal_rank_fusion(vector_results, bm25_results, k=60, top_k=10)

    if not hybrid_results:
        LOGGER.warning("No hybrid results found")
        return []

    # Step 2: Extract section_ids and document sources from top chunks
    section_refs: List[Tuple[str, str]] = []  # (doc_id, section_id)
    seen_sections = set()

    for node_with_score in hybrid_results[:5]:  # Top 5 chunks to find sections
        metadata = node_with_score.node.metadata
        source = metadata.get("source", "")
        section_id = metadata.get("section_id")

        if not section_id or not source:
            continue

        doc_id = Path(source).stem
        key = (doc_id, section_id)

        if key not in seen_sections:
            section_refs.append(key)
            seen_sections.add(key)

    if not section_refs:
        LOGGER.warning("No section_ids found in top chunks, falling back to chunk-level")
        return hybrid_results

    LOGGER.info("Found %d unique sections from top chunks", len(section_refs))

    # Step 3: Load document trees
    config = AppConfig.get()
    trees_path = config.paths.cache_dir / "document_trees.json"
    document_trees = load_document_trees(trees_path)

    if not document_trees:
        LOGGER.warning("No document trees available, falling back to chunk-level")
        return hybrid_results

    # Build tree index: (doc_id, section_id) -> section_node
    tree_index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def index_section(section: Dict[str, Any], doc_id: str):
        """Recursively index all sections."""
        section_id = section.get("section_id")
        if section_id:
            tree_index[(doc_id, section_id)] = section
        for child in section.get("children", []):
            index_section(child, doc_id)

    for tree in document_trees:
        doc_id = tree.get("doc_id", "")
        for section in tree.get("sections", []):
            index_section(section, doc_id)

    # Build node map: use cached version from AppState
    node_map = app_state.get_node_map()

    # Step 4: Collect complete sections (limited to top_sections and token budget)
    collected_chunks: List[NodeWithScore] = []
    collected_count = 0
    total_tokens = 0  # Track token usage for budget enforcement

    for doc_id, section_id in section_refs[:top_sections]:
        key = (doc_id, section_id)
        if key not in tree_index:
            LOGGER.warning("Section (%s, %s) not found in tree index", doc_id, section_id)
            continue

        section_node = tree_index[key]

        # Collect this section and all subsections recursively
        section_chunks: List[NodeWithScore] = []
        _collect_section_recursively(
            section_node,
            section_chunks,
            node_map
        )

        if section_chunks:
            # Check token budget before adding
            section_tokens = sum(len(c.node.text.split()) for c in section_chunks)

            if total_tokens + section_tokens > HIERARCHICAL_MAX_CONTEXT_TOKENS:
                LOGGER.warning("Token budget exceeded (%d + %d > %d), stopping section collection",
                              total_tokens, section_tokens, HIERARCHICAL_MAX_CONTEXT_TOKENS)
                break

            collected_chunks.extend(section_chunks)
            collected_count += 1
            total_tokens += section_tokens

            LOGGER.info("Collected %d chunks (%d tokens) from section %s (total: %d chunks, %d tokens)",
                       len(section_chunks), section_tokens, section_id,
                       len(collected_chunks), total_tokens)

    if not collected_chunks:
        LOGGER.warning("No chunks collected from sections, falling back to chunk-level")
        return hybrid_results

    LOGGER.info("✅ Hierarchical retrieval complete: %d sections, %d total chunks",
               collected_count, len(collected_chunks))

    return collected_chunks


def format_hierarchical_context(
    nodes: List[NodeWithScore],
    document_trees: List[Dict[str, Any]]
) -> str:
    """
    Format hierarchical sections as structured markdown with proper indentation.

    Example output:
        ## 3. Discharge Procedure
        [Source: Ballast Water Management, Section 3]

          ### 3.1 Pre-Discharge Preparation
          [content from chunks]

            #### 3.1.1 Verify Tank Levels
            [content]

    Args:
        nodes: Retrieved chunks with section_id metadata
        document_trees: Document tree structures for hierarchy info

    Returns:
        Formatted markdown string with hierarchical structure
    """
    # Group chunks by (doc_id, section_id)
    section_chunks: Dict[Tuple[str, str], List[NodeWithScore]] = {}

    for node in nodes:
        metadata = node.node.metadata
        source = metadata.get("source", "")
        section_id = metadata.get("section_id")

        if not section_id:
            continue

        doc_id = Path(source).stem
        key = (doc_id, section_id)

        if key not in section_chunks:
            section_chunks[key] = []

        section_chunks[key].append(node)

    # Build tree index for lookup
    tree_index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def index_section(section: Dict[str, Any], doc_id: str):
        section_id = section.get("section_id")
        if section_id:
            tree_index[(doc_id, section_id)] = section
        for child in section.get("children", []):
            index_section(child, doc_id)

    for tree in document_trees:
        doc_id = tree.get("doc_id", "")
        for section in tree.get("sections", []):
            index_section(section, doc_id)

    # Build formatted output
    output_parts = []

    for (doc_id, section_id), chunks in section_chunks.items():
        # Lookup section info
        section_info = tree_index.get((doc_id, section_id), {})
        title = section_info.get("title", section_id)
        level = section_info.get("level", 1)

        # Markdown heading based on level
        heading_prefix = "#" * (level + 1)  # level 1 → ##, level 2 → ###, etc.
        indent = "  " * (level - 1)  # Indentation for visual hierarchy

        # Format section header
        output_parts.append(f"\n{indent}{heading_prefix} {section_id}. {title}")
        output_parts.append(f"{indent}[Source: {doc_id}, Section {section_id}]\n")

        # Add chunk content
        for chunk in chunks:
            content = chunk.node.text
            # Remove metadata header if present (starts with "Document:")
            if "\n---\n" in content:
                content = content.split("\n---\n", 1)[1]

            output_parts.append(f"{indent}{content}\n")

    return "\n".join(output_parts)


def _classify_query_intent_heuristic(query: str) -> str:
    """
    Fallback heuristic-based classification when LLM fails.
    Kept as backup for reliability.

    Returns:
        "greeting" - Just saying hi/bye/thanks
        "chitchat" - Casual conversation
        "clarification" - Asking about previous answer
        "new_query" - Needs document lookup
    """
    query_lower = query.strip().lower()

    # Fast path: exact matches for common patterns
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "yo", "hey there", "what's going on", "good day", "gday",
                 "hi there", "what's up?", "what's up", "how are you", "whats up", "how's it going", "what's new"}

    if query_lower in greetings or len(query.split()) <= 3:
        return "greeting"

    # Clarification indicators
    clarification_patterns = [
        r"what do you mean",
        r"can you explain",
        r"i don't understand",
        r"unclear",
        r"elaborate",
        r"more details"
    ]

    if any(re.search(pattern, query_lower) for pattern in clarification_patterns):
        return "clarification"

    # If it has maritime/technical terms, definitely search
    maritime_indicators = [
        "ship", "vessel", "cargo", "ballast", "marpol", "solas", "imo",
        "procedure", "form", "checklist", "regulation", "policy", "drill", "manual", "chief engineer",
        "safety", "navigation", "engine", "crew", "master", "officer", "ppe", "supply",
        "logbook", "maintenance", "inspection", "emergency", "lifeboat", "fire", "pollution", "incident", "reporting",
        "deck", "engineer", "pilot", "mooring", "anchoring", "cargo handling", "stcw", "mlc", "ism code", "psc", "flag",
        "port state", "ship management", "seafarer", "nautical", "chart", "ice navigation", "hull", "machinery", "equipment"
    ]

    if any(term in query_lower for term in maritime_indicators):
        return "new_query"

    # Default: if it's a question, search
    if (("?" in query and query_lower not in greetings) or (len(query.split()) > 5)):
        return "new_query"

    # Otherwise, probably chitchat
    return "chitchat"


def query_with_confidence(
    app_state: AppState,
    query_text: str,
    retriever_type: str = "hybrid",
    auto_refine: bool = True,
    max_attempts: int = 3,
    confidence_threshold: int = 65,
    fortify: bool = False,
    expand_references: bool = True,
    rerank: bool = False,
    use_conversation_context: bool = False,
    enable_hierarchical: bool = True,
) -> Dict[str, Any]:
    # ============ INTENT CLASSIFICATION ============
    intent = _classify_query_intent_llm(query_text, app_state)
    LOGGER.info("Query intent: %s", intent)

    # Non-search intents: greeting, goodbye, thank_you, chitchat
    if intent in ["greeting", "goodbye", "thank_you", "chitchat"]:
        # Skip retrieval entirely, respond naturally
        LOGGER.info("Skipping retrieval for %s query", intent)

        # Response templates by intent
        responses = {
            "greeting": "Hello! I'm here to help you with maritime documentation and procedures. What would you like to know?",
            "goodbye": "Safe sailing! Feel free to return anytime you need assistance.",
            "thank_you": "You're welcome! Let me know if you need anything else.",
            "chitchat": "I'm your maritime documentation assistant. Feel free to ask me about ship procedures, safety protocols, forms, or regulations!"
        }

        answer = responses.get(intent, responses["greeting"])

        return {
            "query": query_text,
            "answer": answer,
            "confidence_pct": 100,
            "confidence_level": "HIGH 🟢",
            "confidence_note": f"Direct response ({intent})",
            "sources": [],
            "num_sources": 0,
            "retriever_type": "none",
            "context_mode": use_conversation_context,
            "context_turn": app_state.context_turn_count if use_conversation_context else 0,
            "topic_extracted": None,
            "doc_type_preference": None,
            "scope": "none",
            "attempts": 0,
            "best_attempt": 0,
        }

    # For follow_up and clarification, we'll continue to document search
    # but these intents inform the retrieval strategy below
    if intent == "follow_up":
        LOGGER.info("Follow-up detected, will prefer cached context if available")
    elif intent == "clarification":
        LOGGER.info("Clarification requested, will use cached context")
    elif intent == "new_query":
        LOGGER.info("New query detected, proceeding with document search")
    
    # ============ DEBUG ENTRY POINT ============
    if DEBUG_RAG:
        LOGGER.info("=" * 80)
        LOGGER.info("🔍 DEBUG: QUERY START")
        LOGGER.info("Query: %s", query_text[:100])
        LOGGER.info("Context enabled: %s", use_conversation_context)
        LOGGER.info("State at entry:")
        LOGGER.info("  - context_turn_count: %d", app_state.context_turn_count)
        LOGGER.info("  - sticky_chunks: %s (length: %d)", 
                   bool(app_state.sticky_chunks),
                   len(app_state.sticky_chunks) if app_state.sticky_chunks else 0)
        LOGGER.info("  - last_topic: %s", app_state.last_topic)
        LOGGER.info("  - last_doc_type_pref: %s", getattr(app_state, 'last_doc_type_pref', None))
        LOGGER.info("  - last_scope: %s", getattr(app_state, 'last_scope', None))
        LOGGER.info("  - query_history length: %d", len(app_state.query_history))
        LOGGER.info("=" * 80)
    
    config = AppConfig.get()
    app_state.ensure_retrievers()
    vector_retriever = app_state.vector_retriever
    bm25_retriever = app_state.bm25_retriever
    if not vector_retriever or not bm25_retriever:
        raise RuntimeError("Retrievers not initialized. Load or build the index first.")

    original_query = query_text

    def include_session_uploads(
        base_nodes: List[NodeWithScore], query_for_uploads: str
    ) -> List[NodeWithScore]:
        session_id = app_state.current_session_id
        if not session_id:
            return base_nodes

        try:
            upload_manager = app_state.ensure_session_upload_manager()
            session_nodes = upload_manager.search_session_uploads(
                session_id, query_for_uploads, top_k=10
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Session upload retrieval failed: %s", exc)
            return base_nodes

        if not session_nodes:
            return base_nodes

        combined = session_nodes + base_nodes
        seen: set = set()
        deduped: List[NodeWithScore] = []
        for node in combined:
            metadata = getattr(node.node, "metadata", {}) or {}
            key = (
                metadata.get("upload_chunk_id")
                or metadata.get("node_id")
                or metadata.get("doc_id")
                or id(node.node)
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        added = max(0, len(deduped) - len(base_nodes))
        if added:
            LOGGER.info("Added %d session upload chunks to retrieval", added)
        return deduped

    # STEP 1: Extract semantic topic (broader umbrella concept)
    current_topic = _extract_topic_keywords(query_text)
    if DEBUG_RAG:
        LOGGER.info("DEBUG: Extracted topic: %s", current_topic)
    
    # STEP 2: Topic inheritance - if no clear topic, inherit from last query
    if not current_topic and use_conversation_context and app_state.last_topic:
        current_topic = app_state.last_topic
        LOGGER.info("No topic detected, inheriting: %s", current_topic)
    
    # STEP 3: Detect document type preference
    doc_type_preference = _detect_doc_type_preference(query_text)
    if DEBUG_RAG:
        LOGGER.info("DEBUG: Doc type preference: %s", doc_type_preference)
    
    # STEP 4: Detect scope (NEW - third cache dimension)
    current_scope = _detect_scope(query_text)
    if DEBUG_RAG:
        LOGGER.info("DEBUG: Detected scope: %s", current_scope)
    
    # STEP 5: Build cache key (now includes scope)
    cache_key = (current_topic, doc_type_preference, current_scope)
    last_cache_key = (
        app_state.last_topic,
        getattr(app_state, 'last_doc_type_pref', None),
        getattr(app_state, 'last_scope', None)
    )
    
    LOGGER.info("Cache keys - current: %s, last: %s", cache_key, last_cache_key)
    
    # STEP 6: Intent-aware cache decision (LLM router + topic shift detection)
    topic_shift_detected = False
    context_reset_note = None
    should_reuse_cache = False

    if DEBUG_RAG:
        LOGGER.info("DEBUG: Checking cache strategy based on intent: %s", intent)
        LOGGER.info("  - use_conversation_context: %s", use_conversation_context)
        LOGGER.info("  - context_turn_count > 0: %s", app_state.context_turn_count > 0)
        LOGGER.info("  - query_history exists: %s", bool(app_state.query_history))

    # Priority 1: Clarifications ALWAYS use cache (never search again)
    if intent == "clarification" and app_state.sticky_chunks and len(app_state.sticky_chunks) > 0:
        should_reuse_cache = True
        LOGGER.info("🔍 Clarification detected - using cached context only")
        if DEBUG_RAG:
            LOGGER.info("DEBUG: Forcing cache reuse for clarification")

    # Priority 2: Follow-ups prefer cache, but check for topic shift
    elif use_conversation_context and app_state.context_turn_count > 0 and app_state.query_history:
        if DEBUG_RAG:
            LOGGER.info("DEBUG: Running shift detection...")

        # Get last Q&A for shift detection
        last_entry = app_state.query_history[-1]
        last_query = last_entry.get("query", "")
        last_answer = last_entry.get("answer", "")[:200]

        # For follow_ups, prefer cache but still check shift detection as backup
        if intent == "follow_up":
            LOGGER.info("🔍 Follow-up detected - preferring cached context")
            # For explicit follow-ups, skip topic shift detection and use cache if available
            if app_state.sticky_chunks and len(app_state.sticky_chunks) > 0:
                should_reuse_cache = True
                LOGGER.info("✅ Follow-up intent - reusing cache")
            else:
                should_reuse_cache = False
                if DEBUG_RAG:
                    LOGGER.warning("⚠️ Follow-up but no cached chunks available!")
        else:
            # For new_query, use the existing topic shift detection
            topic_shift_detected = _detect_topic_shift_with_gemini(query_text, last_query, last_answer)

            if DEBUG_RAG:
                LOGGER.info("DEBUG: Shift detected: %s", topic_shift_detected)

            if topic_shift_detected:
                # True topic shift - clear everything
                LOGGER.info("🔄 Topic shift detected, clearing context")
                if DEBUG_RAG:
                    LOGGER.info("DEBUG: Clearing sticky_chunks (was: %d)", len(app_state.sticky_chunks) if app_state.sticky_chunks else 0)
                app_state.sticky_chunks.clear()
                app_state.context_turn_count = 0
                context_reset_note = "🔄 Detected topic change (starting fresh search)"
                should_reuse_cache = False
                if DEBUG_RAG:
                    LOGGER.info("DEBUG: After clear - sticky_chunks length: %d", len(app_state.sticky_chunks))
            else:
                # No shift detected - reuse cache
                if DEBUG_RAG:
                    LOGGER.info("DEBUG: No shift - checking cache availability...")
                    LOGGER.info("  - sticky_chunks exists: %s", bool(app_state.sticky_chunks))
                    LOGGER.info("  - sticky_chunks length: %d", len(app_state.sticky_chunks) if app_state.sticky_chunks else 0)

                # Even if cache keys don't match perfectly, reuse if we have chunks
                if app_state.sticky_chunks and len(app_state.sticky_chunks) > 0:
                    should_reuse_cache = True
                    LOGGER.info("✅ No shift detected - reusing cache")
                else:
                    should_reuse_cache = False
                    if DEBUG_RAG:
                        LOGGER.warning("⚠️ No shift but no cached chunks available!")
    else:
        if DEBUG_RAG:
            LOGGER.info("DEBUG: First query or context disabled - fresh retrieval")
    
    # Check hard cap on turns
    if use_conversation_context and app_state.context_turn_count >= MAX_CONTEXT_TURNS:
        LOGGER.info("Context turn limit reached (%d), forcing fresh retrieval", MAX_CONTEXT_TURNS)
        app_state.sticky_chunks.clear()
        app_state.context_turn_count = 0
        context_reset_note = f"🔄 Starting fresh search (reached {MAX_CONTEXT_TURNS} turn limit)"
        should_reuse_cache = False
    
    if fortify:
        prompt = textwrap.dedent(
            f"""You are a maritime documentation expert who helps improve search queries for maritime-related databases.

Given this user query: "{query_text}", your task is to expand it to improve search results in a maritime knowledge base.

Instructions:
1. Identify the underlying requirement or intent of the query.
2. Extract the main search terms.
3. Suggest 2–4 related or synonymous terms actually used in maritime manuals, forms, or procedures.
4. Suggest 1–3 relevant maritime areas (e.g., 'crew management', 'safety procedures', 'ship maintenance', 'ISM Code compliance').
5. Correct typos or unclear phrasing, but do not change the meaning.
6. Keep everything tightly relevant to maritime terminology – no generic fluff.
7. Avoid duplication.
8. Return your answer strictly in the following readable format:

Corrected Query: <text>
Keywords: term1, term2, term3
Related Terms: term1, term2
Maritime Areas: area1, area2

Keep it clean, compact, and human-readable – not JSON, not a list, just plain text formatted as above."""
        )
        fortify_query = config.client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024)
            ),
        )
        query_text = fortify_query.candidates[0].content.parts[0].text

    # Initialize variables
    attempt = 1
    refinement_history: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    final_query_used = query_text
    
    # ============ DECISION POINT ============
    if DEBUG_RAG:
        LOGGER.info("DEBUG: Cache decision - should_reuse_cache: %s", should_reuse_cache)
    
    # DECISION: Reuse cache or retrieve fresh?
    if should_reuse_cache:
        LOGGER.info("🎯 CACHE HIT PATH - Re-scoring cached chunks")
        if DEBUG_RAG:
            LOGGER.info("DEBUG: sticky_chunks length before rescore: %d", len(app_state.sticky_chunks))
        
        # Re-score cached chunks against new query
        nodes = _rescore_cached_chunks(app_state.sticky_chunks, query_text, config)
        if DEBUG_RAG:
            LOGGER.info("DEBUG: Nodes after rescore: %d", len(nodes))

        nodes = include_session_uploads(nodes, query_text)

        retrieval_mode = f"cached+rescored (turn {app_state.context_turn_count + 1}/{MAX_CONTEXT_TURNS})"

        # Calculate confidence for re-scored nodes
        confidence_pct, confidence_level, confidence_note = calculate_confidence(nodes)
        if DEBUG_RAG:
            LOGGER.info("DEBUG: Confidence calculated: %d%%", confidence_pct)
        
        # Set best_result immediately
        best_result = {
            "attempt": 1,
            "query": query_text,
            "nodes": nodes,
            "confidence_pct": confidence_pct,
            "confidence_level": confidence_level,
            "confidence_note": confidence_note,
        }
        final_query_used = query_text
        attempt = 1
        if DEBUG_RAG:
            LOGGER.info("DEBUG: best_result set in cache path")
        
    else:
        LOGGER.info("🔍 FRESH RETRIEVAL PATH")
        # Fresh retrieval
        LOGGER.info("Fresh retrieval: topic=%s, scope=%s, context_enabled=%s",
                   current_topic, current_scope, use_conversation_context)

        # ============ HIERARCHICAL RETRIEVAL INTEGRATION ============
        # Classify retrieval strategy for search intents
        retrieval_strategy = "chunk_level"  # Default
        hierarchical_success = False

        if intent in ["new_query", "follow_up"]:
            # Only classify strategy for actual search queries
            retrieval_strategy = classify_retrieval_strategy(query_text)
            LOGGER.info("📊 Retrieval strategy: %s", retrieval_strategy)

            # Route based on strategy
            if retrieval_strategy == "section_level" and app_state.hierarchical_enabled and enable_hierarchical:
                # Use hierarchical retrieval for procedural queries
                LOGGER.info("🔍 Using hierarchical retrieval (section-level)")

                nodes = retrieve_hierarchical(
                    query_text,
                    app_state,
                    top_sections=HIERARCHICAL_MAX_SECTIONS
                )

                # Fallback: if insufficient context, use chunk-level
                total_text = "".join([n.node.text for n in nodes]) if nodes else ""
                if not nodes or len(total_text) < HIERARCHICAL_MIN_CONTEXT_TOKENS:
                    LOGGER.warning("Hierarchical retrieval insufficient (%d tokens), falling back to chunks",
                                 len(total_text))
                    retrieval_strategy = "chunk_level"
                    # nodes will be retrieved in chunk-level path below
                else:
                    LOGGER.info("✅ Hierarchical retrieval: %d chunks from complete sections", len(nodes))
                    hierarchical_success = True

                    # Apply post-retrieval processing
                    nodes = _apply_section_score_adjustments(nodes)

                    if expand_references and nodes:
                        nodes = _expand_references(nodes, vector_retriever)

                    nodes = include_session_uploads(nodes, query_text)

                    # Reranking for hierarchical results (optional)
                    if rerank and USE_RERANKER and cohere_client:
                        try:
                            documents = [node.node.text[:1000] for node in nodes]
                            rerank_results = cohere_client.rerank(
                                model="rerank-v3.5",
                                query=query_text,
                                documents=documents,
                                top_n=min(len(documents), 30),
                            )
                            nodes = [
                                NodeWithScore(node=nodes[result.index].node, score=result.relevance_score)
                                for result in rerank_results.results
                            ]
                        except Exception as exc:  # pragma: no cover - optional path
                            LOGGER.warning("Reranking failed: %s", exc)

                    # Calculate confidence
                    confidence_pct, confidence_level, confidence_note = calculate_confidence(nodes)

                    # Set best_result for hierarchical path
                    best_result = {
                        "attempt": 1,
                        "query": query_text,
                        "nodes": nodes,
                        "confidence_pct": confidence_pct,
                        "confidence_level": confidence_level,
                        "confidence_note": f"{confidence_note} (hierarchical)",
                    }
                    final_query_used = query_text
                    attempt = 1

            elif retrieval_strategy == "document_level":
                # Not implemented yet - fall back to section_level
                LOGGER.info("Document-level not implemented, using section-level")
                # Try hierarchical with more sections
                if app_state.hierarchical_enabled:
                    nodes = retrieve_hierarchical(query_text, app_state, top_sections=3)

                    total_text = "".join([n.node.text for n in nodes]) if nodes else ""
                    if nodes and len(total_text) >= HIERARCHICAL_MIN_CONTEXT_TOKENS:
                        hierarchical_success = True
                        nodes = _apply_section_score_adjustments(nodes)
                        nodes = include_session_uploads(nodes, query_text)
                        confidence_pct, confidence_level, confidence_note = calculate_confidence(nodes)
                        best_result = {
                            "attempt": 1,
                            "query": query_text,
                            "nodes": nodes,
                            "confidence_pct": confidence_pct,
                            "confidence_level": confidence_level,
                            "confidence_note": f"{confidence_note} (hierarchical)",
                        }
                        final_query_used = query_text
                        attempt = 1
                    else:
                        retrieval_strategy = "chunk_level"
                else:
                    retrieval_strategy = "chunk_level"

        # If strategy is chunk_level OR hierarchical failed, use existing retrieval
        if retrieval_strategy == "chunk_level" or not hierarchical_success:
            if retrieval_strategy == "chunk_level":
                LOGGER.info("🔍 Using chunk-level retrieval")

            while attempt <= max_attempts:
                if retriever_type == "vector":
                    vector_nodes = vector_retriever.retrieve(query_text)
                    bm25_nodes: List[NodeWithScore] = []
                    nodes = list(vector_nodes)
                elif retriever_type == "bm25":
                    bm25_nodes = bm25_retriever.retrieve(query_text)
                    vector_nodes = []
                    nodes = list(bm25_nodes)
                else:
                    vector_nodes = vector_retriever.retrieve(query_text)
                    bm25_nodes = bm25_retriever.retrieve(query_text)
                    nodes = reciprocal_rank_fusion(vector_nodes, bm25_nodes, k=60, top_k=40)

                nodes = _apply_section_score_adjustments(nodes)

                if expand_references and nodes:
                    nodes = _expand_references(nodes, vector_retriever)

                nodes = include_session_uploads(nodes, query_text)

                if rerank and USE_RERANKER and cohere_client:
                    try:
                        documents = [node.node.text[:1000] for node in nodes]
                        rerank_results = cohere_client.rerank(
                            model="rerank-v3.5",
                            query=query_text,
                            documents=documents,
                            top_n=min(len(documents), 30),
                        )
                        nodes = [
                            NodeWithScore(node=nodes[result.index].node, score=result.relevance_score)
                            for result in rerank_results.results
                        ]
                    except Exception as exc:  # pragma: no cover - optional path
                        LOGGER.warning("Reranking failed: %s", exc)

                nodes = maximal_marginal_relevance(nodes, top_k=30, lambda_param=0.6)
                confidence_pct, confidence_level, confidence_note = calculate_confidence(nodes)

                # Apply doc type boosting
                if doc_type_preference:
                    nodes = _apply_doc_type_boost(nodes, doc_type_preference)
                    LOGGER.info("Applied doc type boost: %s", doc_type_preference)

                current_result = {
                    "attempt": attempt,
                    "query": query_text,
                    "nodes": nodes,
                    "confidence_pct": confidence_pct,
                    "confidence_level": confidence_level,
                    "confidence_note": confidence_note,
                }
                if best_result is None or confidence_pct > best_result["confidence_pct"]:
                    best_result = current_result

                refinement_history.append({"attempt": attempt, "query": query_text, "confidence": confidence_pct})

                if confidence_pct < confidence_threshold and attempt < max_attempts and auto_refine:
                    prompt = textwrap.dedent(
                        f"""You are helping to search maritime company documentation (manuals, forms, procedures).

Original question: "{query_text}"

The search works best with simple, direct questions using business English terminology like:
- "What are [role]'s responsibilities?"
- "Which form is used for [task]?"
- "What is the procedure for [action]?"

Avoid fancy language. Use simple words that would appear in a company manual.

Previous attempts: {[entry['query'] for entry in refinement_history]}

Rephrase this question to better match maritime documentation language. Do not over-complicate the language. Do not repeat the question as phrased by the user. Return ONLY the rephrased question, nothing else."""
                    )
                    refined = LlamaSettings.llm.complete(prompt)
                    query_text = refined.text.strip().strip('"').strip("'")
                    attempt += 1
                    continue
                break
        
        if DEBUG_RAG:
            LOGGER.info("DEBUG: Fresh retrieval complete, nodes: %d", len(nodes))
        
        # Store chunks for potential reuse
        if use_conversation_context:
            if DEBUG_RAG:
                LOGGER.info("DEBUG: Storing chunks for future reuse...")
                LOGGER.info("  - Storing %d nodes in sticky_chunks", len(nodes))
            
            app_state.sticky_chunks = nodes
            
            # Only set to 1 if this is the very first query (count was 0)
            if app_state.context_turn_count == 0:
                app_state.context_turn_count = 1
                if DEBUG_RAG:
                    LOGGER.info("DEBUG: First query - set context_turn_count to 1")
            else:
                if DEBUG_RAG:
                    LOGGER.info("DEBUG: Followup - leaving context_turn_count at %d (will increment at end)", app_state.context_turn_count)
            
            app_state.last_topic = current_topic
            if hasattr(app_state, 'last_doc_type_pref'):
                app_state.last_doc_type_pref = doc_type_preference
            if hasattr(app_state, 'last_scope'):
                app_state.last_scope = current_scope
            
            if DEBUG_RAG:
                LOGGER.info("DEBUG: After storage - sticky_chunks length: %d", len(app_state.sticky_chunks))
            
            retrieval_mode = f"fresh_retrieval (turn {app_state.context_turn_count})"
        else:
            retrieval_mode = retriever_type

    if DEBUG_RAG:
        LOGGER.info("DEBUG: Checking best_result before generation...")
    
    if not best_result:
        LOGGER.error("❌ No best_result set! This should never happen.")
        if DEBUG_RAG:
            LOGGER.error("DEBUG state dump:")
            LOGGER.error("  - should_reuse_cache was: %s", should_reuse_cache)
            LOGGER.error("  - use_conversation_context: %s", use_conversation_context)
            LOGGER.error("  - sticky_chunks: %s", bool(app_state.sticky_chunks))
            LOGGER.error("  - context_turn_count: %d", app_state.context_turn_count)
        raise RuntimeError("No retrieval results available.")

    nodes = best_result.get("nodes", nodes)
    confidence_pct = best_result.get("confidence_pct", 0)
    confidence_level = best_result.get("confidence_level", "N/A")
    confidence_note = best_result.get("confidence_note", "")
    final_query_used = best_result.get("query", query_text)

    if DEBUG_RAG:
        LOGGER.info("DEBUG: Building context from %d nodes", len(nodes))

    # Separate context by source type
    upload_context_parts = []
    library_context_parts = []
    sources_info = []
    
    upload_counter = 1
    library_counter = 1
    
    for node in nodes[:10]:
        metadata = node.node.metadata
        source = metadata.get("source", "Unknown")
        section = metadata.get("section", "N/A")
        score = node.score if hasattr(node, "score") else 0.0
        
        is_upload = metadata.get("session_upload", False)
        
        if is_upload:
            upload_context_parts.append(
                f"[Upload {upload_counter}: {source} - {section}]\n{node.node.text}\n"
            )
            upload_counter += 1
        else:
            library_context_parts.append(
                f"[Library {library_counter}: {source} - {section}]\n{node.node.text}\n"
            )
            library_counter += 1

        sources_info.append(
            {
                "source": source,
                "section": section,
                "score": score,
                "title": metadata.get("title", ""),
                "doc_type": metadata.get("doc_type", ""),
                "hierarchy": metadata.get("hierarchy", ""),
                "tab_name": metadata.get("tab_name", ""),
                "form_number": metadata.get("form_number", ""),
                "form_category_name": metadata.get("form_category_name", ""),
                "session_upload": metadata.get("session_upload", False),
                "upload_display_name": metadata.get("upload_display_name", ""),
                "upload_original_name": metadata.get("upload_original_name", ""),
            }
        )

    # Build tiered context
    context_sections = []
    
    if upload_context_parts:
        context_sections.append(
            "=== UPLOADED DOCUMENTS (User attached for this conversation) ===\n"
            + "\n".join(upload_context_parts)
        )
    
    if library_context_parts:
        context_sections.append(
            "=== COMPANY LIBRARY DOCUMENTATION ===\n"
            + "\n".join(library_context_parts)
        )
    
    context = "\n\n".join(context_sections)
    
    # Build conversation history (single method)
    conversation_history = ""
    if use_conversation_context and app_state.query_history:
        conversation_history = _build_conversation_history_context(app_state)
    
    if confidence_pct >= 80:
        confidence_instruction = "You have HIGH confidence sources. Answer authoritatively based on the clear documentation."
    elif confidence_pct >= 60:
        confidence_instruction = "You have MEDIUM confidence sources. Answer based on available information but suggest verification for critical operations."
    else:
        confidence_instruction = "You have LOW confidence sources. Provide what information you found but strongly recommend human verification."

    prompt = f"""You are a maritime safety assistant for ship crew.

CONFIDENCE CONTEXT: {confidence_instruction}

{conversation_history}

NOTE: When answering, prioritize information from UPLOADED DOCUMENTS when relevant, as the user specifically attached these files for this conversation. Cross-reference COMPANY LIBRARY for additional context and procedures.

CRITICAL RULES:
- You ALWAYS answer in English, even if asked in another language.
- Answer facts must come ONLY from the provided documents, but you can use logic, general well-established facts of life, and common sense in crafting your replies, as well.
- General maritime knowledge needed to make logical connections between the provided documents can be used from your internal knowledge base
- If information is missing or unclear, say so explicitly
- When referencing forms, ALWAYS include the complete form name/number (e.g., "Form_DA 005 Drug and Alcohol Test Report")
- Cite sources using [Source number, Source title > Section] format (e.g. [3, Chapter 5 MASTER'S RESPONSIBILITY AND AUTHORITY > Master's Overriding Authority]). Don't use filenames as title, unless no actual title is available. If you have to use filenames, remove the extension.
- Keep answers concise (3-4 minute read maximum)
- Use "Based on the documents on file..."
- If the question asks "which form" or "what form", or "what checklist" start your answer with the exact form or checklist name
- If the question asks for advice, you can synthesize by including common sense/logic/knowledge of the world
- You are allowed to draft messages on behalf of the user, if so asked
- When you can infer something from the context, you can do so, giving a 'heads up' like "it can be assumed that..." or "we can infer from this that..." or likewise
- ALWAYS double check that you have exhausted all available information from the context provided and ensure not to omit any detail that could relate to the query. Better to include extra info than omit parts of the answer that may be crucial to the user.
- When asked to list details from a Form or Checklist, ensure that you don't omit any part or section. Providing the full picture is critical.
- When making a table, don't make a column for the sources, instead place them after the table. If references within the table share the same source, include it only once.
- If this is a followup question referring to previous conversation (using "it", "that", "this", etc.), use the conversation history above to understand the context.

CONTEXT:
{context}

QUESTION: {original_query}

Please provide a clear, concise answer with proper citations."""

    response = LlamaSettings.llm.complete(prompt)
    answer_text = response.text
    
    # Update conversation state
    if DEBUG_RAG:
        LOGGER.info("DEBUG: Updating conversation state...")
    
    if use_conversation_context:
        app_state.context_turn_count += 1
        if DEBUG_RAG:
            LOGGER.info("DEBUG: Incremented context_turn_count to %d", app_state.context_turn_count)
        app_state.last_topic = current_topic
        if hasattr(app_state, 'last_doc_type_pref'):
            app_state.last_doc_type_pref = doc_type_preference
        if hasattr(app_state, 'last_scope'):
            app_state.last_scope = current_scope

    # Calculate sections retrieved for hierarchical
    sections_retrieved_count = 0
    if 'retrieval_strategy' in locals() and retrieval_strategy == "section_level":
        sections_retrieved_count = len(set(
            n.node.metadata.get('section_id')
            for n in nodes
            if n.node.metadata.get('section_id')
        ))

    result = {
        "query": original_query,
        "topic_extracted": current_topic,
        "doc_type_preference": doc_type_preference,
        "scope": current_scope,
        "final_query": final_query_used if final_query_used != original_query else None,
        "refinement_history": refinement_history if len(refinement_history) > 1 else None,
        "attempts": attempt,
        "best_attempt": best_result["attempt"],
        "answer": answer_text,
        "confidence_pct": confidence_pct,
        "confidence_level": confidence_level,
        "confidence_note": confidence_note,
        "sources": sources_info,
        "num_sources": len(nodes),
        "retriever_type": retrieval_mode,
        "retrieval_strategy": retrieval_strategy if 'retrieval_strategy' in locals() else "chunk_level",
        "sections_retrieved": sections_retrieved_count,
        "context_mode": use_conversation_context,
        "context_turn": app_state.context_turn_count if use_conversation_context else 0,
        "context_reset_note": context_reset_note,
    }
    
    if DEBUG_RAG:
        LOGGER.info("=" * 80)
        LOGGER.info("✅ DEBUG: QUERY COMPLETE")
        LOGGER.info("State at exit:")
        LOGGER.info("  - context_turn_count: %d", app_state.context_turn_count)
        LOGGER.info("  - sticky_chunks length: %d", len(app_state.sticky_chunks) if app_state.sticky_chunks else 0)
        LOGGER.info("  - retriever_type: %s", retrieval_mode)
        LOGGER.info("=" * 80)
    
    return result


__all__ = [
    "calculate_confidence",
    "reciprocal_rank_fusion",
    "query_with_confidence",
    "classify_retrieval_strategy",
    "retrieve_hierarchical",
    "format_hierarchical_context",
]
