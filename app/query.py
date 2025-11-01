"""Query orchestration, confidence scoring, and reranking - WITH CONTEXT-AWARE CHAT."""

from __future__ import annotations

import os
import re
import textwrap
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
    TOPIC_SHIFT_THRESHOLD,
)
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


def _detect_topic_shift(query: str, query_history: List[Dict], similarity_threshold: float = TOPIC_SHIFT_THRESHOLD) -> bool:
    """
    Detect if the current query represents a topic shift from recent conversation.
    
    Uses simple keyword overlap heuristic:
    - Extract keywords from recent queries
    - Compare with current query
    - If overlap < threshold, it's likely a new topic
    """
    if not query_history:
        return False  # No history, can't shift
    
    # Get last 2 queries for comparison
    recent_queries = [entry.get("query", "") for entry in query_history[-2:]]
    
    # Simple keyword extraction (lowercase, remove common words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "who", 
                  "when", "where", "how", "about", "for", "to", "of", "in", "on"}
    
    def extract_keywords(text: str) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if len(w) > 3 and w not in stop_words}
    
    current_keywords = extract_keywords(query)
    recent_keywords = set()
    for q in recent_queries:
        recent_keywords.update(extract_keywords(q))
    
    if not current_keywords or not recent_keywords:
        return False  # Can't determine, assume same topic
    
    # Calculate overlap
    overlap = len(current_keywords & recent_keywords) / len(current_keywords)
    
    # Topic shift if very low overlap
    is_shift = overlap < similarity_threshold
    
    if is_shift:
        LOGGER.info("Topic shift detected: overlap=%.2f, current=%s, recent=%s", 
                   overlap, current_keywords, recent_keywords)
    
    return is_shift


def _build_conversation_history_context(app_state: AppState) -> str:
    """Build a concise conversation history for context."""
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
    use_conversation_context: bool = False,  # NEW: Enable context-aware mode
) -> Dict[str, Any]:
    config = AppConfig.get()
    app_state.ensure_retrievers()
    vector_retriever = app_state.vector_retriever
    bm25_retriever = app_state.bm25_retriever
    if not vector_retriever or not bm25_retriever:
        raise RuntimeError("Retrievers not initialized. Load or build the index first.")

    original_query = query_text
    
    # SOFT RESET: Detect topic shift and auto-reset context
    topic_shift_detected = False
    if use_conversation_context and app_state.context_turn_count > 0:
        if _detect_topic_shift(query_text, app_state.query_history):
            LOGGER.info("Topic shift detected, performing soft reset")
            app_state.sticky_chunks.clear()
            app_state.context_turn_count = 0
            topic_shift_detected = True
            context_reset_note = "🔄 Detected topic change - searching fresh sources"
        # Check if we need to reset context (hard cap)
        elif app_state.context_turn_count >= MAX_CONTEXT_TURNS:
            LOGGER.info("Context turn limit reached (%d), forcing fresh retrieval", MAX_CONTEXT_TURNS)
            app_state.sticky_chunks.clear()
            app_state.context_turn_count = 0
            context_reset_note = f"🔄 Starting fresh search (reached {MAX_CONTEXT_TURNS} turn limit)"
        else:
            context_reset_note = None
    else:
        context_reset_note = None
    
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

    attempt = 1
    refinement_history: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None

    # CONTEXT-AWARE LOGIC: Reuse chunks or retrieve fresh?
    if use_conversation_context and app_state.sticky_chunks and app_state.context_turn_count > 0 and not topic_shift_detected:
        # Reuse existing chunks from conversation (no topic shift)
        nodes = app_state.sticky_chunks
        retrieval_mode = f"conversation_context (turn {app_state.context_turn_count + 1}/{MAX_CONTEXT_TURNS})"
        LOGGER.info("Reusing %d sticky chunks for followup query", len(nodes))
    else:
        # Fresh retrieval
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
        
        # Store chunks for potential reuse in followups
        if use_conversation_context:
            app_state.sticky_chunks = nodes
            app_state.context_turn_count = 1
            retrieval_mode = "fresh_retrieval (turn 1)"
        else:
            retrieval_mode = retriever_type

    if not best_result:
        raise RuntimeError("No retrieval results available.")

    nodes = best_result.get("nodes", nodes)
    confidence_pct = best_result.get("confidence_pct", 0)
    confidence_level = best_result.get("confidence_level", "N/A")
    confidence_note = best_result.get("confidence_note", "")
    final_query_used = best_result.get("query", query_text)

    context_parts = []
    sources_info = []
    for index, node in enumerate(nodes[:10], start=1):
        metadata = node.node.metadata
        source = metadata.get("source", "Unknown")
        section = metadata.get("section", "N/A")
        score = node.score if hasattr(node, "score") else 0.0
        context_parts.append(f"[Source {index}: {source} - {section}]\n{node.node.text}\n")
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
            }
        )

    context = "\n".join(context_parts)
    
    # Add conversation history if context mode is enabled
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
    
    # Increment context turn counter if in conversation mode
    if use_conversation_context and app_state.context_turn_count > 0:
        app_state.context_turn_count += 1

    result = {
        "query": original_query,
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
        "context_mode": use_conversation_context,
        "context_turn": app_state.context_turn_count if use_conversation_context else 0,
        "context_reset_note": context_reset_note,
    }
    
    return result


__all__ = ["calculate_confidence", "reciprocal_rank_fusion", "query_with_confidence"]