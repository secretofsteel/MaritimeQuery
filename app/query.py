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


def _extract_topic_keywords(query: str) -> Optional[str]:
    """
    Extract the main semantic topic from a query using Gemini.
    
    Returns:
        Topic string (e.g., "PPE equipment", "bunkering procedures")
        None if no clear topic detected
    """
    prompt = f"""Extract the main topic from this query in 1-4 words. Focus on the subject matter, not the action.

Examples:
"list 3 ppe" → "PPE equipment"
"which are most important?" → "[NO TOPIC]"
"what about bunkering checklist?" → "bunkering operations"
"tell me about MARPOL Annex V" → "MARPOL Annex V"
"what form for enclosed space entry?" → "enclosed space entry"
"how do I report an accident?" → "accident reporting"

Query: "{query}"

Topic (1-4 words or [NO TOPIC]):"""
    
    try:
        response = LlamaSettings.llm.complete(prompt)
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


def _build_conversation_summary(query_history: List[Dict]) -> str:
    """
    Build a running summary of the last few turns instead of full history.
    Saves tokens and keeps focus on current task.
    """
    if not query_history:
        return ""
    
    # Take last 3 exchanges
    recent = query_history[-3:]
    
    if len(recent) == 1:
        # Just one exchange, no need to summarize
        entry = recent[0]
        return f"Previous: User asked about {entry.get('query', 'something')}."
    
    # Build concise summary
    summary_parts = ["Recent conversation:"]
    for idx, entry in enumerate(recent, 1):
        query = entry.get("query", "")[:60]
        topic = entry.get("topic_extracted", "something")
        summary_parts.append(f"{idx}. Asked about {topic}: \"{query}...\"")
    
    return " ".join(summary_parts)


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


def _detect_topic_shift(query: str, query_history: List[Dict]) -> bool:
    """
    Use LLM to detect if the current query represents a topic shift.
    
    Returns:
        True if topic shifted (should reset context)
        False if followup question on same topic
    """
    if not query_history:
        return False  # No history, can't shift
    
    # Get last query and answer for context
    last_entry = query_history[-1]
    last_query = last_entry.get("query", "")
    last_answer = last_entry.get("answer", "")[:200]  # Brief preview
    
    # Simple heuristics first (fast path)
    # If query has pronouns, it's clearly a followup
    pronouns = r'\b(it|that|this|these|those|they|them)\b'
    if re.search(pronouns, query.lower()):
        LOGGER.debug("Detected pronoun in query, assuming followup")
        return False
    
    # If query is very short and question-like, likely a followup
    if len(query.split()) <= 5 and query.strip().endswith('?'):
        LOGGER.debug("Short question detected, assuming followup")
        return False
    
    # Ask LLM to classify (only for ambiguous cases)
    prompt = f"""Previous question: "{last_query}"
Previous answer preview: "{last_answer}"

New question: "{query}"

Is the new question a FOLLOWUP about the same topic, or a DIFFERENT topic?

Reply with ONLY one word:
- FOLLOWUP (if related to previous topic)
- DIFFERENT (if completely new topic)

Reply:"""
    
    try:
        response = LlamaSettings.llm.complete(prompt)
        classification = response.text.strip().upper()
        
        is_shift = "DIFFERENT" in classification
        LOGGER.info("Topic classification: %s (is_shift=%s)", classification, is_shift)
        return is_shift
        
    except Exception as exc:
        LOGGER.warning("Topic detection failed: %s, assuming followup", exc)
        return False  # Default to NOT resetting on errors


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
    
    # STEP 1: Extract semantic topic from current query
    current_topic = _extract_topic_keywords(query_text)
    
    # STEP 2: Topic inheritance - if no clear topic, inherit from last query
    if not current_topic and use_conversation_context and app_state.last_topic:
        current_topic = app_state.last_topic
        LOGGER.info("No topic detected, inheriting: %s", current_topic)
    
    # STEP 3: Detect document type preference (form, checklist, etc.)
    doc_type_preference = _detect_doc_type_preference(query_text)
    
    # STEP 4: Build cache key from semantic topic + doc type
    cache_key = (current_topic, doc_type_preference)
    last_cache_key = (app_state.last_topic, app_state.last_doc_type_pref)
    
    LOGGER.info("Cache keys - current: %s, last: %s", cache_key, last_cache_key)
    
    # STEP 5: Detect topic shift or hard cap
    topic_shift_detected = False
    context_reset_note = None
    
    if use_conversation_context and app_state.context_turn_count > 0:
        # Check for topic shift (semantic comparison)
        if current_topic and app_state.last_topic and current_topic != app_state.last_topic:
            LOGGER.info("Topic shift detected: %s → %s", app_state.last_topic, current_topic)
            app_state.sticky_chunks.clear()
            app_state.context_turn_count = 0
            topic_shift_detected = True
            context_reset_note = f"🔄 Detected topic change ({app_state.last_topic} → {current_topic})"
        # Check hard cap
        elif app_state.context_turn_count >= MAX_CONTEXT_TURNS:
            LOGGER.info("Context turn limit reached (%d), forcing fresh retrieval", MAX_CONTEXT_TURNS)
            app_state.sticky_chunks.clear()
            app_state.context_turn_count = 0
            context_reset_note = f"🔄 Starting fresh search (reached {MAX_CONTEXT_TURNS} turn limit)"
    
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

    # Initialize variables that will be used later
    attempt = 1
    refinement_history: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    final_query_used = query_text
    
    # CONTEXT-AWARE LOGIC: Decide whether to reuse chunks or retrieve fresh
    # Cache key comparison: same topic + same doc type = reuse
    should_reuse_cache = (
        use_conversation_context and 
        app_state.sticky_chunks and 
        len(app_state.sticky_chunks) > 0 and
        app_state.context_turn_count > 0 and 
        not topic_shift_detected and
        cache_key == last_cache_key and
        cache_key[0] is not None  # Must have a topic
    )
    
    if should_reuse_cache:
        # Reuse existing chunks (same semantic task)
        nodes = app_state.sticky_chunks
        retrieval_mode = f"cached (turn {app_state.context_turn_count + 1}/{MAX_CONTEXT_TURNS})"
        LOGGER.info("Cache hit: reusing %d chunks for topic=%s", len(nodes), current_topic)
        
        # Calculate confidence for cached nodes
        confidence_pct, confidence_level, confidence_note = calculate_confidence(nodes)
        
        # Set best_result immediately (skip retrieval loop)
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
        
    else:
        # Fresh retrieval needed
        LOGGER.info("Fresh retrieval: topic=%s, context_enabled=%s, has_chunks=%s", 
                   current_topic, use_conversation_context, bool(app_state.sticky_chunks))
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
            
            # Apply doc type boosting based on metadata
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
        
        # Store chunks and topic for potential reuse in followups
        if use_conversation_context:
            app_state.sticky_chunks = nodes
            app_state.context_turn_count = 1
            app_state.last_topic = current_topic
            app_state.last_doc_type_pref = doc_type_preference
            retrieval_mode = "fresh_retrieval (turn 1)"
        else:
            retrieval_mode = retriever_type

    if not best_result:
        # This should never happen, but if it does, log what went wrong
        LOGGER.error("No best_result set! Debug info: should_reuse_cache=%s, use_context=%s, has_chunks=%s, turn_count=%d, cache_key=%s, last_cache_key=%s",
                    should_reuse_cache, use_conversation_context, bool(app_state.sticky_chunks), 
                    app_state.context_turn_count, cache_key, last_cache_key)
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
    
    # Build conversation summary (replaces full history for token efficiency)
    conversation_summary = ""
    if use_conversation_context and app_state.query_history:
        conversation_summary = _build_conversation_summary(app_state.query_history)
    
    if confidence_pct >= 80:
        confidence_instruction = "You have HIGH confidence sources. Answer authoritatively based on the clear documentation."
    elif confidence_pct >= 60:
        confidence_instruction = "You have MEDIUM confidence sources. Answer based on available information but suggest verification for critical operations."
    else:
        confidence_instruction = "You have LOW confidence sources. Provide what information you found but strongly recommend human verification."

    prompt = f"""You are a maritime safety assistant for ship crew.

CONFIDENCE CONTEXT: {confidence_instruction}

{conversation_summary}

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
    if use_conversation_context:
        if app_state.context_turn_count > 0:
            app_state.context_turn_count += 1
        app_state.last_topic = current_topic
        app_state.conversation_summary = conversation_summary

    result = {
        "query": original_query,
        "topic_extracted": current_topic,  # Store for next turn
        "doc_type_preference": doc_type_preference,
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