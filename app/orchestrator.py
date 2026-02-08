"""Phase 5: Query Orchestrator & Decomposition.

This module implements intelligent query routing, decomposition, filtered
retrieval, and strategy-aware synthesis for the Maritime RAG system.

Phase 5a: QueryAnalyzer (this file's initial scope)
Phase 5b: FilteredRetriever
Phase 5c: QueryDecomposer
Phase 5d: ResultSynthesizer
Phase 5e: Integration + orchestrated_query() entry point
"""

from __future__ import annotations

import json as json_module
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, TYPE_CHECKING

from google.genai import types
from google.genai.types import ThinkingConfig
from llama_index.core.schema import NodeWithScore

from .config import AppConfig
from .constants import CONTEXT_HISTORY_WINDOW, MAX_CONTEXT_TURNS
from .logger import LOGGER
from .query import reciprocal_rank_fusion, calculate_confidence

if TYPE_CHECKING:
    from .state import AppState


# =============================================================================
# Enums
# =============================================================================

class Intent(str, Enum):
    """User intent classification."""
    GREETING = "greeting"
    GOODBYE = "goodbye"
    THANK_YOU = "thank_you"
    OFF_TOPIC = "off_topic"
    INCOMPLETE = "incomplete"
    FOLLOW_UP_SAME = "follow_up_same"   # Same topic, same facet (drill-down)
    FOLLOW_UP_NEW = "follow_up_new"     # Same topic, different facet
    NEW_QUERY = "new_query"


class QueryType(str, Enum):
    """Classification of the query's information need."""
    SIMPLE_FACTUAL = "simple_factual"
    PROCEDURAL = "procedural"
    MULTI_PART = "multi_part"
    COMPLIANCE = "compliance"


class ScopeAssessment(str, Enum):
    """How broad/narrow the query is."""
    FOCUSED = "focused"
    BROAD = "broad"
    TOO_BROAD = "too_broad"


class RetrievalStrategy(str, Enum):
    """How retrieval should be executed."""
    CHUNK = "chunk"
    SECTION = "section"
    FILTERED_PARALLEL = "filtered_parallel"


class MergeStrategy(str, Enum):
    """How results from (possibly multiple) retrievals should be combined."""
    DIRECT = "direct"
    SYNTHESIZE = "synthesize"
    COMPLIANCE = "compliance"


# =============================================================================
# Dataclasses
# =============================================================================

NON_RETRIEVAL_INTENTS = frozenset({
    Intent.GREETING,
    Intent.GOODBYE,
    Intent.THANK_YOU,
    Intent.OFF_TOPIC,
    Intent.INCOMPLETE,
})


@dataclass
class QueryAnalysis:
    """Complete analysis of a user query — output of QueryAnalyzer."""

    original_query: str
    intent: Intent
    query_type: Optional[QueryType]         # None for non-retrieval intents
    topic: Optional[str]
    detected_sources: List[str]             # e.g. ["RISQ", "PMS"]
    detected_doc_types: List[str]           # e.g. ["REGULATION", "PROCEDURE"]
    has_regulatory_standard: bool
    scope: ScopeAssessment
    retrieval_strategy: RetrievalStrategy
    merge_strategy: MergeStrategy
    requires_decomposition: bool
    doc_type_hint: Optional[str]
    direct_response: Optional[str]          # For non-retrieval intents
    topic_inherited: bool = False           # True if topic carried from context


@dataclass
class SubQuery:
    """A single retrieval sub-query with optional source filters.

    Produced by the QueryDecomposer (Phase 5c) for complex queries,
    or constructed directly by the orchestrator for simple queries.
    """

    text: str                                           # Retrieval query text
    source_label: str                                   # Human label: "RISQ requirements"
    doc_type_filter: Optional[List[str]] = None         # e.g. ["REGULATION", "VETTING"]
    title_filter: Optional[str] = None                  # Substring match on title
    is_standard: bool = False                           # Compliance: is this the benchmark?
    top_k: int = 20                                     # Per-source retrieval budget


@dataclass
class RetrievalResult:
    """Result of a single filtered retrieval pass.

    Carries the sub-query that produced it, the ranked nodes,
    and a human-readable label for the synthesizer.
    """

    sub_query: SubQuery
    nodes: List[NodeWithScore]
    source_label: str


# =============================================================================
# Source / document type reference patterns
# =============================================================================

# Known source patterns the LLM should recognize in queries.
# Provided as part of the classification prompt so the model can
# map user language ("Rightship", "our procedures") to structured labels.

SOURCE_PATTERNS = """
REGULATORY SOURCES (international/industry standards):
- RISQ / RISQ2 / RightShip: RightShip's inspection questionnaire (vetting standard)
- TMSA / TMSA3: Tanker Management Self-Assessment (OCIMF vetting standard)
- SIRE / SIRE 2.0: Ship Inspection Report Programme (OCIMF vetting standard)
- CDI: Chemical Distribution Institute (chemical tanker vetting)
- MARPOL: International Convention for the Prevention of Pollution from Ships
- SOLAS: International Convention for the Safety of Life at Sea
- ISM / ISM Code: International Safety Management Code
- ISPS / ISPS Code: International Ship and Port Facility Security Code
- STCW: Standards of Training, Certification and Watchkeeping
- MLC: Maritime Labour Convention
- IMDG / IMDG Code: International Maritime Dangerous Goods Code
- IMSBC / IMSBC Code: International Maritime Solid Bulk Cargoes Code
- BLU Code: Code of Practice for the Safe Loading and Unloading of Bulk Carriers
- BWM / BWM Convention: Ballast Water Management Convention
- USCG: United States Coast Guard (regulations/inspections)
- AMSA: Australian Maritime Safety Authority (regulations/inspections)
- PSC: Port State Control (inspections)

COMPANY SOURCES (internal documentation):
- IMS: Integrated Management System (company procedures/manuals)
- SMS: Safety Management System (company safety procedures)
- PMS: Planned Maintenance System (maintenance procedures/records)
- "our procedures" / "our policy" / "company policy": Company internal documentation

DOCUMENT TYPES:
- REGULATION: International conventions, codes (MARPOL, SOLAS, ISM, etc.)
- VETTING: Inspection questionnaires and standards (RISQ, TMSA, SIRE, CDI)
- PROCEDURE: Company operational procedures
- POLICY: Company policies
- FORM: Standardized forms (often with codes like DA-005, N-001)
- CHECKLIST: Operational checklists
- MANUAL: Reference manuals, technical manuals
- CIRCULAR: Company circulars, notices
"""


# =============================================================================
# QueryAnalyzer
# =============================================================================

class QueryAnalyzer:
    """Classify and analyze a user query in a single Flash Lite call.

    Replaces the scattered classification functions:
    - analyze_query_comprehensive()
    - _classify_query_intent_llm()
    - classify_retrieval_strategy()
    - _detect_doc_type_preference()
    - _detect_topic_shift_with_gemini()

    All analysis is consolidated into one structured LLM call, with
    deterministic derivation of downstream fields (merge strategy,
    decomposition need, regulatory standard detection) in Python.
    """

    MODEL = "gemini-2.5-flash-lite"

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def analyze(
        self,
        query: str,
        last_query: Optional[str] = None,
        last_answer_preview: Optional[str] = None,
        context_turn_count: int = 0,
    ) -> QueryAnalysis:
        """Fully analyze a query and return a QueryAnalysis object.

        Args:
            query: Current user query.
            last_query: Previous query text (for follow-up detection).
            last_answer_preview: Truncated previous answer.
            context_turn_count: How many turns deep we are in a conversation.

        Returns:
            QueryAnalysis with all fields populated.
        """
        has_context = context_turn_count > 0 and bool(last_query)

        # --- LLM classification ---
        raw = self._classify(query, last_query, last_answer_preview, has_context)

        # --- Parse intent ---
        intent = self._parse_intent(raw.get("intent", "new_query"), has_context)

        # --- Non-retrieval fast path ---
        if intent in NON_RETRIEVAL_INTENTS:
            direct_response = self._generate_non_retrieval_response(
                query, intent,
            )
            LOGGER.info(
                "QueryAnalyzer NON-RETRIEVAL: intent=%s response_len=%d",
                intent.value, len(direct_response) if direct_response else 0,
            )
            return QueryAnalysis(
                original_query=query,
                intent=intent,
                query_type=None,
                topic=raw.get("topic"),
                detected_sources=[],
                detected_doc_types=[],
                has_regulatory_standard=False,
                scope=ScopeAssessment.FOCUSED,
                retrieval_strategy=RetrievalStrategy.CHUNK,
                merge_strategy=MergeStrategy.DIRECT,
                requires_decomposition=False,
                doc_type_hint=None,
                direct_response=direct_response,
            )

        # --- Retrieval path: extract and derive fields ---
        query_type = self._parse_query_type(raw.get("query_type", "simple_factual"))
        topic = raw.get("topic") or None
        detected_sources = raw.get("detected_sources") or []
        detected_doc_types = raw.get("detected_doc_types") or []
        scope = self._parse_scope(raw.get("scope", "focused"))
        retrieval_strategy = self._parse_retrieval_strategy(
            raw.get("retrieval_strategy", "chunk"),
        )
        doc_type_hint = raw.get("doc_type_hint") or None

        # Topic inheritance: carry forward from previous turn if needed
        topic_inherited = False
        if not topic and has_context and intent in (
            Intent.FOLLOW_UP_SAME,
            Intent.FOLLOW_UP_NEW,
        ):
            # The LLM couldn't extract a topic from "What forms for that?"
            # but we know the conversation is about the previous topic.
            # The orchestrator will supply the inherited topic from state.
            # We flag it here so the caller knows to inject it.
            topic_inherited = True
            LOGGER.info(
                "QueryAnalyzer: Topic inheritance flagged — intent=%s, "
                "no topic extracted, will inherit from previous turn",
                intent.value,
            )

        # Derived fields (deterministic — no LLM needed)
        has_regulatory_standard = self._has_regulatory_standard(detected_doc_types)
        merge_strategy = self._derive_merge_strategy(query_type)
        requires_decomposition = self._needs_decomposition(query_type, scope)

        # Override retrieval strategy for compliance queries
        if query_type == QueryType.COMPLIANCE and requires_decomposition:
            retrieval_strategy = RetrievalStrategy.FILTERED_PARALLEL
            LOGGER.info(
                "QueryAnalyzer: COMPLIANCE override → retrieval_strategy=FILTERED_PARALLEL",
            )

        LOGGER.info(
            "QueryAnalyzer RESULT: intent=%s type=%s scope=%s merge=%s "
            "decompose=%s topic='%s' sources=%s doc_types=%s "
            "retrieval=%s has_standard=%s topic_inherited=%s",
            intent.value, query_type.value, scope.value, merge_strategy.value,
            requires_decomposition, topic, detected_sources, detected_doc_types,
            retrieval_strategy.value, has_regulatory_standard, topic_inherited,
        )

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            query_type=query_type,
            topic=topic,
            detected_sources=detected_sources,
            detected_doc_types=detected_doc_types,
            has_regulatory_standard=has_regulatory_standard,
            scope=scope,
            retrieval_strategy=retrieval_strategy,
            merge_strategy=merge_strategy,
            requires_decomposition=requires_decomposition,
            doc_type_hint=doc_type_hint,
            direct_response=None,
            topic_inherited=topic_inherited,
        )

    # -----------------------------------------------------------------
    # LLM call: single structured classification
    # -----------------------------------------------------------------

    def _classify(
        self,
        query: str,
        last_query: Optional[str],
        last_answer_preview: Optional[str],
        has_context: bool,
    ) -> Dict[str, Any]:
        """Single Flash Lite call that classifies everything we need."""

        context_block = ""
        follow_up_instruction = ""

        if has_context and last_query:
            context_block = f"""
PREVIOUS CONVERSATION:
User asked: "{last_query}"
Assistant answered: "{last_answer_preview}"

"""
            follow_up_instruction = """
  FOLLOW-UP DETECTION (only when previous conversation is provided):
  - "follow_up_same": User wants more detail on the SAME facet of the same topic.
    Examples: "Tell me more", "Can you elaborate?", "What about step 3?"
  - "follow_up_new": User stays on the same TOPIC but asks about a DIFFERENT facet.
    Examples: After asking about bunkering procedure → "What forms do I need for that?"
    After asking about drug testing → "What does RISQ say about this?"
  - "new_query": Completely different topic or operational area."""
        else:
            follow_up_instruction = """
  Note: No previous conversation context available. Use "new_query" for any maritime question."""

        prompt = f"""{context_block}CURRENT QUERY: "{query}"

Analyze this query for a maritime documentation assistant. Return a JSON object.

RESPONSE FORMAT (strict JSON, no markdown):
{{
    "intent": "<intent>",
    "query_type": "<type or null>",
    "topic": "<topic in 2-4 words, or null>",
    "detected_sources": ["<source1>", "<source2>"],
    "detected_doc_types": ["<doc_type1>", "<doc_type2>"],
    "scope": "<scope>",
    "retrieval_strategy": "<strategy>",
    "doc_type_hint": "<type or null>"
}}

FIELD DEFINITIONS:

1. "intent" — What the user wants:
  - "greeting": Hello, hi, good morning, etc.
  - "goodbye": Bye, thanks that's all, see you, etc.
  - "thank_you": Thanks, got it, perfect, roger that, cool beans, etc.
  - "off_topic": Non-maritime question (weather, jokes, personal questions, etc.)
  - "incomplete": Message appears cut off mid-sentence or is unintelligible fragments.
  - "new_query": New maritime question needing document search.
{follow_up_instruction}

2. "query_type" — Classification of the information need (null if non-retrieval intent):
  - "simple_factual": Asks for a specific fact, definition, or data point.
    Examples: "What is the SOPEP?", "Who signs the drug test form?", "What PPE for deck work?"
  - "procedural": Asks how to do something, steps, procedures, what-if scenarios.
    Examples: "How do I report an oil spill?", "What's the procedure for bunkering?", "What if we find stowaways?"
  - "multi_part": Asks multiple distinct questions in one message.
    Examples: "What forms for bunkering AND what's the Chief Engineer's role?"
  - "compliance": Compares, checks, or analyzes documents against each other.
    Examples: "Compare our D&A policy with Rightship", "Gap analysis on hot works: IMS vs RISQ"
    Also: "Difference between ISM and ISPS codes" (regulation vs regulation comparison)

3. "topic" — The query subject in 2-5 words.
   Examples: "bunkering operations", "drug and alcohol", "fire safety", "ice navigation", etc.
   If the query asks about specific forms of checklists, include their codes.
   Return null if no clear maritime topic.

4. "detected_sources" — Specific document sources mentioned or implied in the query.
   Return empty list [] if none detected. Match against these known patterns:
{SOURCE_PATTERNS}

5. "detected_doc_types" — Document types referenced or implied. Use the type labels above.
   Examples: If user mentions "RISQ" → include "VETTING". If "our procedures" → include "PROCEDURE".
   Return empty list [] if none detected.

6. "scope" — How broad is this query?
  - "focused": Specific topic, specific section, answerable in one response.
  - "broad": Wide topic but partially answerable. Example: "Compare our maintenance with RISQ"
  - "too_broad": Entire document systems, no topic filter. Example: "Full gap analysis IMS vs RISQ"

7. "retrieval_strategy" — How to search:
  - "chunk": Specific facts, definitions, discrete details.
  - "section": Procedural queries needing complete instructions/steps/context.
  - "filtered_parallel": Comparison/compliance queries needing balanced multi-source retrieval.

8. "doc_type_hint" — If the user explicitly asks for a specific type:
   "Form", "Checklist", "Procedure", "Regulation", "Circular", "Vetting" or null.

RULES:
- Return ONLY valid JSON, no markdown code blocks, no explanation.
- For non-retrieval intents, still provide best-guess values for all fields.
- "how to" / "procedure" / "steps" / "what if" → retrieval_strategy = "section"
- Comparison / gap analysis / "vs" / "compare" → retrieval_strategy = "filtered_parallel"
- When in doubt about retrieval_strategy, default to "chunk".
- detected_sources and detected_doc_types must be arrays, even if empty.

JSON Response:"""

        try:
            config = AppConfig.get()
            response = config.client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            result = json_module.loads(response.text)
            LOGGER.info(
                "QueryAnalyzer: intent=%s type=%s topic=%s scope=%s sources=%s",
                result.get("intent"),
                result.get("query_type"),
                result.get("topic"),
                result.get("scope"),
                result.get("detected_sources"),
            )
            return result

        except Exception as exc:
            LOGGER.warning("QueryAnalyzer classification failed: %s — using fallback", exc)
            return self._fallback_classification(query)

    # -----------------------------------------------------------------
    # Non-retrieval response generation
    # -----------------------------------------------------------------

    def _generate_non_retrieval_response(
        self,
        query: str,
        intent: Intent,
    ) -> str:
        """Generate a natural response for non-retrieval intents.

        Uses a lightweight Flash Lite call for personality. Falls back
        to templates if the call fails.
        """
        persona_prompts = {
            Intent.GREETING: (
                f'The user said: "{query}"\n'
                "You are a friendly maritime documentation assistant. "
                "Greet them warmly and briefly mention you can help with "
                "ship procedures, safety protocols, forms, and regulations. "
                "Keep it to 1-2 sentences."
            ),
            Intent.GOODBYE: (
                f'The user said: "{query}"\n'
                "You are a maritime documentation assistant. "
                "Say a friendly goodbye. Keep it to 1 sentence."
            ),
            Intent.THANK_YOU: (
                f'The user said: "{query}"\n'
                "You are a maritime documentation assistant. "
                "Acknowledge their thanks warmly and let them know "
                "you're here if they need anything else. Keep it to 1 sentence."
            ),
            Intent.OFF_TOPIC: (
                f'The user said: "{query}"\n'
                "You are a specialized maritime documentation assistant. "
                "Politely redirect them to maritime topics. Mention you can "
                "help with procedures, safety, forms, and regulations. "
                "Keep it friendly and to 1-2 sentences."
            ),
            Intent.INCOMPLETE: (
                f'The user said: "{query}"\n'
                "The user's message appears to be cut off or incomplete. "
                "Politely ask them to resend their complete question. "
                "Keep it to 1 sentence."
            ),
        }

        persona_prompt = persona_prompts.get(intent)
        if not persona_prompt:
            return self._template_response(intent)

        try:
            config = AppConfig.get()
            response = config.client.models.generate_content(
                model=self.MODEL,
                contents=persona_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )
            text = response.text.strip()
            if text:
                return text
        except Exception as exc:
            LOGGER.warning(
                "Non-retrieval response generation failed: %s — using template", exc,
            )

        return self._template_response(intent)

    @staticmethod
    def _template_response(intent: Intent) -> str:
        """Fallback templates when LLM generation fails."""
        templates = {
            Intent.GREETING: (
                "Hello! I'm here to help you with maritime documentation "
                "and procedures. What would you like to know?"
            ),
            Intent.GOODBYE: "Take care! Feel free to return anytime you need assistance.",
            Intent.THANK_YOU: "You're welcome! Let me know if you need anything else.",
            Intent.OFF_TOPIC: (
                "I'm a specialized maritime documentation assistant. "
                "I can help with ship procedures, safety protocols, forms, "
                "and regulations — feel free to ask!"
            ),
            Intent.INCOMPLETE: (
                "It looks like your message may have been cut off. "
                "Could you resend your complete question?"
            ),
        }
        return templates.get(intent, templates[Intent.GREETING])

    # -----------------------------------------------------------------
    # Parsing helpers (LLM output → typed values)
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_intent(raw_intent: str, has_context: bool) -> Intent:
        """Map LLM string output to Intent enum with validation."""
        mapping = {
            "greeting": Intent.GREETING,
            "goodbye": Intent.GOODBYE,
            "thank_you": Intent.THANK_YOU,
            "off_topic": Intent.OFF_TOPIC,
            "chitchat": Intent.OFF_TOPIC,       # legacy alias
            "incomplete": Intent.INCOMPLETE,
            "follow_up_same": Intent.FOLLOW_UP_SAME,
            "follow_up_new": Intent.FOLLOW_UP_NEW,
            "follow_up": Intent.FOLLOW_UP_SAME,  # legacy alias → default to same
            "clarification": Intent.FOLLOW_UP_SAME,  # legacy alias
            "new_query": Intent.NEW_QUERY,
        }
        intent = mapping.get(raw_intent.lower().strip())

        if intent is None:
            LOGGER.warning("Unknown intent '%s', defaulting to NEW_QUERY", raw_intent)
            return Intent.NEW_QUERY

        # Guard: follow-up intents only make sense with context
        if intent in (Intent.FOLLOW_UP_SAME, Intent.FOLLOW_UP_NEW) and not has_context:
            LOGGER.info("Follow-up intent without context, treating as NEW_QUERY")
            return Intent.NEW_QUERY

        return intent

    @staticmethod
    def _parse_query_type(raw_type: str) -> QueryType:
        mapping = {
            "simple_factual": QueryType.SIMPLE_FACTUAL,
            "procedural": QueryType.PROCEDURAL,
            "multi_part": QueryType.MULTI_PART,
            "compliance": QueryType.COMPLIANCE,
        }
        qt = mapping.get(raw_type.lower().strip() if raw_type else "simple_factual")
        if qt is None:
            LOGGER.warning("Unknown query_type '%s', defaulting to SIMPLE_FACTUAL", raw_type)
            return QueryType.SIMPLE_FACTUAL
        return qt

    @staticmethod
    def _parse_scope(raw_scope: str) -> ScopeAssessment:
        mapping = {
            "focused": ScopeAssessment.FOCUSED,
            "broad": ScopeAssessment.BROAD,
            "too_broad": ScopeAssessment.TOO_BROAD,
        }
        scope = mapping.get(raw_scope.lower().strip() if raw_scope else "focused")
        if scope is None:
            LOGGER.warning("Unknown scope '%s', defaulting to FOCUSED", raw_scope)
            return ScopeAssessment.FOCUSED
        return scope

    @staticmethod
    def _parse_retrieval_strategy(raw_strategy: str) -> RetrievalStrategy:
        mapping = {
            "chunk": RetrievalStrategy.CHUNK,
            "chunk_level": RetrievalStrategy.CHUNK,     # legacy alias
            "section": RetrievalStrategy.SECTION,
            "section_level": RetrievalStrategy.SECTION,  # legacy alias
            "filtered_parallel": RetrievalStrategy.FILTERED_PARALLEL,
        }
        rs = mapping.get(raw_strategy.lower().strip() if raw_strategy else "chunk")
        if rs is None:
            LOGGER.warning("Unknown retrieval_strategy '%s', defaulting to CHUNK", raw_strategy)
            return RetrievalStrategy.CHUNK
        return rs

    # -----------------------------------------------------------------
    # Derivation helpers (deterministic, no LLM)
    # -----------------------------------------------------------------

    @staticmethod
    def _has_regulatory_standard(detected_doc_types: List[str]) -> bool:
        """True if any detected doc type is a regulatory/vetting standard."""
        regulatory_types = {"REGULATION", "VETTING"}
        return any(dt.upper() in regulatory_types for dt in detected_doc_types)

    @staticmethod
    def _derive_merge_strategy(query_type: Optional[QueryType]) -> MergeStrategy:
        if query_type == QueryType.COMPLIANCE:
            return MergeStrategy.COMPLIANCE
        if query_type == QueryType.MULTI_PART:
            return MergeStrategy.SYNTHESIZE
        return MergeStrategy.DIRECT

    @staticmethod
    def _needs_decomposition(
        query_type: Optional[QueryType],
        scope: ScopeAssessment,
    ) -> bool:
        """Decomposition needed for complex types, unless scope blocks it."""
        if scope == ScopeAssessment.TOO_BROAD:
            return False  # Blocked — will push back on scope instead
        return query_type in (QueryType.MULTI_PART, QueryType.COMPLIANCE)

    # -----------------------------------------------------------------
    # Fallback classification (no LLM)
    # -----------------------------------------------------------------

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Heuristic fallback when the LLM call fails entirely."""
        query_lower = query.strip().lower()

        # Simple greeting detection
        greetings = {
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "hey there",
        }
        if query_lower in greetings or (len(query.split()) <= 2 and "?" not in query):
            return {"intent": "greeting"}

        # Simple goodbye
        goodbyes = {"bye", "goodbye", "see you", "take care"}
        if query_lower in goodbyes:
            return {"intent": "goodbye"}

        # Default: treat as new query with minimal info
        return {
            "intent": "new_query",
            "query_type": "simple_factual",
            "topic": None,
            "detected_sources": [],
            "detected_doc_types": [],
            "scope": "focused",
            "retrieval_strategy": "chunk",
            "doc_type_hint": None,
        }


# =============================================================================
# FilteredRetriever (Phase 5b)
# =============================================================================

class FilteredRetriever:
    """Execute retrieval with source filtering and parallel sub-queries.

    Wraps the existing FTS5 and vector retrievers, adding:
    - doc_type and title metadata filtering (pushed into retrievers)
    - Reciprocal rank fusion (vector + BM25 per sub-query)
    - Cohere reranking (always-on, per sub-query)
    - Parallel execution of multiple sub-queries (outer parallelism only)

    Pipeline per sub-query (sequential):
        FTS5 filtered → Vector filtered → RRF fusion → Cohere rerank

    Multiple sub-queries run in parallel via ThreadPoolExecutor.
    """

    RERANK_MODEL = "rerank-v3.5"
    MAX_WORKERS = 4

    def __init__(self, app_state: "AppState"):
        self.app_state = app_state
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        self._cohere_client = self._init_cohere()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def retrieve_single(self, sub_query: SubQuery) -> RetrievalResult:
        """Run filtered hybrid retrieval for a single sub-query.

        Sequential inner pipeline:
        FTS5 (filtered) → Vector (filtered) → RRF → Cohere rerank.
        """
        fts_retriever = self.app_state.fts5_retriever
        vector_retriever = self.app_state.vector_retriever

        if not fts_retriever or not vector_retriever:
            LOGGER.error("Retrievers not initialized — returning empty result")
            return RetrievalResult(
                sub_query=sub_query,
                nodes=[],
                source_label=sub_query.source_label,
            )

        # 1. BM25 keyword search (filtered)
        bm25_nodes = fts_retriever.retrieve_filtered(
            query_str=sub_query.text,
            top_k=sub_query.top_k,
            doc_type_filter=sub_query.doc_type_filter,
            title_filter=sub_query.title_filter,
        )

        # 2. Vector similarity search (filtered)
        vector_nodes = vector_retriever.retrieve_filtered(
            query_str=sub_query.text,
            top_k=sub_query.top_k,
            doc_type_filter=sub_query.doc_type_filter,
            title_filter=sub_query.title_filter,
        )

        # 3. Reciprocal rank fusion
        fused = reciprocal_rank_fusion(
            vector_nodes, bm25_nodes,
            k=60,
            top_k=sub_query.top_k,
        )

        # 4. Rerank (always-on)
        reranked = self._rerank(fused, sub_query.text, top_n=sub_query.top_k)

        LOGGER.info(
            "FilteredRetriever [%s]: %d BM25 + %d vector → %d fused → %d reranked",
            sub_query.source_label,
            len(bm25_nodes),
            len(vector_nodes),
            len(fused),
            len(reranked),
        )

        return RetrievalResult(
            sub_query=sub_query,
            nodes=reranked,
            source_label=sub_query.source_label,
        )

    def retrieve_parallel(
        self, sub_queries: List[SubQuery],
    ) -> List[RetrievalResult]:
        """Run multiple filtered retrievals in parallel.

        Each sub-query executes its own sequential hybrid pipeline.
        Outer parallelism only — keeps thread management simple and
        avoids nested executor issues.

        Returns results in the same order as ``sub_queries``.
        """
        if not sub_queries:
            return []

        LOGGER.info(
            "FilteredRetriever PARALLEL: starting %d sub-queries: %s",
            len(sub_queries),
            [(sq.source_label, sq.doc_type_filter) for sq in sub_queries],
        )
        t0 = time.time()

        if len(sub_queries) == 1:
            results = [self.retrieve_single(sub_queries[0])]
        else:
            # Map futures to their original index to preserve order
            future_to_idx = {}
            for idx, sq in enumerate(sub_queries):
                future = self._executor.submit(self.retrieve_single, sq)
                future_to_idx[future] = idx

            results: List[Optional[RetrievalResult]] = [None] * len(sub_queries)

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                sq = sub_queries[idx]
                try:
                    results[idx] = future.result(timeout=30.0)
                except Exception as exc:
                    LOGGER.error(
                        "Sub-query retrieval failed for '%s': %s",
                        sq.source_label, exc,
                    )
                    results[idx] = RetrievalResult(
                        sub_query=sq,
                        nodes=[],
                        source_label=sq.source_label,
                    )

        elapsed = time.time() - t0
        total_nodes = sum(len(r.nodes) for r in results if r)
        LOGGER.info(
            "FilteredRetriever PARALLEL COMPLETE: %d sub-queries, "
            "%d total nodes, %.1fs",
            len(sub_queries), total_nodes, elapsed,
        )

        return results  # type: ignore[return-value]

    def retrieve_unfiltered(
        self, query: str, top_k: int = 20,
    ) -> List[NodeWithScore]:
        """Run standard hybrid retrieval with no source filters.

        Convenience method for simple queries that skip decomposition.
        Still benefits from RRF fusion and Cohere reranking.
        """
        result = self.retrieve_single(
            SubQuery(text=query, source_label="general", top_k=top_k),
        )
        return result.nodes

    # -----------------------------------------------------------------
    # Cohere reranking
    # -----------------------------------------------------------------

    @staticmethod
    def _init_cohere():
        """Initialize Cohere client for reranking."""
        try:
            import cohere
            api_key = os.getenv("COHERE_API_KEY")
            if api_key:
                return cohere.ClientV2(api_key)
            LOGGER.warning("COHERE_API_KEY not set — reranking disabled")
        except ImportError:
            LOGGER.warning("cohere package not installed — reranking disabled")
        return None

    def _rerank(
        self,
        nodes: List[NodeWithScore],
        query: str,
        top_n: int = 20,
    ) -> List[NodeWithScore]:
        """Rerank nodes via Cohere. Falls back to original order on failure."""
        if not nodes or not self._cohere_client:
            return nodes

        try:
            documents = [node.node.text[:1000] for node in nodes]
            rerank_response = self._cohere_client.rerank(
                model=self.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(len(documents), top_n),
            )

            reranked = [
                NodeWithScore(
                    node=nodes[r.index].node,
                    score=r.relevance_score,
                )
                for r in rerank_response.results
            ]
            return reranked

        except Exception as exc:
            LOGGER.warning("Cohere rerank failed: %s — using original order", exc)
            return nodes


# =============================================================================
# QueryDecomposer (Phase 5c)
# =============================================================================

class DecompositionError(Exception):
    """Raised when query decomposition fails after retry."""
    pass


class QueryDecomposer:
    """Break complex queries into targeted sub-queries for filtered retrieval.

    Handles two decomposition patterns via a single Flash prompt:
    - MULTI_PART: splits independent sub-questions
    - COMPLIANCE: identifies comparison sources with doc_type filters

    For simple queries (requires_decomposition=False), provides
    ``as_single_query()`` to wrap into a single SubQuery without an LLM call.
    """

    MODEL = "gemini-2.5-flash"
    MAX_SUB_QUERIES = 4

    # Scope → top_k mapping
    TOP_K_BY_SCOPE = {
        ScopeAssessment.FOCUSED: 20,
        ScopeAssessment.BROAD: 25,
        ScopeAssessment.TOO_BROAD: 20,  # shouldn't reach decomposer, but safe default
    }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def decompose(self, analysis: QueryAnalysis) -> List[SubQuery]:
        """Decompose a complex query into sub-queries.

        Should only be called when ``analysis.requires_decomposition`` is True.
        Raises ``DecompositionError`` if decomposition fails after one retry.

        Args:
            analysis: Fully populated QueryAnalysis from the analyzer.

        Returns:
            List of SubQuery objects (1 to MAX_SUB_QUERIES).

        Raises:
            DecompositionError: If LLM decomposition fails twice.
        """
        top_k = self.TOP_K_BY_SCOPE.get(analysis.scope, 20)

        # Try decomposition with one retry
        for attempt in range(2):
            try:
                raw_sub_queries = self._call_llm(analysis)
                sub_queries = self._parse_sub_queries(raw_sub_queries, top_k)

                if not sub_queries:
                    LOGGER.warning(
                        "Decomposition attempt %d returned empty results", attempt + 1,
                    )
                    continue

                # Cap check — push back if too many
                if len(sub_queries) > self.MAX_SUB_QUERIES:
                    LOGGER.info(
                        "Decomposition produced %d sub-queries (max %d) — pushing back",
                        len(sub_queries), self.MAX_SUB_QUERIES,
                    )
                    raise DecompositionError(
                        "Your question covers too many areas to answer well in one go. "
                        "Could you narrow it down to focus on fewer topics or sources? "
                        f"I can handle up to {self.MAX_SUB_QUERIES} sub-topics at a time."
                    )

                LOGGER.info(
                    "Decomposed '%s' into %d sub-queries: %s",
                    analysis.original_query[:60],
                    len(sub_queries),
                    [sq.source_label for sq in sub_queries],
                )
                return sub_queries

            except DecompositionError:
                raise  # Don't retry on intentional push-back
            except Exception as exc:
                LOGGER.warning(
                    "Decomposition attempt %d failed: %s", attempt + 1, exc,
                )

        raise DecompositionError(
            "I had trouble breaking down your question. "
            "Could you try rephrasing it or asking one part at a time?"
        )

    def as_single_query(self, analysis: QueryAnalysis) -> List[SubQuery]:
        """Wrap a non-decomposed query into a single SubQuery.

        Used when ``requires_decomposition`` is False.
        No LLM call — purely deterministic construction.
        """
        top_k = self.TOP_K_BY_SCOPE.get(analysis.scope, 20)

        # Use doc_type_filter if the analyzer detected specific types
        doc_type_filter = analysis.detected_doc_types if analysis.detected_doc_types else None

        return [
            SubQuery(
                text=analysis.original_query,
                source_label=analysis.topic or "general query",
                doc_type_filter=None,
                title_filter=None,
                is_standard=False,
                top_k=top_k,
            )
        ]

    # -----------------------------------------------------------------
    # LLM decomposition call
    # -----------------------------------------------------------------

    def _call_llm(self, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """Single Flash call to decompose the query."""

        # Build context about what we already know
        sources_info = ""
        if analysis.detected_sources:
            sources_info = f"\nDetected sources in the query: {', '.join(analysis.detected_sources)}"

        doc_types_info = ""
        if analysis.detected_doc_types:
            doc_types_info = f"\nDetected document types: {', '.join(analysis.detected_doc_types)}"

        prompt = f"""You are a query decomposition engine for a maritime documentation system.

ORIGINAL QUERY: "{analysis.original_query}"
QUERY TYPE: {analysis.query_type.value if analysis.query_type else "unknown"}
TOPIC: {analysis.topic or "not identified"}{sources_info}{doc_types_info}

YOUR TASK: Break this query into independent sub-queries for parallel retrieval.

{"=" * 60}

{"MULTI_PART INSTRUCTIONS" if analysis.query_type == QueryType.MULTI_PART else "COMPLIANCE INSTRUCTIONS"}

{self._multi_part_instructions() if analysis.query_type == QueryType.MULTI_PART else self._compliance_instructions()}

{"=" * 60}

AVAILABLE DOC_TYPE VALUES for filtering:
- "REGULATION": International conventions/codes (MARPOL, SOLAS, ISM, ISPS, etc.)
- "VETTING": Inspection questionnaires/standards (RISQ, TMSA, SIRE, CDI)
- "PROCEDURE": Company operational procedures
- "POLICY": Company policies
- "FORM": Standardized forms
- "CHECKLIST": Operational checklists
- "MANUAL": Reference/technical manuals
- "CIRCULAR": Company circulars/notices

KNOWN MARITIME SOURCES (use these to determine doc_type filters):
{SOURCE_PATTERNS}

RESPONSE FORMAT — return a JSON array of sub-query objects:
[
    {{
        "text": "Clear, specific retrieval query text",
        "source_label": "Short human-readable label (2-5 words)",
        "doc_type_filter": ["TYPE1", "TYPE2"] or null,
        "is_standard": false
    }}
]

RULES:
- Return ONLY valid JSON, no markdown, no explanation.
- Each sub-query "text" should be a clear, self-contained search query.
- "source_label" is a concise description shown to the user (e.g. "Company D&A policy", "RISQ hot work requirements").
- "doc_type_filter" narrows retrieval to specific document types. Use null for no filter.
- Maximum {self.MAX_SUB_QUERIES} sub-queries. If the question needs more, return the {self.MAX_SUB_QUERIES} most important.
- Sub-queries must be distinct — no overlapping retrieval targets.

JSON Response:"""

        config = AppConfig.get()
        response = config.client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        result = json_module.loads(response.text)

        LOGGER.info(
            "QueryDecomposer RAW LLM response: %s",
            json_module.dumps(result, indent=None)[:500],
        )

        # Accept both a raw list and a wrapped object
        if isinstance(result, dict):
            result = result.get("sub_queries") or result.get("queries") or []
        if not isinstance(result, list):
            LOGGER.warning("Decomposer returned non-list: %s", type(result))
            return []

        return result

    # -----------------------------------------------------------------
    # Instruction blocks
    # -----------------------------------------------------------------

    @staticmethod
    def _multi_part_instructions() -> str:
        return """The user asked multiple distinct questions in one message.
Split them into independent sub-queries. Each sub-query should:
- Address exactly ONE of the user's questions
- Be self-contained (include enough context to retrieve without the other sub-queries)
- Have a descriptive source_label

Example:
Query: "What forms do I need for bunkering and what is the Chief Engineer's role in drills?"
→ Two sub-queries:
  1. text="forms required for bunkering operations", source_label="Bunkering forms"
  2. text="Chief Engineer role responsibilities emergency drills", source_label="CE drill duties"

For multi-part queries, doc_type_filter is usually null (retrieve from all sources)
unless a specific sub-question clearly targets a specific document type."""

    @staticmethod
    def _compliance_instructions() -> str:
        return """The user wants to compare, check compliance, or do gap analysis between sources.
Create one sub-query PER SOURCE being compared. Each sub-query should:
- Target the SAME topic but from a DIFFERENT source
- Use doc_type_filter to isolate the source
- Set is_standard=true for the regulatory/vetting benchmark (the standard being checked against)
- Set is_standard=false for the company's own documents

Example:
Query: "Compare our hot work procedures with RISQ requirements"
→ Two sub-queries:
  1. text="hot work safety procedures and requirements", source_label="Company hot work procedures",
     doc_type_filter=["PROCEDURE", "POLICY"], is_standard=false
  2. text="hot work safety requirements and standards", source_label="RISQ hot work requirements",
     doc_type_filter=["VETTING"], is_standard=true

If comparing two regulations (e.g. ISM vs ISPS), neither is "the standard" — set both to is_standard=false.
If comparing company docs against a regulation, the regulation is the standard."""

    # -----------------------------------------------------------------
    # Parsing and validation
    # -----------------------------------------------------------------

    def _parse_sub_queries(
        self,
        raw: List[Dict[str, Any]],
        top_k: int,
    ) -> List[SubQuery]:
        """Convert raw LLM output dicts into validated SubQuery objects."""
        sub_queries = []

        for item in raw:
            if not isinstance(item, dict):
                LOGGER.warning("Skipping non-dict sub-query item: %s", item)
                continue

            text = (item.get("text") or "").strip()
            if not text:
                LOGGER.warning("Skipping sub-query with empty text")
                continue

            source_label = (item.get("source_label") or "sub-query").strip()

            # Validate doc_type_filter values
            doc_type_filter = item.get("doc_type_filter")
            if doc_type_filter:
                valid_types = {
                    "REGULATION", "VETTING", "PROCEDURE", "POLICY",
                    "FORM", "CHECKLIST", "MANUAL", "CIRCULAR",
                }
                doc_type_filter = [
                    dt.upper() for dt in doc_type_filter
                    if isinstance(dt, str) and dt.upper() in valid_types
                ]
                if not doc_type_filter:
                    doc_type_filter = None

            is_standard = bool(item.get("is_standard", False))

            sub_queries.append(SubQuery(
                text=text,
                source_label=source_label,
                doc_type_filter=doc_type_filter,
                title_filter=None,  # Not used in Phase 5c (see decision 5c-D3)
                is_standard=is_standard,
                top_k=top_k,
            ))

        return sub_queries


# =============================================================================
# ResultSynthesizer (Phase 5d)
# =============================================================================

@dataclass
class SubAnswer:
    """A complete answer generated from one sub-query's retrieval results."""

    source_label: str
    text: str
    nodes: List[NodeWithScore]
    confidence_pct: int
    confidence_level: str
    confidence_note: str
    is_standard: bool = False


class ResultSynthesizer:
    """Generate per-sub-query answers and synthesize across them.

    Two operating modes:
    - **DIRECT (simple queries):** Stream a single answer from retrieved nodes.
      Identical to the current RAG answer generation flow.
    - **Complex queries (SYNTHESIZE / COMPLIANCE):** Generate blocking sub-answers
      for each retrieval result, then stream a synthesis that merges them.

    The per-sub-query answers use an extracted version of the existing RAG prompt.
    Synthesis uses strategy-specific prompts tailored to each merge pattern.
    """

    MODEL = "gemini-2.5-flash"
    THINKING_BUDGET_STANDARD = 1024
    THINKING_BUDGET_GAP_ANALYSIS = 4096

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_sub_answer(
        self,
        sub_query: SubQuery,
        nodes: List[NodeWithScore],
    ) -> SubAnswer:
        """Generate a focused answer for one sub-query (blocking).

        Called once per sub-query in the complex path. The user does NOT
        see these directly — they feed into the synthesis step.
        """
        t0 = time.time()

        if not nodes:
            LOGGER.warning(
                "ResultSynthesizer: No nodes for sub-query '%s' — returning empty",
                sub_query.source_label,
            )
            return SubAnswer(
                source_label=sub_query.source_label,
                text="No relevant information found for this topic.",
                nodes=[],
                confidence_pct=0,
                confidence_level="LOW 🔴",
                confidence_note="No results retrieved",
                is_standard=sub_query.is_standard,
            )

        confidence_pct, confidence_level, confidence_note = self._calculate_confidence(nodes)
        context = self._build_context_from_nodes(nodes, numbered=False)
        prompt = self._build_sub_answer_prompt(sub_query, context, confidence_pct)

        try:
            config = AppConfig.get()
            response = config.client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=ThinkingConfig(
                        thinking_budget=self.THINKING_BUDGET_STANDARD,
                    ),
                ),
            )
            answer_text = response.text.strip()
        except Exception as exc:
            LOGGER.error(
                "ResultSynthesizer: Sub-answer generation failed for '%s': %s",
                sub_query.source_label, exc,
            )
            answer_text = "Failed to generate answer for this section."

        elapsed = time.time() - t0
        LOGGER.info(
            "ResultSynthesizer SUB-ANSWER [%s]: %d chars, confidence=%d%% (%s), "
            "%d nodes, %.1fs",
            sub_query.source_label,
            len(answer_text),
            confidence_pct,
            confidence_level,
            len(nodes),
            elapsed,
        )

        return SubAnswer(
            source_label=sub_query.source_label,
            text=answer_text,
            nodes=nodes,
            confidence_pct=confidence_pct,
            confidence_level=confidence_level,
            confidence_note=confidence_note,
            is_standard=sub_query.is_standard,
        )

    def generate_direct_stream(
        self,
        query: str,
        nodes: List[NodeWithScore],
        conversation_history: str = "",
        confidence_pct: int = 0,
    ) -> Generator[str, None, None]:
        """Stream a single answer directly from nodes (DIRECT merge path).

        This is the simple-query path — functionally identical to the
        current ``generate_answer_stream()`` but routed through the
        orchestrator.
        """
        context = self._build_context_from_nodes(nodes, numbered=True)
        confidence_instruction = self._confidence_instruction(confidence_pct)

        prompt = f"""You are a maritime safety assistant for ship crew.

CONFIDENCE CONTEXT: {confidence_instruction}

{conversation_history}

CRITICAL RULES:
- You ALWAYS answer in English, even if asked in another language.
- Answer facts must come ONLY from the provided documents, but you can use logic, general well-established facts of life, and common sense in crafting your replies.
- General maritime knowledge needed to make logical connections between the provided documents can be used from your internal knowledge base.
- If information is missing or unclear, say so explicitly.
- When referencing forms, ALWAYS include the complete form name/number (e.g., "Form_DA 005 Drug and Alcohol Test Report").
- Cite sources using [Source number, Source title > Section] format (e.g. [3, Chapter 5 MASTER'S RESPONSIBILITY AND AUTHORITY > Master's Overriding Authority]). Don't use filenames as title, unless no actual title is available.
- Keep answers concise (3-4 minute read maximum).
- Use "Based on the documents on file..."
- If the question asks "which form" or "what form", or "what checklist", start your answer with the exact form or checklist name.
- If the question asks for advice, you can synthesize by including common sense/logic/knowledge of the world.
- You are allowed to draft messages on behalf of the user, if so asked.
- When you can infer something from the context, you can do so, giving a 'heads up' like "it can be assumed that..." or "we can infer from this that..." or likewise.
- ALWAYS double check that you have exhausted all available information from the context provided and ensure not to omit any detail that could relate to the query.
- When asked to list details from a Form or Checklist, ensure that you don't omit any part or section.
- When making a table, don't make a column for the sources, instead place them after the table.
- If this is a followup question referring to previous conversation (using "it", "that", "this", etc.), use the conversation history above to understand the context.

CONTEXT:
{context}

QUESTION: {query}

Please provide a clear, concise answer with proper citations."""

        LOGGER.info(
            "ResultSynthesizer DIRECT STREAM: query='%s' nodes=%d confidence=%d%%",
            query[:60], len(nodes), confidence_pct,
        )

        yield from self._stream_gemini(prompt, self.THINKING_BUDGET_STANDARD)

    def generate_synthesis_stream(
        self,
        original_query: str,
        sub_answers: List[SubAnswer],
        merge_strategy: MergeStrategy,
        has_regulatory_standard: bool = False,
    ) -> Generator[str, None, None]:
        """Stream the final synthesized answer from multiple sub-answers.

        Called after all sub-answers are generated (complex path only).
        Uses strategy-specific prompts for different merge patterns.
        """
        prompt = self._build_synthesis_prompt(
            original_query, sub_answers, merge_strategy, has_regulatory_standard,
        )

        # Gap analysis gets more thinking budget
        thinking_budget = self.THINKING_BUDGET_STANDARD
        if merge_strategy == MergeStrategy.COMPLIANCE:
            thinking_budget = self.THINKING_BUDGET_GAP_ANALYSIS

        LOGGER.info(
            "ResultSynthesizer SYNTHESIS STREAM: strategy=%s standard=%s "
            "sub_answers=%d thinking_budget=%d",
            merge_strategy.value,
            has_regulatory_standard,
            len(sub_answers),
            thinking_budget,
        )

        yield from self._stream_gemini(prompt, thinking_budget)

    # -----------------------------------------------------------------
    # Context and prompt builders
    # -----------------------------------------------------------------

    @staticmethod
    def _build_context_from_nodes(
        nodes: List[NodeWithScore],
        numbered: bool = True,
    ) -> str:
        """Format retrieved nodes into a context string for the LLM.

        Args:
            nodes: Ranked nodes from retrieval.
            numbered: If True, prefix each chunk with a number (for DIRECT
                path source citations). If False, use descriptive labels
                (for sub-answers that feed into synthesis).
        """
        parts = []
        for i, node in enumerate(nodes[:15], 1):
            metadata = node.node.metadata
            source = metadata.get("source", "Unknown")
            section = metadata.get("section", "N/A")
            title = metadata.get("title", "")

            if numbered:
                label = f"[{i}: {title or source} > {section}]"
            else:
                # Descriptive label for synthesis path — no collision risk
                label = f"[{title or source} > {section}]"

            parts.append(f"{label}\n{node.node.text}")

        return "\n\n".join(parts)

    def _build_sub_answer_prompt(
        self,
        sub_query: SubQuery,
        context: str,
        confidence_pct: int,
    ) -> str:
        """Build prompt for a single sub-query answer (blocking call)."""
        confidence_instruction = self._confidence_instruction(confidence_pct)

        return f"""You are a maritime documentation expert answering a focused question.

SOURCE SCOPE: You are answering specifically about: {sub_query.source_label}

CONFIDENCE CONTEXT: {confidence_instruction}

CRITICAL RULES:
- Answer facts must come ONLY from the provided documents.
- General maritime knowledge can be used for logical connections.
- If information is missing or unclear, say so explicitly.
- When referencing forms, include the complete form name/number.
- Cite sources using descriptive format: [Document Title > Section] (e.g. [Chapter 5 MASTER'S RESPONSIBILITY > Master's Overriding Authority]).
- Do NOT use numbered source references — use descriptive titles only.
- Keep the answer focused and thorough for the specific question.
- ALWAYS double check that you have exhausted all available information.
- When asked to list details from a Form or Checklist, ensure completeness.

CONTEXT:
{context}

QUESTION: {sub_query.text}

Provide a focused, thorough answer based on the documents above."""

    def _build_synthesis_prompt(
        self,
        original_query: str,
        sub_answers: List[SubAnswer],
        merge_strategy: MergeStrategy,
        has_regulatory_standard: bool,
    ) -> str:
        """Build the strategy-specific synthesis prompt."""

        # Format sub-answers with their labels
        sub_answer_blocks = []
        for sa in sub_answers:
            standard_tag = " (REGULATORY STANDARD)" if sa.is_standard else ""
            sub_answer_blocks.append(
                f"--- {sa.source_label}{standard_tag} ---\n{sa.text}"
            )
        joined_answers = "\n\n".join(sub_answer_blocks)

        if merge_strategy == MergeStrategy.SYNTHESIZE:
            return self._synthesis_prompt_multi_part(original_query, joined_answers)
        elif merge_strategy == MergeStrategy.COMPLIANCE:
            if has_regulatory_standard:
                return self._synthesis_prompt_gap_analysis(original_query, joined_answers)
            else:
                return self._synthesis_prompt_comparison(original_query, joined_answers)
        else:
            # DIRECT shouldn't reach here, but handle gracefully
            return self._synthesis_prompt_multi_part(original_query, joined_answers)

    @staticmethod
    def _synthesis_prompt_multi_part(query: str, sub_answers_text: str) -> str:
        return f"""You are a maritime safety assistant synthesizing a comprehensive answer.

The user asked a multi-part question. Below are focused answers for each part.
Your job is to weave them into a SINGLE coherent response that flows naturally.

RULES:
- Preserve ALL specific details, section references, form numbers, and citations from each sub-answer.
- Do not drop or summarize away any factual content.
- Create smooth transitions between topics.
- Use the source citations as they appear in the sub-answers (descriptive format like [Document > Section]).
- Keep the combined answer concise but complete (3-5 minute read maximum).
- If any sub-answer notes missing information, preserve that caveat.
- At the end, list all sources referenced, grouped by topic.

SUB-ANSWERS:
{sub_answers_text}

ORIGINAL QUESTION: {query}

Provide a unified, well-structured response that addresses all parts of the question."""

    @staticmethod
    def _synthesis_prompt_gap_analysis(query: str, sub_answers_text: str) -> str:
        return f"""You are a maritime compliance expert performing a gap analysis.

The user wants to check company documentation against a regulatory standard.
Below are the findings from each source. The source marked (REGULATORY STANDARD)
is the benchmark — the company's documents should be checked against it.

YOUR TASK: Produce a structured gap analysis with these sections:

1. **Summary**: Brief overview of the comparison scope.
2. **Fully Covered**: Requirements from the standard that the company documents adequately address. Cite specific sections from both sources.
3. **Partially Covered**: Requirements that are addressed but incomplete, vague, or lacking detail compared to the standard. Explain what's missing.
4. **Gaps Identified**: Requirements from the standard that are NOT addressed at all in the company documents. These are the critical findings.
5. **Additional Items**: Anything in the company documents that goes beyond the standard's requirements (good practice).
6. **Recommendations**: Priority actions to close the gaps, ordered by importance.

RULES:
- Be specific — cite sections, form numbers, and exact requirements.
- Use citations in descriptive format: [Document Title > Section].
- Do NOT invent or assume compliance — if the company source doesn't explicitly address a requirement, mark it as a gap.
- Be thorough but structured. This is a professional compliance document.
- List all sources at the end, grouped by source.

SOURCE FINDINGS:
{sub_answers_text}

ORIGINAL QUESTION: {query}

Provide the gap analysis following the structure above."""

    @staticmethod
    def _synthesis_prompt_comparison(query: str, sub_answers_text: str) -> str:
        return f"""You are a maritime documentation expert comparing multiple sources.

The user wants to understand how different sources address the same topic.
Below are the findings from each source. This is a neutral comparison —
no source is treated as the definitive standard.

YOUR TASK: Produce a structured comparison:

1. **Overview**: What topic is being compared and which sources are involved.
2. **Common Ground**: Where the sources agree or cover the same requirements.
3. **Differences**: Where the sources diverge — different requirements, scope, or level of detail. Be specific about what each source says differently.
4. **Unique to Each**: Requirements or information that appears in only one source.
5. **Summary**: Key takeaways from the comparison.

RULES:
- Be balanced — present each source fairly.
- Use citations in descriptive format: [Document Title > Section].
- Be specific about which source says what.
- List all sources at the end, grouped by source.

SOURCE FINDINGS:
{sub_answers_text}

ORIGINAL QUESTION: {query}

Provide the comparison following the structure above."""

    # -----------------------------------------------------------------
    # Confidence and source collection
    # -----------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        nodes: List[NodeWithScore],
    ) -> Tuple[int, str, str]:
        """Delegate to the battle-tested calculation from query.py."""
        return calculate_confidence(nodes)

    @staticmethod
    def aggregate_confidence(
        sub_answers: List[SubAnswer],
    ) -> Tuple[int, str, str]:
        """Aggregate confidence across multiple sub-answers.

        Uses average confidence with a flag when any sub-answer is weak.
        """
        if not sub_answers:
            return 0, "LOW 🔴", "No sub-answers generated"

        pcts = [sa.confidence_pct for sa in sub_answers]
        avg = sum(pcts) // len(pcts)
        min_pct = min(pcts)

        # Flag weak sub-answers
        weak_labels = [
            sa.source_label for sa in sub_answers
            if sa.confidence_pct < 60
        ]

        if avg >= 80 and not weak_labels:
            level = "HIGH 🟢"
            note = "Strong matches across all sources"
        elif avg >= 60:
            level = "MEDIUM 🟡"
            note = "Moderate matches"
            if weak_labels:
                note += f" — low confidence for: {', '.join(weak_labels)}"
        else:
            level = "LOW 🔴"
            note = "Weak matches across sources — recommend human verification"

        LOGGER.info(
            "ResultSynthesizer CONFIDENCE: avg=%d%% min=%d%% level=%s weak=%s",
            avg, min_pct, level, weak_labels or "none",
        )

        return avg, level, note

    @staticmethod
    def collect_sources(
        sub_answers: List[SubAnswer],
    ) -> List[Dict[str, Any]]:
        """Collect and group source information from all sub-answers.

        Returns the same format as the existing ``sources_info`` list
        for backward compatibility, with an added ``group_label`` field.
        """
        sources = []
        for sa in sub_answers:
            for node in sa.nodes[:10]:
                metadata = node.node.metadata
                sources.append({
                    "source": metadata.get("source", "Unknown"),
                    "section": metadata.get("section", "N/A"),
                    "score": node.score if hasattr(node, "score") else 0.0,
                    "title": metadata.get("title", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "hierarchy": metadata.get("hierarchy", ""),
                    "tab_name": metadata.get("tab_name", ""),
                    "form_number": metadata.get("form_number", ""),
                    "form_category_name": metadata.get("form_category_name", ""),
                    "session_upload": metadata.get("session_upload", False),
                    "upload_display_name": metadata.get("upload_display_name", ""),
                    "upload_original_name": metadata.get("upload_original_name", ""),
                    "group_label": sa.source_label,
                })
        return sources

    # -----------------------------------------------------------------
    # Streaming helper
    # -----------------------------------------------------------------

    def _stream_gemini(
        self,
        prompt: str,
        thinking_budget: int,
    ) -> Generator[str, None, None]:
        """Stream a Gemini response, yielding text chunks."""
        try:
            config = AppConfig.get()
            response_stream = config.client.models.generate_content_stream(
                model=self.MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=ThinkingConfig(
                        thinking_budget=thinking_budget,
                    ),
                ),
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            LOGGER.error("ResultSynthesizer streaming failed: %s", exc)
            yield f"\n\n⚠️ An error occurred while generating the response: {exc}"

    @staticmethod
    def _confidence_instruction(confidence_pct: int) -> str:
        if confidence_pct >= 80:
            return "You have HIGH confidence sources. Answer authoritatively based on the clear documentation."
        elif confidence_pct >= 60:
            return "You have MEDIUM confidence sources. Answer based on available information but suggest verification for critical operations."
        else:
            return "You have LOW confidence sources. Provide what information you found but strongly recommend human verification."


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Singleton instances (stateless, safe to reuse)
_analyzer = QueryAnalyzer()
_decomposer = QueryDecomposer()


def analyze_query(
    query: str,
    last_query: Optional[str] = None,
    last_answer_preview: Optional[str] = None,
    context_turn_count: int = 0,
) -> QueryAnalysis:
    """Analyze a query using the shared QueryAnalyzer instance.

    This is the main entry point for Phase 5a. It will be called by
    the QueryOrchestrator once integration is complete (Phase 5e).

    For now, it can be used standalone to test classification:

        from app.orchestrator import analyze_query
        result = analyze_query("What is the muster list?")
        print(result.intent, result.query_type, result.scope)
    """
    return _analyzer.analyze(
        query=query,
        last_query=last_query,
        last_answer_preview=last_answer_preview,
        context_turn_count=context_turn_count,
    )


def decompose_query(analysis: QueryAnalysis) -> List[SubQuery]:
    """Decompose a query using the shared QueryDecomposer instance.

    Handles both complex (LLM decomposition) and simple (direct wrap) paths.

        from app.orchestrator import analyze_query, decompose_query
        analysis = analyze_query("Compare our D&A policy with RISQ")
        sub_queries = decompose_query(analysis)
    """
    if analysis.requires_decomposition:
        return _decomposer.decompose(analysis)
    return _decomposer.as_single_query(analysis)


# =============================================================================
# Orchestrated Query (Phase 5e) — main entry point
# =============================================================================

def _build_conversation_history(app_state: "AppState") -> str:
    """Build concise conversation history for prompt injection.

    Mirrors the existing _build_conversation_history_context() from query.py
    but uses the updated CONTEXT_HISTORY_WINDOW (4 turns).
    """
    if not app_state.query_history:
        return ""

    recent = app_state.query_history[-CONTEXT_HISTORY_WINDOW:]
    parts = ["=== CONVERSATION HISTORY ==="]
    for idx, entry in enumerate(recent, 1):
        query = entry.get("query", "")
        answer = entry.get("answer", "")
        preview = answer[:300] + "..." if len(answer) > 300 else answer
        parts.append(f"\nTurn {idx}:")
        parts.append(f"User: {query}")
        parts.append(f"Assistant: {preview}")
    parts.append("\n=== END HISTORY ===\n")
    return "\n".join(parts)


def _include_session_uploads(
    app_state: "AppState",
    nodes: List[NodeWithScore],
    query: str,
) -> List[NodeWithScore]:
    """Inject session upload nodes, prioritized above library results.

    Session uploads are always relevant (user uploaded them for a reason),
    so they are prepended and deduplicated against library results.
    """
    session_id = app_state.current_session_id
    if not session_id:
        return nodes

    try:
        upload_manager = app_state.ensure_session_upload_manager()
        session_nodes = upload_manager.search_session_uploads(
            session_id, query, top_k=10,
        )
    except Exception as exc:
        LOGGER.warning("Session upload retrieval failed: %s", exc)
        return nodes

    if not session_nodes:
        return nodes

    combined = session_nodes + nodes
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

    added = max(0, len(deduped) - len(nodes))
    if added:
        LOGGER.info("Orchestrator: Added %d session upload chunks", added)
    return deduped


def orchestrated_query(
    app_state: "AppState",
    query_text: str,
    use_conversation_context: bool = True,
    enable_hierarchical: bool = True,
    status_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Main entry point for the Phase 5 query orchestrator.

    Replaces ``query_with_confidence()`` with intelligent routing,
    decomposition, filtered retrieval, and strategy-aware synthesis.

    Args:
        app_state: Application state with retrievers, history, session info.
        query_text: Raw user query.
        use_conversation_context: Whether to inject conversation history.
        enable_hierarchical: Whether section-level retrieval is allowed.
        status_callback: Optional ``callable(str)`` for UI progress updates.
            Typically ``lambda msg: status.write(msg)`` from Streamlit.

    Returns:
        Dict matching the existing ``query_with_confidence()`` format:
        ``answer_stream``, ``confidence_pct``, ``confidence_level``,
        ``confidence_note``, ``sources``, ``num_sources``, etc.
    """
    def _status(msg: str) -> None:
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass  # UI callback failure is non-fatal
        LOGGER.info("Orchestrator STATUS: %s", msg)

    t0 = time.time()
    synthesizer = ResultSynthesizer()

    # ------------------------------------------------------------------
    # Step 1: Query Analysis
    # ------------------------------------------------------------------
    _status("🔍 Analyzing query...")

    last_query = None
    last_answer_preview = None
    if use_conversation_context and app_state.query_history:
        last_entry = app_state.query_history[-1]
        last_query = last_entry.get("query", "")
        last_answer_preview = last_entry.get("answer", "")[:200]

    analysis = _analyzer.analyze(
        query=query_text,
        last_query=last_query,
        last_answer_preview=last_answer_preview,
        context_turn_count=app_state.context_turn_count if use_conversation_context else 0,
    )

    # ------------------------------------------------------------------
    # Step 2: Non-retrieval fast path
    # ------------------------------------------------------------------
    if analysis.intent in NON_RETRIEVAL_INTENTS:
        _status("✅ Direct response")
        return {
            "query": query_text,
            "answer": analysis.direct_response or "",
            "answer_stream": None,
            "confidence_pct": 100,
            "confidence_level": "HIGH 🟢",
            "confidence_note": f"Direct response ({analysis.intent.value})",
            "sources": [],
            "num_sources": 0,
            "retriever_type": "none",
            "retrieval_strategy": "none",
            "context_mode": use_conversation_context,
            "context_turn": app_state.context_turn_count,
            "context_reset_note": None,
            "topic_extracted": analysis.topic,
            "doc_type_preference": None,
            "scope": analysis.scope.value,
            "attempts": 0,
            "best_attempt": 0,
            "sections_retrieved": 0,
        }

    # ------------------------------------------------------------------
    # Step 3: TOO_BROAD pushback
    # ------------------------------------------------------------------
    if analysis.scope == ScopeAssessment.TOO_BROAD:
        _status("⚠️ Query too broad — requesting clarification")
        pushback = (
            "Your question covers a very broad area that would be difficult to answer "
            "thoroughly in a single response. Could you narrow it down? For example:\n"
            "- Focus on a specific topic or procedure\n"
            "- Ask about a particular regulation or standard\n"
            "- Compare specific sections rather than entire documents"
        )
        LOGGER.info("Orchestrator: TOO_BROAD pushback for query '%s'", query_text[:60])
        return {
            "query": query_text,
            "answer": pushback,
            "answer_stream": None,
            "confidence_pct": 100,
            "confidence_level": "HIGH 🟢",
            "confidence_note": "Query scope assessment",
            "sources": [],
            "num_sources": 0,
            "retriever_type": "none",
            "retrieval_strategy": analysis.retrieval_strategy.value,
            "context_mode": use_conversation_context,
            "context_turn": app_state.context_turn_count,
            "context_reset_note": None,
            "topic_extracted": analysis.topic,
            "doc_type_preference": analysis.doc_type_hint,
            "scope": analysis.scope.value,
            "attempts": 0,
            "best_attempt": 0,
            "sections_retrieved": 0,
        }

    # ------------------------------------------------------------------
    # Step 4: Topic inheritance for follow-ups
    # ------------------------------------------------------------------
    topic = analysis.topic
    if analysis.topic_inherited and app_state.last_topic:
        topic = app_state.last_topic
        LOGGER.info("Orchestrator: Inherited topic '%s' from previous turn", topic)

    # Context reset check
    context_reset_note = None
    if use_conversation_context and app_state.context_turn_count >= MAX_CONTEXT_TURNS:
        app_state.context_turn_count = 0
        app_state.last_topic = None
        context_reset_note = f"Conversation context reset after {MAX_CONTEXT_TURNS} turns."
        LOGGER.info("Orchestrator: Context reset after %d turns", MAX_CONTEXT_TURNS)

    # ------------------------------------------------------------------
    # Step 5: Query Decomposition
    # ------------------------------------------------------------------
    _status("🧩 Planning retrieval strategy...")

    try:
        sub_queries = decompose_query(analysis)
    except DecompositionError as exc:
        _status("⚠️ Could not decompose query")
        LOGGER.warning("Orchestrator: DecompositionError — %s", exc)
        return {
            "query": query_text,
            "answer": str(exc),
            "answer_stream": None,
            "confidence_pct": 100,
            "confidence_level": "HIGH 🟢",
            "confidence_note": "Decomposition feedback",
            "sources": [],
            "num_sources": 0,
            "retriever_type": "none",
            "retrieval_strategy": analysis.retrieval_strategy.value,
            "context_mode": use_conversation_context,
            "context_turn": app_state.context_turn_count,
            "context_reset_note": context_reset_note,
            "topic_extracted": topic,
            "doc_type_preference": analysis.doc_type_hint,
            "scope": analysis.scope.value,
            "attempts": 0,
            "best_attempt": 0,
            "sections_retrieved": 0,
        }

    LOGGER.info(
        "Orchestrator: %d sub-queries, merge_strategy=%s",
        len(sub_queries), analysis.merge_strategy.value,
    )

    # ------------------------------------------------------------------
    # Step 6: Filtered Retrieval
    # ------------------------------------------------------------------
    _status(f"📚 Searching documents ({len(sub_queries)} {'queries' if len(sub_queries) > 1 else 'query'})...")

    retriever = FilteredRetriever(app_state)

    if len(sub_queries) == 1:
        retrieval_results = [retriever.retrieve_single(sub_queries[0])]
    else:
        retrieval_results = retriever.retrieve_parallel(sub_queries)

    # Inject session uploads into each result's nodes
    for rr in retrieval_results:
        rr.nodes = _include_session_uploads(app_state, rr.nodes, rr.sub_query.text)

    total_nodes = sum(len(rr.nodes) for rr in retrieval_results)
    LOGGER.info("Orchestrator: Retrieval complete — %d total nodes", total_nodes)

    # ------------------------------------------------------------------
    # Step 7: Generate answer(s) and stream
    # ------------------------------------------------------------------
    is_simple = (
        analysis.merge_strategy == MergeStrategy.DIRECT
        and len(retrieval_results) == 1
    )

    if is_simple:
        # ----- DIRECT PATH: stream single answer -----
        _status("✍️ Generating answer...")
        nodes = retrieval_results[0].nodes
        confidence_pct, confidence_level, confidence_note = (
            synthesizer._calculate_confidence(nodes)
        )

        conversation_history = ""
        if use_conversation_context and analysis.intent in (
            Intent.FOLLOW_UP_SAME, Intent.FOLLOW_UP_NEW,
        ):
            conversation_history = _build_conversation_history(app_state)

        answer_stream = synthesizer.generate_direct_stream(
            query=query_text,
            nodes=nodes,
            conversation_history=conversation_history,
            confidence_pct=confidence_pct,
        )

        all_nodes = nodes
        sources = synthesizer.collect_sources([
            SubAnswer(
                source_label="general",
                text="",
                nodes=nodes,
                confidence_pct=confidence_pct,
                confidence_level=confidence_level,
                confidence_note=confidence_note,
            )
        ])

    else:
        # ----- COMPLEX PATH: sub-answers then synthesis -----
        sub_answers: List[SubAnswer] = []
        for i, rr in enumerate(retrieval_results, 1):
            _status(
                f"✍️ Checking for... "
                f"{rr.source_label}..."
            )
            sa = synthesizer.generate_sub_answer(rr.sub_query, rr.nodes)
            sub_answers.append(sa)

        confidence_pct, confidence_level, confidence_note = (
            synthesizer.aggregate_confidence(sub_answers)
        )

        _status("🔗 Synthesizing final response...")
        answer_stream = synthesizer.generate_synthesis_stream(
            original_query=query_text,
            sub_answers=sub_answers,
            merge_strategy=analysis.merge_strategy,
            has_regulatory_standard=analysis.has_regulatory_standard,
        )

        all_nodes = []
        for sa in sub_answers:
            all_nodes.extend(sa.nodes)
        sources = synthesizer.collect_sources(sub_answers)

    # ------------------------------------------------------------------
    # Step 8: Update state
    # ------------------------------------------------------------------
    if use_conversation_context:
        if analysis.intent == Intent.NEW_QUERY:
            app_state.context_turn_count = 1
        else:
            app_state.context_turn_count += 1
        app_state.last_topic = topic

    elapsed = time.time() - t0
    LOGGER.info(
        "Orchestrator COMPLETE: strategy=%s sub_queries=%d nodes=%d "
        "confidence=%d%% elapsed=%.1fs",
        analysis.merge_strategy.value,
        len(sub_queries),
        total_nodes,
        confidence_pct,
        elapsed,
    )

    # ------------------------------------------------------------------
    # Step 9: Return backward-compatible result dict
    # ------------------------------------------------------------------
    return {
        "query": query_text,
        "answer": "",  # Placeholder — streamed via answer_stream
        "answer_stream": answer_stream,
        "confidence_pct": confidence_pct,
        "confidence_level": confidence_level,
        "confidence_note": confidence_note,
        "sources": sources,
        "num_sources": len(all_nodes),
        "retriever_type": "orchestrated",
        "retrieval_strategy": analysis.retrieval_strategy.value,
        "context_mode": use_conversation_context,
        "context_turn": app_state.context_turn_count,
        "context_reset_note": context_reset_note,
        "topic_extracted": topic,
        "doc_type_preference": analysis.doc_type_hint,
        "scope": analysis.scope.value,
        "attempts": 1,
        "best_attempt": 1,
        "sections_retrieved": 0,
    }


__all__ = [
    # Enums
    "Intent",
    "QueryType",
    "ScopeAssessment",
    "RetrievalStrategy",
    "MergeStrategy",
    # Dataclasses
    "QueryAnalysis",
    "SubQuery",
    "RetrievalResult",
    "SubAnswer",
    # Exceptions
    "DecompositionError",
    # Constants
    "NON_RETRIEVAL_INTENTS",
    "SOURCE_PATTERNS",
    # Classes
    "QueryAnalyzer",
    "FilteredRetriever",
    "QueryDecomposer",
    "ResultSynthesizer",
    # Functions
    "analyze_query",
    "decompose_query",
    "orchestrated_query",
]
