"""Session state helpers for the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from collections import defaultdict

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever


from .config import AppConfig
from .feedback import FeedbackSystem
from .indexing import IncrementalIndexManager, load_cached_nodes_and_index
from .logger import LOGGER


@dataclass
class AppState:
    nodes: List[Document] = field(default_factory=list)
    index: Optional[VectorStoreIndex] = None
    vector_retriever: Optional[VectorIndexRetriever] = None
    bm25_retriever: Optional[BM25Retriever] = None
    query_history: List[Dict] = field(default_factory=list)
    history_log: List[Dict] = field(default_factory=list)
    last_result: Optional[Dict] = None
    manager: Optional[IncrementalIndexManager] = None
    feedback_system: FeedbackSystem = field(default_factory=FeedbackSystem)
    history_loaded: bool = False
    history_log_path: Optional[Path] = None
    _index_load_attempted: bool = False  # Track if we've tried loading
    
    # Context-aware conversation state
    sticky_chunks: List[Any] = field(default_factory=list)  # Reused chunks for followups
    context_turn_count: int = 0  # Track turns in current conversation thread
    conversation_active: bool = False  # Is context mode enabled?

    def ensure_index_loaded(self) -> bool:
        """
        Lazy-load the index if not already present.
        
        Returns:
            True if index is ready (loaded or already present)
            False if index could not be loaded
        """
        # Already loaded and ready
        if self.nodes and self.index:
            LOGGER.debug("Index already loaded: %d nodes", len(self.nodes))
            return True
        
        # Already tried and failed, don't spam attempts
        if self._index_load_attempted and not self.nodes:
            LOGGER.debug("Index load previously failed, skipping retry")
            return False
        
        # Mark that we're attempting to load
        self._index_load_attempted = True
        
        LOGGER.info("Attempting to load cached index...")
        try:
            nodes, index = load_cached_nodes_and_index()
            
            if nodes and index:
                self.nodes = nodes
                self.index = index
                # Force retriever recreation
                self.vector_retriever = None
                self.bm25_retriever = None
                self.ensure_retrievers()
                
                # Update manager if it exists
                if self.manager:
                    self.manager.nodes = nodes
                
                LOGGER.info("Successfully loaded %d cached nodes", len(nodes))
                return True
            else:
                LOGGER.warning("No cached index found. User must build index first.")
                return False
                
        except Exception as exc:
            LOGGER.exception("Failed to load cached index: %s", exc)
            return False

    def ensure_retrievers(self) -> None:
        """Initialise retrievers if nodes and index are ready."""
        if not self.nodes or not self.index:
            LOGGER.debug("Skipping retriever init: nodes=%s, index=%s", 
                        bool(self.nodes), bool(self.index))
            return
        
        if self.vector_retriever is None:
            self.vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=40)
            LOGGER.debug("Initialized vector retriever")
            
        if self.bm25_retriever is None:
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=40)
            LOGGER.debug("Initialized BM25 retriever")

    def is_ready_for_queries(self) -> bool:
        """Check if the system is ready to handle queries."""
        return (
            self.nodes is not None 
            and len(self.nodes) > 0
            and self.index is not None
            and self.vector_retriever is not None
            and self.bm25_retriever is not None
        )

    def ensure_manager(self) -> IncrementalIndexManager:
        """Provide an incremental manager bound to the current state."""
        if self.manager is None:
            config = AppConfig.get()
            paths = config.paths
            self.manager = IncrementalIndexManager(
                docs_path=paths.docs_path,
                gemini_cache_path=paths.gemini_json_cache,
                nodes_cache_path=paths.nodes_cache_path,
                cache_info_path=paths.cache_info_path,
                chroma_path=paths.chroma_path,
            )
            self.manager.nodes = self.nodes
        return self.manager

    def document_titles_by_source(self) -> Dict[str, str]:
        titles = {}
        for node in self.nodes:
            source = node.metadata.get("source")
            title = node.metadata.get("title")
            if source and title:
                titles[source] = title
        return titles

    def documents_grouped_by_type(self) -> Dict[str, List[str]]:
        grouped: Dict[str, set] = defaultdict(set)
        for node in self.nodes:
            metadata = node.metadata
            doc_type = str(metadata.get("doc_type", "UNCATEGORIZED")).upper()
            title = metadata.get("title") or metadata.get("source") or "Untitled"
            if doc_type == "FORM":
                form_number = metadata.get("form_number")
                if form_number:
                    title = f"{form_number} - {title}"
            grouped[doc_type].add(title)
        return {doc_type: sorted(list(titles)) for doc_type, titles in grouped.items()}

    def reset_session(self) -> None:
        """Clear the in-memory conversation without touching the persisted log."""
        self.query_history.clear()
        self.last_result = None
        self.sticky_chunks.clear()
        self.context_turn_count = 0
        self.conversation_active = False
        LOGGER.debug("Session reset: cleared query history and context state")

    def _ensure_history_path(self) -> Path:
        if self.history_log_path is None:
            paths = AppConfig.get().paths
            self.history_log_path = paths.cache_dir / "chat_history.jsonl"
            self.history_log_path.parent.mkdir(parents=True, exist_ok=True)
        return self.history_log_path

    def ensure_history_loaded(self) -> None:
        if self.history_loaded:
            return
        self.history_log.clear()
        log_path = self._ensure_history_path()
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    self.history_log.append(record)
        self.history_loaded = True
        LOGGER.debug("Loaded %d history entries", len(self.history_log))

    def append_history(self, result: Dict) -> None:
        self.ensure_history_loaded()
        self.query_history.append(result)
        self.history_log.append(result)
        self.last_result = result
        log_path = self._ensure_history_path()
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    def clear_history(self) -> None:
        self.history_log.clear()
        self.reset_session()
        log_path = self._ensure_history_path()
        if log_path.exists():
            log_path.unlink()
        self.history_loaded = True
        LOGGER.info("History cleared")


__all__ = ["AppState"]