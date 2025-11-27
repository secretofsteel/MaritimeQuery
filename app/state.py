"""Session state helpers for the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from collections import defaultdict

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever


from .config import AppConfig
from .feedback import FeedbackSystem
from .indexing import IncrementalIndexManager, load_cached_nodes_and_index
from .logger import LOGGER
from .sessions import SessionManager
from .session_uploads import SessionUploadManager, SessionUploadChunk




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
    # Session upload management
    session_upload_manager: Optional[SessionUploadManager] = None
    _session_upload_metadata_cache: Dict[str, List[SessionUploadChunk]] = field(default_factory=dict)

    # REMOVED: _index_load_attempted (now in st.session_state)
    
    # Context-aware conversation state
    sticky_chunks: List[Any] = field(default_factory=list)  # Reused chunks for followups
    context_turn_count: int = 0  # Track turns in current conversation thread
    conversation_active: bool = False  # Is context mode enabled?
    last_topic: Optional[str] = None  # Semantic topic from last query (for inheritance)
    conversation_summary: str = ""  # Running summary instead of full history
    last_doc_type_pref: Optional[str] = None  # Last detected doc type preference
    last_scope: Optional[str] = None  # NEW: Last detected scope (company/regulatory/operational/safety/general)
    session_manager: Optional[SessionManager] = None
    current_session_id: Optional[str] = None
    _session_messages_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    session_upload_manager: Optional[SessionUploadManager] = None
    _session_upload_metadata_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _session_upload_chunks_cache: Dict[str, List[SessionUploadChunk]] = field(default_factory=dict)
    _node_map_cache: Optional[Dict[str, Any]] = None  # Cached node map for hierarchical retrieval
    hierarchical_enabled: bool = False  # Whether hierarchical retrieval is available

    def ensure_index_loaded(self) -> bool:
        """
        Lazy-load the index if not already present.

        Returns:
            True if index is ready (loaded or already present)
            False if index could not be loaded
        """
        # Import here to avoid circular dependency at module level
        try:
            import streamlit as st
        except ImportError:
            # Fallback for non-Streamlit contexts (testing, etc.)
            LOGGER.warning("Streamlit not available, using in-memory flag for index loading")
            st = None

        # Already loaded and ready
        if self.nodes and self.index:
            LOGGER.debug("Index already loaded: %d nodes", len(self.nodes))
            # Check hierarchical support if not already checked
            if not self.hierarchical_enabled:
                self._validate_hierarchical_support()
            return True

        # Check session state flag (persistent across Streamlit reruns)
        if st is not None:
            if "index_load_attempted" not in st.session_state:
                st.session_state["index_load_attempted"] = False

            # Already tried and failed, don't spam attempts
            if st.session_state["index_load_attempted"] and not self.nodes:
                LOGGER.debug("Index load previously failed, skipping retry")
                return False

            # Mark that we're attempting to load
            st.session_state["index_load_attempted"] = True

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

                # Validate hierarchical support
                self._validate_hierarchical_support()

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

    def ensure_session_manager(self) -> SessionManager:
        """Initialise and cache the chat session manager."""
        if self.session_manager is None:
            self.session_manager = SessionManager()
        return self.session_manager

    def ensure_session_upload_manager(self) -> SessionUploadManager:
        """Initialise and cache the session upload manager."""
        if self.session_upload_manager is None:
            self.session_upload_manager = SessionUploadManager()
        return self.session_upload_manager

    def create_new_session(self, title: str = "New Chat") -> str:
        """Create and switch to a brand new chat session."""
        manager = self.ensure_session_manager()
        session_id = manager.create_session(title=title)
        self.current_session_id = session_id
        self.reset_session()
        self._session_messages_cache[session_id] = []
        self.refresh_session_upload_cache(session_id)
        return session_id

    def switch_session(self, session_id: str) -> None:
        """
        Switch the active chat session, loading messages if necessary.
        
        NEW: Automatically restores conversation context (topic, cached chunks, etc.)
        from the last assistant message in the loaded session.
        """
        manager = self.ensure_session_manager()
        if not manager.get_session(session_id):
            LOGGER.warning("Requested session does not exist: %s", session_id)
            return

        if session_id != self.current_session_id:
            self.current_session_id = session_id
            self.reset_session()  # Clear current context

        # Drop any cached messages so they reload fresh from disk
        self._session_messages_cache.pop(session_id, None)
        self.get_current_session_messages()
        self.refresh_session_upload_cache(session_id)
        
        # NEW: Restore context from session
        self.restore_session_context()

    def restore_session_context(self) -> None:
        """
        Restore conversation context from the currently loaded session.
        
        This method:
        1. Finds the last assistant message with context_state
        2. Restores topic, scope, doc_type preferences, and turn count
        3. Retrieves cached chunks from ChromaDB using stored IDs
        4. Re-indexes uploaded files from their JSONL extractions
        
        Called automatically after switch_session().
        """
        if not self.current_session_id:
            LOGGER.debug("No active session to restore context from")
            return
        
        messages = self.get_current_session_messages()
        if not messages:
            LOGGER.debug("No messages in session, nothing to restore")
            return
        
        # Find LAST assistant message with context_state
        last_context_state = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                context_state = msg.get("context_state")
                if context_state:
                    last_context_state = context_state
                    break
        
        if not last_context_state:
            LOGGER.info("No context state found in session, starting fresh")
            return
        
        # Restore context fields
        self.last_topic = last_context_state.get("topic")
        self.last_doc_type_pref = last_context_state.get("doc_type_pref")
        self.last_scope = last_context_state.get("scope")
        self.context_turn_count = last_context_state.get("turn_count", 0)
        
        LOGGER.info("📂 Restored session context: topic='%s', scope='%s', turn=%d/6", 
                self.last_topic, self.last_scope, self.context_turn_count)
        
        # Restore cached chunks by retrieving from ChromaDB using IDs
        cached_chunk_ids = last_context_state.get("cached_chunk_ids", [])
        if cached_chunk_ids and self.nodes:
            # Build node map (id -> node) for fast lookup
            LOGGER.debug("Building node map for chunk restoration...")
            node_map = {}
            for node in self.nodes:
                node_id = getattr(node, 'node_id', getattr(node, 'id_', None))
                if node_id:
                    node_map[node_id] = node
            
            # Retrieve chunks
            restored_chunks = []
            for chunk_id in cached_chunk_ids:
                if chunk_id in node_map:
                    restored_chunks.append(node_map[chunk_id])
                else:
                    LOGGER.warning("Chunk ID not found in index: %s", chunk_id)
            
            if restored_chunks:
                self.sticky_chunks = restored_chunks
                LOGGER.info("✅ Restored %d/%d cached chunks", 
                        len(restored_chunks), len(cached_chunk_ids))
            else:
                LOGGER.warning("⚠️ Could not restore any cached chunks (IDs not found in current index)")
                # This is OK - just means we'll do fresh retrieval on next query
        
        elif cached_chunk_ids:
            LOGGER.warning("⚠️ Cannot restore chunks: index not loaded")
        
        # Re-index uploaded files if they exist
        # Note: This is a placeholder - full implementation needed in session_uploads.py
        self._restore_session_uploads()

    def _restore_session_uploads(self) -> None:
        """
        Re-index uploaded files from JSONL extractions.
        
        Reads extraction JSONLs from sessions/{session_id}/uploads/
        and temporarily re-indexes them (generates embeddings and adds to index).
        
        This is called automatically when loading a session with uploaded files.
        
        NOTE: Currently a placeholder. Full implementation requires:
        1. Reading JSONL extraction files
        2. Converting to Document objects
        3. Generating embeddings
        4. Temporarily adding to index (session-scoped)
        5. Updating session_upload_chunks_cache
        """
        if not self.current_session_id:
            return
        
        from .config import AppConfig
        config = AppConfig.get()
        upload_dir = config.paths.cache_dir / "sessions" / self.current_session_id / "uploads"
        
        if not upload_dir.exists():
            LOGGER.debug("No upload directory for session")
            return
        
        # Find all extraction JSONLs
        extraction_files = list(upload_dir.glob("*.jsonl"))
        if not extraction_files:
            LOGGER.debug("No uploaded file extractions to restore")
            return
        
        LOGGER.info("📎 Found %d uploaded files to re-index...", len(extraction_files))
        
        # TODO: Full implementation
        # For Phase 1, we'll just log that files exist
        # User will need to re-upload if they want to query them
        LOGGER.info("⚠️ Upload restoration not yet implemented - re-upload files if needed")

    def _message_to_display_dict(self, message: Any) -> Dict[str, Any]:
        """Flatten a Message object into the format expected by the UI."""
        base: Dict[str, Any] = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
            "timestamp": getattr(message, "timestamp", datetime.now()).isoformat(),
        }
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, dict):
            base.update(metadata)
        return base

    def get_current_session_messages(self) -> List[Dict[str, Any]]:
        """Return cached messages for the current session, loading from disk if needed."""
        session_id = self.current_session_id
        if not session_id:
            return []

        if session_id not in self._session_messages_cache:
            manager = self.ensure_session_manager()
            messages = [
                self._message_to_display_dict(message)
                for message in manager.load_messages(session_id)
            ]
            self._session_messages_cache[session_id] = messages

        return self._session_messages_cache[session_id]

    def add_message_to_current_session(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist and cache a new chat message for the active session.
        
        NEW: For assistant messages, automatically captures and saves context state
        (topic, scope, cached chunks) for later restoration.
        """
        session_id = self.current_session_id
        if not session_id:
            session_id = self.create_new_session()

        # NEW: If this is an assistant message, capture context state
        if role == "assistant" and metadata is not None:
            # Build context state snapshot
            context_state = {
                "topic": self.last_topic,
                "doc_type_pref": self.last_doc_type_pref,
                "scope": self.last_scope,
                "turn_count": self.context_turn_count,
                "cached_chunk_ids": [],
            }
            
            # Capture cached chunk IDs (not full chunks - just references)
            if self.sticky_chunks:
                # Get up to 40 chunk IDs for restoration
                context_state["cached_chunk_ids"] = [
                    getattr(chunk, 'node_id', getattr(chunk, 'id_', f"chunk_{i}"))
                    for i, chunk in enumerate(self.sticky_chunks[:40])
                ]
                LOGGER.debug("Captured %d cached chunk IDs for context restoration", 
                            len(context_state["cached_chunk_ids"]))
            
            # Add to message metadata
            metadata["context_state"] = context_state

        manager = self.ensure_session_manager()
        manager.add_message(session_id, role=role, content=content, metadata=metadata)

        # Update cache
        cached_entry: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            cached_entry.update(metadata)

        self._session_messages_cache.setdefault(session_id, []).append(cached_entry)

    def get_session_upload_metadata(self) -> List[Dict[str, Any]]:
        session_id = self.current_session_id
        if not session_id:
            return []

        if session_id not in self._session_upload_metadata_cache:
            manager = self.ensure_session_upload_manager()
            self._session_upload_metadata_cache[session_id] = manager.load_metadata(session_id)

        return self._session_upload_metadata_cache.get(session_id, [])

    def get_session_upload_chunks(self) -> List[SessionUploadChunk]:
        session_id = self.current_session_id
        if not session_id:
            return []

        if session_id not in self._session_upload_chunks_cache:
            manager = self.ensure_session_upload_manager()
            self._session_upload_chunks_cache[session_id] = manager.load_chunks(session_id)

        return self._session_upload_chunks_cache.get(session_id, [])

    def refresh_session_upload_cache(self, session_id: Optional[str] = None) -> None:
        target = session_id or self.current_session_id
        if not target:
            return
        self._session_upload_metadata_cache.pop(target, None)
        self._session_upload_chunks_cache.pop(target, None)

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
        self.last_topic = None
        self.conversation_summary = ""
        self.last_scope = None  # NEW: Clear scope on reset
        LOGGER.debug("Session reset: cleared query history and context state")

    def delete_session_with_uploads(self, session_id: str) -> bool:
        manager = self.ensure_session_manager()
        deleted = manager.delete_session(session_id)
        if deleted:
            upload_manager = self.ensure_session_upload_manager()
            upload_manager.delete_session_uploads(session_id)
            self.refresh_session_upload_cache(session_id)
            self._session_messages_cache.pop(session_id, None)
        return deleted

    def clear_all_sessions(self) -> int:
        manager = self.ensure_session_manager()
        count = manager.clear_all_sessions()
        upload_manager = self.ensure_session_upload_manager()
        upload_manager.clear_all()
        self._session_messages_cache.clear()
        self._session_upload_metadata_cache.clear()
        self._session_upload_chunks_cache.clear()
        return count

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

    def _validate_hierarchical_support(self) -> None:
        """Validate that document trees exist for hierarchical retrieval."""
        config = AppConfig.get()
        trees_path = config.paths.cache_dir / "document_trees.json"

        if not trees_path.exists():
            LOGGER.warning("⚠️  Document trees not found at %s", trees_path)
            LOGGER.warning("   Hierarchical retrieval is DISABLED")
            LOGGER.warning("   To enable: Rebuild index via Admin → Full Rebuild")
            self.hierarchical_enabled = False
        else:
            from .indexing import load_document_trees
            trees = load_document_trees(trees_path)
            if trees:
                LOGGER.info("✅ Loaded %d document trees - hierarchical retrieval ENABLED", len(trees))
                self.hierarchical_enabled = True
            else:
                LOGGER.warning("⚠️  Document trees file empty - hierarchical retrieval DISABLED")
                self.hierarchical_enabled = False

    def get_node_map(self) -> Dict[str, Any]:
        """
        Get or build cached node map for hierarchical retrieval.

        Returns:
            Dictionary mapping node_id -> NodeWithScore
        """
        if self._node_map_cache is None:
            from llama_index.core.schema import NodeWithScore

            node_map: Dict[str, Any] = {}
            for item in self.nodes:
                # Get node_id (handles multiple attribute names)
                node_id = None
                if hasattr(item, 'node_id'):
                    node_id = item.node_id
                elif hasattr(item, 'id_'):
                    node_id = item.id_
                elif hasattr(item, 'doc_id'):
                    node_id = item.doc_id

                if node_id:
                    node_map[node_id] = NodeWithScore(node=item, score=0.5)

            self._node_map_cache = node_map
            LOGGER.info("Built node map cache with %d entries", len(node_map))

        return self._node_map_cache

    def invalidate_node_map_cache(self) -> None:
        """Invalidate node map cache when index is rebuilt."""
        self._node_map_cache = None
        LOGGER.debug("Node map cache invalidated")


__all__ = ["AppState"]
