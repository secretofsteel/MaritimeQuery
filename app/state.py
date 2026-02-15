"""Application state management (Streamlit-free)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from collections import defaultdict

from llama_index.core import Document, VectorStoreIndex
from llama_index.core import Settings as LlamaSettings
from llama_index.core.retrievers import VectorIndexRetriever



from .config import AppConfig
from .feedback import FeedbackSystem
from .indexing import IncrementalIndexManager, load_cached_nodes_and_index
from .logger import LOGGER
from .sessions import SessionManager
from .session_uploads import SessionUploadManager, SessionUploadChunk
from .retrieval import PgFTSRetriever, PgNodeLoader, TenantAwareVectorRetriever
from .nodes import get_node_count, NodeRepository



@dataclass
class AppState:
    tenant_id: str = "shared"
    # Core state
    nodes: List[Document] = field(default_factory=list)
    index: Optional[VectorStoreIndex] = None
    # Retrievers
    fts_retriever: Optional[PgFTSRetriever] = None
    vector_retriever: Optional[TenantAwareVectorRetriever] = None
    bm25_retriever: Optional[PgFTSRetriever] = None  # Alias for backward compatibility
    query_history: List[Dict] = field(default_factory=list)
    history_log: List[Dict] = field(default_factory=list)
    last_result: Optional[Dict] = None
    _managers: Dict[str, IncrementalIndexManager] = field(default_factory=dict)
    feedback_system: FeedbackSystem = field(default_factory=FeedbackSystem)
    history_loaded: bool = False
    history_log_path: Optional[Path] = None

    # Session upload management
    session_upload_manager: Optional[SessionUploadManager] = None
    _session_upload_metadata_cache: Dict[str, List[SessionUploadChunk]] = field(default_factory=dict)
    
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
    _shared_qdrant_client: Optional[Any] = field(default=None, repr=False)
    hierarchical_enabled: bool = False  # Whether hierarchical retrieval is available
    _index_load_attempted: bool = field(default=False, init=False)

    def ensure_nodes_loaded(self) -> List[Document]:
        """Load nodes from PostgreSQL if not already in memory."""
        from .nodes import NodeRepository

        if not self.nodes:
            tenant_id = self.tenant_id
            
            # Load tenant-specific nodes
            repo = NodeRepository(tenant_id=tenant_id)
            tenant_nodes = repo.get_all_nodes()
            
            # Also load shared nodes if tenant is not 'shared'
            if tenant_id != "shared":
                repo_shared = NodeRepository(tenant_id="shared")
                shared_nodes = repo_shared.get_all_nodes()
                self.nodes = tenant_nodes + shared_nodes
                LOGGER.info("Loaded %d tenant + %d shared nodes", len(tenant_nodes), len(shared_nodes))
            else:
                self.nodes = tenant_nodes
                LOGGER.info("Loaded %d nodes (tenant=%s)", len(self.nodes), tenant_id)
        
        return self.nodes

    def ensure_index_loaded(self) -> bool:
        """Ensure search index is ready."""
        if self._index_load_attempted:
            return self.index is not None

        self._index_load_attempted = True
        
        # Load Qdrant-backed index
        if self.index is None:
            from .indexing import load_cached_nodes_and_index
            nodes, index = load_cached_nodes_and_index()  # No tenant_id
            
            if index is not None:
                self.index = index
                # Don't store nodes - FTS queries PostgreSQL directly
                LOGGER.info("Loaded Qdrant-backed index")
            else:
                LOGGER.info("No cached index found")
                return False
        
        # Set up retrievers
        self.ensure_retrievers()
        self._validate_hierarchical_support()
        
        return self.index is not None

    def ensure_retrievers(self) -> None:
        """Ensure retriever instances are ready.

        Creates:
        - FTS5Retriever — queries PostgreSQL directly (tenant-aware)
        - TenantAwareVectorRetriever — queries Qdrant with tenant filter
        """
        from .config import AppConfig
        from .vector_store import get_qdrant_client, ensure_collection

        tenant_id = self.tenant_id

        # 2. FTS5 (PostgreSQL) Retriever
        if self.fts_retriever is None:
            self.fts_retriever = PgFTSRetriever(
                tenant_id=tenant_id,
                similarity_top_k=20,
            )
            LOGGER.debug("Created PgFTSRetriever for tenant %s", tenant_id)

        # Point bm25_retriever to fts5 for compatibility
        self.bm25_retriever = self.fts_retriever

        # Tenant-aware vector retriever
        if self.vector_retriever is None and self.index is not None:
            # Use shared client if injected (FastAPI path), else create one
            qdrant_client = self._shared_qdrant_client
            if qdrant_client is None:
                qdrant_client = get_qdrant_client()

            config = AppConfig.get()
            collection_name = ensure_collection(qdrant_client)

            self.vector_retriever = TenantAwareVectorRetriever(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                embed_model=LlamaSettings.embed_model,
                tenant_id=tenant_id,
                similarity_top_k=20,
            )
            LOGGER.debug("Created TenantAwareVectorRetriever for tenant %s", tenant_id)

    def is_ready_for_queries(self) -> bool:
        """Check if the system is ready to handle queries."""
        return (
            self.index is not None
            and self.vector_retriever is not None
            and self.fts_retriever is not None  # Changed from bm25_retriever
        )

    def ensure_manager(self, target_tenant_id: Optional[str] = None) -> IncrementalIndexManager:
        """Provide an incremental manager bound to a tenant.

        Args:
            target_tenant_id: Override tenant for admin operations.
                Defaults to self.tenant_id if not provided.

        Managers are cached per tenant_id to avoid recreation ping-pong
        when admin panel alternates between tenants.
        """
        tenant_id = target_tenant_id or self.tenant_id

        if tenant_id not in self._managers:
            config = AppConfig.get()
            paths = config.paths
            manager = IncrementalIndexManager(
                docs_path=config.docs_path_for(tenant_id),
                gemini_cache_path=config.gemini_cache_for(tenant_id),
                cache_info_path=paths.cache_info_path,
                tenant_id=tenant_id,
            )
            manager.nodes = self.nodes
            self._managers[tenant_id] = manager
            LOGGER.info("Created IncrementalIndexManager for tenant '%s'", tenant_id)

        return self._managers[tenant_id]

    def ensure_session_manager(self) -> SessionManager:
        """
        Get or create SessionManager, scoped to current tenant.

        Returns:
            SessionManager instance for the current tenant.

        Raises:
            RuntimeError: If no tenant_id set on AppState.
        """
        tenant_id = self.tenant_id

        if tenant_id is None:
            raise RuntimeError(
                "No tenant_id set on AppState. "
                "Ensure tenant_id is provided when constructing AppState."
            )
        
        # Create new manager if none exists or tenant changed
        if self.session_manager is None:
            self.session_manager = SessionManager(tenant_id=tenant_id)
            LOGGER.debug("Created SessionManager for tenant: %s", tenant_id)
        
        elif self.session_manager.tenant_id != tenant_id:
            # Tenant changed (e.g., during testing) - recreate manager
            LOGGER.warning(
                "Tenant changed from %s to %s - recreating SessionManager",
                self.session_manager.tenant_id, tenant_id
            )
            self.session_manager = SessionManager(tenant_id=tenant_id)
            
            # Also clear session caches since they belong to old tenant
            self._session_messages_cache.clear()
            self.current_session_id = None
        
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

        # Rebuild query_history from messages
        messages = self.get_current_session_messages()
        self.query_history.clear()

        for msg in messages:
            if msg.get("role") == "assistant":
                query_result = {
                    "query": msg.get("query", ""),
                    "answer": msg.get("content", ""),
                    "topic_extracted": msg.get("topic_extracted"),
                    "confidence_pct": msg.get("confidence_pct", 0),
                    "sources": msg.get("sources", []),
                }
                self.query_history.append(query_result)

        LOGGER.info("Rebuilt query_history with %d entries", len(self.query_history))

    def restore_session_context(self) -> None:
        """
        Restore conversation context from the currently loaded session.
        
        This method:
        1. Finds the last assistant message with context_state
        2. Restores topic, scope, doc_type preferences, and turn count
        3. Retrieves cached chunks from node map using stored IDs
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
        
        # Restore cached chunks by retrieving from node map using IDs
        cached_chunk_ids = last_context_state.get("cached_chunk_ids", [])
        if cached_chunk_ids and self.nodes:
            # Build node map (id -> node) for fast lookup
            LOGGER.debug("Building node map for chunk restoration...")
            from llama_index.core.schema import NodeWithScore

            node_map = {}
            for node in self.nodes:
                # Get the actual node ID
                node_id = getattr(node, 'id_', getattr(node, 'node_id', None))
                if node_id:
                    node_map[node_id] = node

            # Retrieve chunks and wrap as NodeWithScore
            restored_chunks = []
            missing_chunks = 0
            for chunk_id in cached_chunk_ids:
                if chunk_id in node_map:
                    # Wrap in NodeWithScore (same format as sticky_chunks)
                    node = node_map[chunk_id]
                    restored_chunks.append(NodeWithScore(node=node, score=0.8))
                else:
                    missing_chunks += 1

            if missing_chunks > 0:
                LOGGER.warning("⚠️  %d/%d cached chunks not found in current index (may have been rebuilt)",
                            missing_chunks, len(cached_chunk_ids))
        
        elif cached_chunk_ids:
            LOGGER.warning("⚠️ Cannot restore chunks: index not loaded")
        
        # Re-index uploaded files if they exist
        # Note: This is a placeholder - full implementation needed in session_uploads.py
        self._restore_session_uploads()

    def _restore_session_uploads(self) -> None:
        """
        Re-index uploaded files from JSONL extractions.
        """
        if not self.current_session_id:
            return
        
        upload_manager = self.ensure_session_upload_manager()
        
        # Check if there are any extraction JSONLs to restore
        from .config import AppConfig
        config = AppConfig.get()
        upload_dir = config.paths.cache_dir / "sessions" / self.current_session_id / "uploads"
        
        if not upload_dir.exists():
            LOGGER.debug("No upload directory for session")
            return
        
        # Count extraction files
        extraction_files = list(upload_dir.glob("*.jsonl"))
        if not extraction_files:
            LOGGER.debug("No uploaded file extractions to restore")
            return
        
        LOGGER.info("📎 Found %d uploaded files to re-index...", len(extraction_files))

        LOGGER.info("📎 Re-indexing %d uploaded file(s)...", len(extraction_files))

        # Restore uploads from JSONLs
        try:
            result = upload_manager.restore_uploads_from_jsonl(self.current_session_id)

            status = result.get("status")
            if status == "restored":
                restored = result.get("restored_count", 0)
                failed = result.get("failed_count", 0)
                total_chunks = result.get("total_chunks", 0)

                if restored > 0:
                    LOGGER.info("✅ Restored %d uploaded files (%d chunks)", restored, total_chunks)
                if failed > 0:
                    LOGGER.warning("⚠️  Failed to restore %d uploaded files", failed)

                # Refresh cache so UI shows restored files
                self.refresh_session_upload_cache(self.current_session_id)
            elif status == "no_uploads":
                LOGGER.debug("No uploads to restore")
            else:
                LOGGER.warning("Upload restoration failed: %s", result.get("reason", "Unknown error"))

        except Exception as exc:
            LOGGER.error("Failed to restore session uploads: %s", exc)
            import traceback
            traceback.print_exc()

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
                # sticky_chunks contains NodeWithScore objects with .node attribute
                chunk_ids = []
                for i, item in enumerate(self.sticky_chunks[:40]):
                    # Handle both NodeWithScore and raw Document objects
                    if hasattr(item, 'node'):
                        # It's a NodeWithScore
                        node = item.node
                    else:
                        # It's a raw Document
                        node = item
                    
                    # Get ID from node
                    node_id = getattr(node, 'id_', getattr(node, 'node_id', f"chunk_{i}"))
                    chunk_ids.append(node_id)
                
                context_state["cached_chunk_ids"] = chunk_ids
                LOGGER.debug("Captured %d cached chunk IDs for context restoration", len(chunk_ids))
            
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

    def documents_grouped_by_type(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return {doc_type: [(display_title, source_filename), ...]} sorted by title."""
        grouped: Dict[str, Dict[str, str]] = defaultdict(dict)  # {type: {source: title}}
        for node in self.nodes:
            metadata = node.metadata
            source = metadata.get("source", "")
            if not source or source in grouped.get(
                str(metadata.get("doc_type", "UNCATEGORIZED")).upper(), {}
            ):
                continue

            doc_type = str(metadata.get("doc_type", "UNCATEGORIZED")).upper()
            title = metadata.get("title") or source or "Untitled"

            # Handle FORM documents - avoid double form numbers
            if doc_type == "FORM":
                form_number = metadata.get("form_number")
                if form_number:
                    form_normalized = form_number.replace(" ", "").upper()
                    title_start = title.split("-")[0].strip().replace(" ", "").upper()
                    if not title_start.startswith(form_normalized):
                        title = f"{form_number} - {title}"

            grouped[doc_type][source] = title

        return {
            doc_type: sorted(
                [(title, source) for source, title in sources.items()],
                key=lambda t: t[0],
            )
            for doc_type, sources in grouped.items()
        }

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
        Get node map for hierarchical retrieval.
        
        Builds a map of node_id -> NodeWithScore for looking up
        specific chunks by ID.
        
        Note: This still loads nodes into memory for the map.
        Consider refactoring hierarchical retrieval to use
        NodeRepository.get_node_by_id() instead for better scaling.
        """
        if self._node_map_cache is None:
            from llama_index.core.schema import NodeWithScore

            tenant_id = self.tenant_id
            
            # Load nodes from PostgreSQL
            repo = NodeRepository(tenant_id=tenant_id)
            nodes = repo.get_all_nodes()
            
            node_map: Dict[str, Any] = {}
            for item in nodes:
                node_id = getattr(item, 'node_id', None) or getattr(item, 'id_', None)
                if node_id:
                    node_map[node_id] = NodeWithScore(node=item, score=0.5)
            
            self._node_map_cache = node_map
            LOGGER.info("Built node map cache with %d entries", len(node_map))
        
        return self._node_map_cache

    def invalidate_node_map_cache(self) -> None:
        """
        Invalidate all node-related caches.
        
        Call this after index rebuild, document changes, or tenant change.
        """
        self._node_map_cache = None
        self.nodes = []  # Clear lazy-loaded nodes
        
        # Clear retrievers so they get recreated with correct tenant
        self.fts_retriever = None
        self.bm25_retriever = None
        self.vector_retriever = None
        
        LOGGER.debug("Node caches and retrievers invalidated")


__all__ = ["AppState"]
