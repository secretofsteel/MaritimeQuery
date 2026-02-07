"""Business logic services extracted from UI layer.

This module contains pure business logic for library management operations.
Functions here have NO Streamlit dependencies and return results that the
UI layer can display however it wants.

This separation enables:
- Unit testing without Streamlit
- Reuse in CLI tools or API endpoints
- Cleaner UI code that focuses on presentation
- Easier migration to FastAPI later
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import chromadb
from llama_index.core.schema import TextNode

import streamlit as st

from .config import AppConfig
from .logger import LOGGER

if TYPE_CHECKING:
    from .state import AppState


# ==============================================================================
# RESULT TYPES
# ==============================================================================

@dataclass
class SyncMemoryResult:
    """Result of sync_memory_to_db operation."""
    old_count: int
    new_count: int
    removed: int
    success: bool
    error: Optional[str] = None


@dataclass
class RebuildTreesResult:
    """Result of rebuild_document_trees operation."""
    trees_built: int
    files_skipped: int
    errors: List[str]
    success: bool
    error: Optional[str] = None


@dataclass 
class DeleteDocumentResult:
    """Result of delete_document operation."""
    success: bool
    deleted_from_disk: bool
    sync_result: Optional[Any] = None  # SyncResult from indexing
    error: Optional[str] = None


@dataclass
class DeleteLibraryResult:
    """Result of delete_entire_library operation."""
    files_deleted: int
    success: bool
    error: Optional[str] = None


@dataclass
class BulkUploadResult:
    """Result of bulk upload operation."""
    files_copied: int
    files_skipped: int
    duplicates_deleted: int
    success: bool
    error: Optional[str] = None


# ==============================================================================
# SYNC MEMORY TO DB
# ==============================================================================

def sync_memory_to_db(app_state: "AppState") -> SyncMemoryResult:
    """
    Verify and repair consistency between SQLite and ChromaDB.
    
    In Phase 3+, SQLite is the source of truth for text/metadata.
    ChromaDB is the source of truth for vectors.
    
    This function:
    1. Compares node counts between SQLite and ChromaDB
    2. If mismatched, rebuilds SQLite from ChromaDB (vectors are expensive)
    3. Rebuilds FTS5 index
    
    Use when you suspect data inconsistency.
    
    Args:
        app_state: Application state (for ChromaDB access)
    
    Returns:
        SyncMemoryResult with counts and status
    """
    import streamlit as st
    from llama_index.core.schema import TextNode
    import chromadb
    
    from .database import rebuild_fts_index, get_node_count
    from .nodes import NodeRepository, bulk_insert_nodes
    
    config = AppConfig.get()
    tenant_id = st.session_state.get("tenant_id", "shared")
    
    try:
        # Get counts
        sqlite_count = get_node_count(tenant_id)
        
        # Get ChromaDB count
        client = chromadb.PersistentClient(path=str(config.paths.chroma_path))
        collection = client.get_collection("maritime_docs")
        chroma_count = collection.count()
        
        LOGGER.info("Sync check: SQLite=%d, ChromaDB=%d", sqlite_count, chroma_count)
        
        # If counts match (within tolerance), just rebuild FTS
        if abs(sqlite_count - chroma_count) <= 5:
            rebuild_fts_index()
            return SyncMemoryResult(
                old_count=sqlite_count,
                new_count=sqlite_count,
                removed=0,
                success=True
            )
        
        # Counts don't match - rebuild SQLite from ChromaDB
        LOGGER.warning("Count mismatch - rebuilding SQLite from ChromaDB")
        
        # Fetch everything from ChromaDB
        result = collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        # Build nodes
        nodes = []
        for i, chunk_id in enumerate(result['ids']):
            node = TextNode(
                id_=chunk_id,
                text=result['documents'][i],
                metadata=result['metadatas'][i],
                # Note: embeddings stay in ChromaDB, not SQLite
            )
            nodes.append(node)
        
        # Clear and rebuild SQLite
        repo = NodeRepository(tenant_id=tenant_id)
        repo.clear_all()
        inserted = bulk_insert_nodes(nodes, tenant_id=tenant_id)
        
        # Rebuild FTS
        rebuild_fts_index()
        
        # Invalidate caches
        app_state.invalidate_node_map_cache()
        app_state.fts5_retriever = None
        app_state.ensure_retrievers()
        
        new_count = get_node_count(tenant_id)
        
        LOGGER.info("Synced SQLite to ChromaDB: %d → %d nodes", sqlite_count, new_count)
        
        return SyncMemoryResult(
            old_count=sqlite_count,
            new_count=new_count,
            removed=sqlite_count - new_count if sqlite_count > new_count else 0,
            success=True
        )
        
    except Exception as exc:
        LOGGER.exception("Failed to sync memory to DB")
        return SyncMemoryResult(
            old_count=0,
            new_count=0,
            removed=0,
            success=False,
            error=str(exc)
        )


# ==============================================================================
# REBUILD DOCUMENT TREES
# ==============================================================================

def rebuild_document_trees(app_state: "AppState") -> RebuildTreesResult:
    """
    Rebuild document trees from existing Gemini cache without touching index.
    
    This is FAST (~5 seconds) because it:
    - Doesn't re-extract documents (uses cached Gemini extractions)
    - Doesn't re-chunk documents
    - Doesn't re-embed chunks
    - Just rebuilds tree JSON structure from existing cache
    
    Args:
        app_state: Application state (for nodes mapping)
    
    Returns:
        RebuildTreesResult with counts and status
    """
    from .indexing import (
        load_jsonl,
        save_document_trees,
        map_chunks_to_tree_sections
    )
    from .extraction import build_document_tree
    
    config = AppConfig.get()
    paths = config.paths
    
    try:
        # Load Gemini cache
        LOGGER.info("Loading Gemini extraction cache...")
        cached_records = load_jsonl(paths.gemini_json_cache)
        
        if not cached_records:
            return RebuildTreesResult(
                trees_built=0,
                files_skipped=0,
                errors=["No Gemini cache found"],
                success=False,
                error="No Gemini cache found. Please rebuild the index first."
            )
        
        LOGGER.info("Found %d cached extractions", len(cached_records))
        
        # Build trees for all files with valid extractions
        document_trees = []
        skipped = 0
        errors = []
        
        for filename, cached_record in cached_records.items():
            gemini_meta = cached_record.get("gemini", {})
            
            # Skip files with extraction errors
            if "parse_error" in gemini_meta or "extraction_error" in gemini_meta:
                LOGGER.debug("Skipping %s (extraction error)", filename)
                skipped += 1
                continue
            
            # Skip files with no sections
            if not gemini_meta.get("sections"):
                LOGGER.debug("Skipping %s (no sections)", filename)
                skipped += 1
                continue
            
            # Build tree for this document
            doc_path = paths.docs_path / filename
            doc_id = doc_path.stem  # Filename without extension
            
            try:
                tree = build_document_tree(gemini_meta, doc_id)
                document_trees.append(tree)
                LOGGER.debug("Built tree for %s with %d sections",
                            doc_id, len(tree.get("sections", [])))
            except Exception as exc:
                LOGGER.exception("Failed to build tree for %s", filename)
                errors.append(f"{filename}: {str(exc)}")
                skipped += 1
                continue
        
        LOGGER.info("Built %d document trees (%d files skipped)",
                    len(document_trees), skipped)
        
        # Map chunks to tree sections (if nodes available)
        if app_state.nodes:
            LOGGER.info("Mapping %d chunks to tree sections...", len(app_state.nodes))
            document_trees = map_chunks_to_tree_sections(app_state.nodes, document_trees)
            LOGGER.info("Chunk mapping complete")
        else:
            LOGGER.warning("No nodes loaded - skipping chunk mapping")
        
        # Save trees to JSON
        trees_path = paths.cache_dir / "document_trees.json"
        save_document_trees(document_trees, trees_path)
        LOGGER.info("Saved trees to %s", trees_path)
        
        # Reload hierarchical state in app
        app_state._validate_hierarchical_support()
        
        return RebuildTreesResult(
            trees_built=len(document_trees),
            files_skipped=skipped,
            errors=errors,
            success=True
        )
        
    except Exception as exc:
        LOGGER.exception("Failed to rebuild document trees")
        return RebuildTreesResult(
            trees_built=0,
            files_skipped=0,
            errors=[str(exc)],
            success=False,
            error=str(exc)
        )


# ==============================================================================
# DELETE DOCUMENT
# ==============================================================================

def delete_document_by_source(
    source_filename: str,
    app_state: "AppState",
    tenant_id: str = None
) -> DeleteDocumentResult:
    """
    Delete a single document by source filename.
    
    This operation:
    1. Removes from ChromaDB, nodes, and Gemini cache
    2. Deletes source file from disk
    3. Runs sync_library for consistency
    4. Updates app_state
    
    Args:
        source_filename: Filename of the source document
        app_state: Application state to update
    
    Returns:
        DeleteDocumentResult with status
    """
    if not source_filename:
        return DeleteDocumentResult(
            success=False,
            deleted_from_disk=False,
            error="Empty source filename"
        )
    
    config = AppConfig.get()
    source_path = config.paths.docs_path / source_filename
    
    try:
        manager = app_state.ensure_manager()
        if tenant_id:
            manager.tenant_id = tenant_id
        manager.nodes = app_state.nodes
        
        # Remove from database (ChromaDB, nodes, Gemini cache)
        manager._remove_documents({source_filename})
        
        # Delete source file
        deleted_from_disk = False
        if source_path.exists():
            source_path.unlink()
            deleted_from_disk = True
            LOGGER.info("Deleted source file: %s", source_path)
        else:
            LOGGER.warning("Source file not found: %s", source_path)
        
        # Update app state
        app_state.nodes = manager.nodes
        app_state.invalidate_node_map_cache()
        
        LOGGER.info("Deleted document: %s", source_filename)
        
        # Run full sync for consistency
        try:
            changes, report = manager.sync_library(app_state.index)
            
            app_state.nodes = manager.nodes
            app_state.invalidate_node_map_cache()
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
            
            LOGGER.info("Synced after deletion: +%d, ~%d, -%d",
                       len(changes.added), len(changes.modified), len(changes.deleted))
            
            return DeleteDocumentResult(
                success=True,
                deleted_from_disk=deleted_from_disk,
                sync_result=changes
            )
            
        except Exception as sync_exc:
            LOGGER.warning("Sync after deletion failed: %s", sync_exc)
            # Try lightweight retriever rebuild as fallback
            try:
                app_state.ensure_retrievers()
            except Exception:
                pass
            
            return DeleteDocumentResult(
                success=True,
                deleted_from_disk=deleted_from_disk,
                error=f"Sync failed: {sync_exc}"
            )
        
    except Exception as exc:
        LOGGER.exception("Delete failed for %s", source_filename)
        return DeleteDocumentResult(
            success=False,
            deleted_from_disk=False,
            error=str(exc)
        )


def batch_delete_documents(
    filenames: List[str],
    app_state: "AppState"
) -> Tuple[int, int, Optional[str]]:
    """
    Delete multiple documents and sync once at the end.
    
    Args:
        filenames: List of filenames to delete
        app_state: Application state to update
    
    Returns:
        Tuple of (deleted_count, db_entries_removed, error_message)
    """
    config = AppConfig.get()
    docs_path = config.paths.docs_path
    
    # Step 1: Delete all files from disk
    deleted_count = 0
    for filename in filenames:
        file_path = docs_path / filename
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_count += 1
                LOGGER.info("Deleted file: %s", filename)
            except Exception as exc:
                LOGGER.exception("Failed to delete %s", filename)
        else:
            LOGGER.warning("File not found: %s", filename)
    
    # Step 2: Sync library once to clean up database
    db_entries_removed = 0
    if deleted_count > 0:
        try:
            manager = app_state.ensure_manager()
            manager.nodes = app_state.nodes
            
            sync_result, _ = manager.sync_library(
                app_state.index,
                force_retry_errors=False
            )
            
            # Update app state
            app_state.nodes = manager.nodes
            app_state.invalidate_node_map_cache()
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
            
            db_entries_removed = len(sync_result.deleted)
            LOGGER.info("Batch deleted %d files, removed %d from DB",
                       deleted_count, db_entries_removed)
            
            return deleted_count, db_entries_removed, None
            
        except Exception as exc:
            LOGGER.exception("Sync after batch delete failed")
            return deleted_count, 0, str(exc)
    
    return deleted_count, db_entries_removed, None


# ==============================================================================
# DELETE ENTIRE LIBRARY
# ==============================================================================

def delete_entire_library(app_state: "AppState") -> DeleteLibraryResult:
    """
    Nuclear option: delete ALL documents and reset completely.
    
    This operation:
    1. Deletes all source files from docs directory
    2. Clears ChromaDB collection
    3. Clears SQLite nodes table
    4. Deletes all cache files
    5. Resets app_state completely
    
    Args:
        app_state: Application state to reset
    
    Returns:
        DeleteLibraryResult with status
    """
    import streamlit as st
    from .nodes import NodeRepository
    
    config = AppConfig.get()
    tenant_id = st.session_state.get("tenant_id", "shared")
    
    try:
        # Delete all source files
        docs_path = config.paths.docs_path
        deleted_count = 0
        
        for file_path in docs_path.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
        
        # Clear ChromaDB
        manager = app_state.ensure_manager()
        manager.collection.delete()
        manager.collection = manager.chroma_client.get_or_create_collection("maritime_docs")
        
        # Clear SQLite nodes
        repo = NodeRepository(tenant_id=tenant_id)
        repo.clear_all()
        
        # Clear all cache files
        cache_files = [
            config.paths.nodes_cache_path,  # Legacy pickle (may not exist)
            config.paths.cache_info_path,
            manager.sync_cache_file,
            config.paths.gemini_json_cache,
        ]
        
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
        
        # Reset state completely
        app_state._nodes_cache = None
        app_state.index = None
        app_state.vector_retriever = None
        app_state.fts5_retriever = None
        app_state.bm25_retriever = None
        app_state.manager = None
        app_state.invalidate_node_map_cache()
        
        LOGGER.info("Nuclear delete: %d files", deleted_count)
        
        return DeleteLibraryResult(
            files_deleted=deleted_count,
            success=True
        )
        
    except Exception as exc:
        LOGGER.exception("Nuclear delete failed")
        return DeleteLibraryResult(
            files_deleted=0,
            success=False,
            error=str(exc)
        )


# ==============================================================================
# BULK UPLOAD HELPERS
# ==============================================================================

def copy_uploaded_files(
    uploaded_files: List[Any],
    docs_path: Path
) -> Tuple[int, int, List[str]]:
    """
    Copy uploaded files to the docs directory.
    
    Args:
        uploaded_files: Streamlit UploadedFile objects
        docs_path: Destination directory
    
    Returns:
        Tuple of (copied_count, skipped_count, failed_filenames)
    """
    docs_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    failed = []
    
    for uploaded_file in uploaded_files:
        try:
            file_path = docs_path / uploaded_file.name
            with file_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())
            copied_count += 1
        except Exception as exc:
            LOGGER.exception("Failed to copy %s", uploaded_file.name)
            failed.append(uploaded_file.name)
            skipped_count += 1
    
    return copied_count, skipped_count, failed


def delete_duplicate_files(
    filenames: List[str],
    docs_path: Path
) -> int:
    """
    Delete files that are being overwritten.
    
    Args:
        filenames: List of filenames to delete
        docs_path: Directory containing files
    
    Returns:
        Number of files deleted
    """
    deleted = 0
    for filename in filenames:
        file_path = docs_path / filename
        if file_path.exists():
            file_path.unlink()
            deleted += 1
    return deleted


__all__ = [
    # Result types
    "SyncMemoryResult",
    "RebuildTreesResult", 
    "DeleteDocumentResult",
    "DeleteLibraryResult",
    "BulkUploadResult",
    # Functions
    "sync_memory_to_db",
    "rebuild_document_trees",
    "delete_document_by_source",
    "batch_delete_documents",
    "delete_entire_library",
    "copy_uploaded_files",
    "delete_duplicate_files",
]
