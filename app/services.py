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

from .vector_store import get_qdrant_client, ensure_collection
from llama_index.core.schema import TextNode

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
    Verify and repair consistency between SQLite and Qdrant.
    
    Qdrant is source of truth for vectors. SQLite mirrors text/metadata.
    Compares TOTAL counts across all tenants, then rebuilds SQLite if needed.
    """
    from llama_index.core.schema import TextNode
    
    from .database import rebuild_fts_index, get_node_count, db_connection
    from .nodes import NodeRepository, bulk_insert_nodes
    from .indexing import clear_all_nodes_for_rebuild
    
    config = AppConfig.get()
    
    try:
        # Get TOTAL SQLite count (all tenants)
        with db_connection() as conn:
            sqlite_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        
        # Get TOTAL Qdrant count
        qdrant_client = get_qdrant_client()
        collection_name = ensure_collection(qdrant_client)
        collection_info = qdrant_client.get_collection(collection_name)
        qdrant_count = collection_info.points_count
        
        LOGGER.info("Sync check: SQLite=%d (all tenants), Qdrant=%d", sqlite_count, qdrant_count)
        
        # If counts match (within tolerance), just rebuild FTS
        if abs(sqlite_count - qdrant_count) <= 5:
            rebuild_fts_index()
            return SyncMemoryResult(
                old_count=sqlite_count,
                new_count=sqlite_count,
                removed=0,
                success=True
            )
        
        # Counts don't match — rebuild SQLite from Qdrant
        LOGGER.warning("Count mismatch (%d vs %d) — rebuilding SQLite from Qdrant",
                       sqlite_count, qdrant_count)
        
        # Fetch everything from Qdrant
        all_points = []
        offset = None
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset
        
        # Clear ALL SQLite nodes
        clear_all_nodes_for_rebuild()
        
        # Rebuild with correct tenant_id per node
        nodes = []
        for point in all_points:
            metadata = {k: v for k, v in point.payload.items() if k != "_node_content"}
            text = point.payload.get("text", "") or point.payload.get("_node_content", "") or point.payload.get("document", "")
            
            # Helper to extract text if it's buried in _node_content JSON string
            if not text and "_node_content" in point.payload:
                 try:
                     import json
                     content_dict = json.loads(point.payload["_node_content"])
                     text = content_dict.get("text", "")
                 except:
                     pass

            node = TextNode(
                id_=str(point.id),
                text=text,
                metadata=metadata,
            )
            nodes.append(node)
        
        # Group by tenant and insert
        nodes_by_tenant = {}
        for node in nodes:
            tid = node.metadata.get("tenant_id", "shared")
            nodes_by_tenant.setdefault(tid, []).append(node)
        
        total_inserted = 0
        for tid, tenant_nodes in nodes_by_tenant.items():
            inserted = bulk_insert_nodes(tenant_nodes, tenant_id=tid)
            total_inserted += inserted
            LOGGER.info("Inserted %d nodes for tenant '%s'", inserted, tid)
        
        # Rebuild FTS
        rebuild_fts_index()
        
        # Invalidate caches
        app_state.invalidate_node_map_cache()
        app_state.fts5_retriever = None
        app_state.ensure_retrievers()
        
        LOGGER.info("Synced SQLite to Qdrant: %d → %d nodes", sqlite_count, total_inserted)
        
        return SyncMemoryResult(
            old_count=sqlite_count,
            new_count=total_inserted,
            removed=sqlite_count - total_inserted if sqlite_count > total_inserted else 0,
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
    Rebuild document trees from existing Gemini caches across all tenants.
    
    Loads cached extractions from all per-tenant JSONL files and builds
    hierarchical tree structures for navigation and retrieval.
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
        # Load Gemini cache from ALL per-tenant JSONLs
        LOGGER.info("Loading Gemini extraction caches...")
        cached_records = {}
        
        for jsonl_path in paths.cache_dir.glob("gemini_extract_cache_*.jsonl"):
            tenant_records = load_jsonl(jsonl_path)
            cached_records.update(tenant_records)
            LOGGER.info("Loaded %d records from %s", len(tenant_records), jsonl_path.name)
        
        # Also try legacy single cache as fallback
        if not cached_records and paths.gemini_json_cache.exists():
            cached_records = load_jsonl(paths.gemini_json_cache)
            LOGGER.info("Loaded %d records from legacy cache", len(cached_records))
        
        if not cached_records:
            return RebuildTreesResult(
                trees_built=0,
                files_skipped=0,
                errors=["No Gemini cache found"],
                success=False,
                error="No Gemini cache found. Please rebuild the index first."
            )
        
        LOGGER.info("Found %d cached extractions across all tenants", len(cached_records))
        
        # Build trees for all files with valid extractions
        document_trees = []
        skipped = 0
        errors = []
        
        for filename, cached_record in cached_records.items():
            gemini_meta = cached_record.get("gemini", {})
            
            if "parse_error" in gemini_meta or "extraction_error" in gemini_meta:
                LOGGER.debug("Skipping %s (extraction error)", filename)
                skipped += 1
                continue
            
            if not gemini_meta.get("sections"):
                LOGGER.debug("Skipping %s (no sections)", filename)
                skipped += 1
                continue
            
            # Resolve doc path through tenant folders
            doc_path = config.find_doc_file(filename)
            if not doc_path:
                doc_path = paths.docs_path / filename  # fallback for tree building
            
            doc_id = doc_path.stem
            
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
        
        if app_state.nodes:
            LOGGER.info("Mapping %d chunks to tree sections...", len(app_state.nodes))
            document_trees = map_chunks_to_tree_sections(app_state.nodes, document_trees)
            LOGGER.info("Chunk mapping complete")
        else:
            LOGGER.warning("No nodes loaded - skipping chunk mapping")
        
        trees_path = paths.cache_dir / "document_trees.json"
        save_document_trees(document_trees, trees_path)
        LOGGER.info("Saved trees to %s", trees_path)
        
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

def _resolve_tenant_for_file(source_filename: str, app_state: "AppState") -> str:
    """Look up which tenant owns a file by checking node metadata.
    
    Falls back to 'shared' if no node found (e.g. file failed extraction).
    """
    for node in app_state.nodes:
        if node.metadata.get("source") == source_filename:
            return node.metadata.get("tenant_id", "shared")
    return "shared"

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
     
    # Resolve tenant to find correct subfolder
    if not tenant_id:
        tenant_id = _resolve_tenant_for_file(source_filename, app_state)
     
    source_path = config.docs_path_for(tenant_id) / source_filename
    
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
    
    # Step 1: Delete all files from disk
    deleted_count = 0
    for filename in filenames:
        # Resolve path via tenant metadata or find across folders
        file_path = config.find_doc_file(filename)
        if file_path and file_path.exists():
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
    Nuclear option: delete ALL documents across ALL tenants and reset completely.
    
    This operation:
    1. Deletes all source files from ALL tenant subfolders
    2. Clears ChromaDB collection
    3. Clears ALL SQLite nodes (all tenants)
    4. Deletes all cache files (including per-tenant JSONLs)
    5. Resets app_state completely
    """
    from .nodes import NodeRepository
    from .indexing import clear_all_nodes_for_rebuild
    
    config = AppConfig.get()
    
    try:
        docs_base = config.paths.docs_path
        deleted_count = 0
        
        # Delete files from ALL tenant subfolders
        for tenant_dir in docs_base.iterdir():
            if tenant_dir.is_dir():
                for file_path in tenant_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_count += 1
        
        # Also clean up any legacy files directly in docs/ (pre-migration)
        for file_path in docs_base.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
        
        # Clear Qdrant (all tenants — single collection)
        manager = app_state.ensure_manager()
        manager.qdrant_client.delete_collection(manager.collection_name)
        manager.collection_name = ensure_collection(manager.qdrant_client)
        
        # Clear ALL SQLite nodes (all tenants)
        clear_all_nodes_for_rebuild()
        
        # Delete all per-tenant Gemini caches
        cache_dir = config.paths.cache_dir
        for jsonl in cache_dir.glob("gemini_extract_cache_*.jsonl"):
            jsonl.unlink()
            LOGGER.info("Deleted cache: %s", jsonl.name)
        
        # Delete legacy single cache if present
        if config.paths.gemini_json_cache.exists():
            config.paths.gemini_json_cache.unlink()
        
        # Delete per-tenant sync caches
        for sync_cache in cache_dir.glob("sync_cache_*.json"):
            sync_cache.unlink()
            LOGGER.info("Deleted sync cache: %s", sync_cache.name)
        
        # Delete other cache files
        other_caches = [
            config.paths.nodes_cache_path,   # Legacy pickle
            config.paths.cache_info_path,
            cache_dir / "sync_cache.json",   # Legacy single sync cache
        ]
        for cache_file in other_caches:
            if cache_file.exists():
                cache_file.unlink()
        
        # Reset state completely
        app_state._nodes_cache = None
        app_state.index = None
        app_state.vector_retriever = None
        app_state.fts5_retriever = None
        app_state.bm25_retriever = None
        app_state._managers.clear()
        app_state.invalidate_node_map_cache()
        
        LOGGER.info("Nuclear delete: %d files across all tenants", deleted_count)
        
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

import re

def sanitize_markdown_tables(text: str) -> str:
    """
    Fix degenerate markdown tables from LLM output.
    
    Handles:
    - Separator rows with hundreds of dashes (truncate to sane width)
    - Cell content with runaway repeated characters
    """
    lines = text.split('\n')
    sanitized = []
    
    for line in lines:
        if not line.strip().startswith('|'):
            sanitized.append(line)
            continue
        
        # Split into cells
        cells = line.split('|')
        fixed_cells = []
        
        for cell in cells:
            stripped = cell.strip()
            
            # Fix separator cells: :--...-- or ---...--- (more than 20 dashes)
            sep_match = re.match(r'^(:?)-{4,}(:?)$', stripped)
            if sep_match:
                left = sep_match.group(1)   # ':' or ''
                right = sep_match.group(2)  # ':' or ''
                fixed_cells.append(f' {left}---{right} ')
                continue
            
            # Fix runaway repeated characters in content cells (>200 chars of same char)
            if len(stripped) > 200:
                collapsed = re.sub(r'(.)\1{50,}', lambda m: m.group(1) * 3, stripped)
                if len(collapsed) < len(stripped):
                    fixed_cells.append(f' {collapsed} ')
                    continue
            
            fixed_cells.append(cell)
        
        sanitized.append('|'.join(fixed_cells))
    
    return '\n'.join(sanitized)

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
    "sanitize_markdown_tables",
]
