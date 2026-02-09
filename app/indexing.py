"""Index creation, chunking, and incremental updates with status tracking."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import Document

from .config import AppConfig
from .constants import CHUNK_OVERLAP, CHUNK_SIZE
from .metadata_updates import apply_corrections_to_gemini_record
from .extraction import gemini_extract_record, to_documents_from_gemini, build_document_tree
from .files import current_files_index, load_jsonl, upsert_jsonl_record, write_jsonl
from .logger import LOGGER
from .parallel_processing import ParallelDocumentProcessor, ParallelEmbeddingGenerator
from .processing_status import (
    DocumentProcessingStatus,
    ProcessingReport,
    StageResult,
    StageStatus,
    save_processing_report,
)
from .nodes import NodeRepository, bulk_insert_nodes
from .database import init_db, rebuild_fts_index

def clear_all_nodes_for_rebuild() -> int:
    """Clear ALL nodes from SQLite regardless of tenant."""
    from .database import db_connection
    
    with db_connection() as conn:
        cursor = conn.execute("DELETE FROM nodes")
        deleted = cursor.rowcount
    
    LOGGER.warning("Cleared ALL %d nodes from SQLite for rebuild", deleted)
    return deleted

def chunk_documents(documents: Iterable[Document], assign_section_ids: bool = True) -> List[Document]:
    """
    Split LlamaIndex documents into sentence-aware chunks with metadata headers.

    Args:
        documents: Documents to chunk
        assign_section_ids: If True, assign section_id to chunk metadata based on section name

    Returns:
        List of chunked documents with section_ids in metadata
    """
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes: List[Document] = []
    for document in documents:
        metadata = document.metadata
        header_parts = [f"Document: {metadata.get('source','Unknown')}"]
        if metadata.get("title"):
            header_parts.append(f"Title: {metadata['title']}")
        if metadata.get("doc_type"):
            header_parts.append(f"Type: {metadata['doc_type']}")
        if metadata.get("form_category_name") and metadata.get("form_number"):
            header_parts.append(f"Form: {metadata['form_number']} - {metadata['form_category_name']}")
        if metadata.get("hierarchy"):
            header_parts.append(f"Hierarchy: {metadata['hierarchy']}")
        if metadata.get("section") and metadata["section"] != "Full Document Content":
            header_parts.append(f"Section: {metadata['section']}")
        header = "\n".join(header_parts)

        # Extract section_id from section name if enabled
        section_id = None
        if assign_section_ids and metadata.get("section"):
            section_name = metadata["section"]
            # Use extraction function for consistency
            from .extraction import _parse_section_identifier
            parsed = _parse_section_identifier(section_name)
            if parsed:
                section_id = parsed["section_id"]

        for chunk_text in parser.split_text(document.text):
            full_text = f"{header}\n---\n{chunk_text}"
            chunk_metadata = {**metadata, "chunk_text_len": len(chunk_text)}

            # Add section_id to metadata if available
            if section_id:
                chunk_metadata["section_id"] = section_id

            nodes.append(Document(text=full_text, metadata=chunk_metadata))
    return nodes


def cache_nodes(
    nodes: List[Document], 
    documents_count: int, 
    cache_path: Path, 
    info_path: Path,
    tenant_id: str = "shared",
    skip_clear: bool = False
) -> None:
    """
    Save nodes to SQLite database (replaces pickle storage).
    
    Also saves cache info JSON for compatibility with existing code.
    
    Args:
        nodes: List of TextNode/Document objects to save
        documents_count: Number of source documents
        cache_path: Legacy path (ignored, kept for signature compatibility)
        info_path: Path for cache info JSON
        tenant_id: Tenant identifier for multi-tenancy
    """
    from .nodes import bulk_insert_nodes
    from .database import init_db, rebuild_fts_index
    
    # Ensure database is initialized
    init_db()
    
    # Clear existing nodes for this tenant and insert new ones
    if not skip_clear:
        repo = NodeRepository(tenant_id=tenant_id)
        repo.clear_all()
    
    # Bulk insert for performance
    inserted = bulk_insert_nodes(nodes, tenant_id=tenant_id)
    
    # Rebuild FTS index after bulk insert
    rebuild_fts_index()
    
    # Save cache info (for compatibility and debugging)
    cache_info = {
        "timestamp": datetime.now().isoformat(),
        "num_nodes": len(nodes),
        "num_documents": documents_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "storage": "sqlite",  # Flag that we're using SQLite now
        "tenant_id": tenant_id,
    }
    with info_path.open("w") as file:
        json.dump(cache_info, file, indent=2)
    
    LOGGER.info("Saved %d nodes to SQLite (tenant=%s)", inserted, tenant_id)


def save_document_trees(trees: List[Dict[str, Any]], trees_path: Path) -> None:
    """
    Save document trees to JSON file.

    Args:
        trees: List of document tree structures
        trees_path: Path to save JSON file (data/document_trees.json)
    """
    trees_path.parent.mkdir(parents=True, exist_ok=True)

    with trees_path.open("w", encoding="utf-8") as f:
        json.dump(trees, f, indent=2, ensure_ascii=False)

    LOGGER.info("Saved %d document trees to %s", len(trees), trees_path)


def load_document_trees(trees_path: Path) -> List[Dict[str, Any]]:
    """
    Load document trees from JSON file.

    Args:
        trees_path: Path to document trees JSON file

    Returns:
        List of document tree structures, empty list if file doesn't exist
    """
    if not trees_path.exists():
        LOGGER.debug("No document trees file found at %s", trees_path)
        return []

    try:
        with trees_path.open("r", encoding="utf-8") as f:
            trees = json.load(f)
        LOGGER.info("Loaded %d document trees from %s", len(trees), trees_path)
        return trees
    except Exception as exc:
        LOGGER.error("Failed to load document trees: %s", exc)
        return []


def map_chunks_to_tree_sections(
    nodes: List[Document],
    trees: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Map chunk IDs to document tree sections based on section_id metadata.

    Updates tree structures in-place by populating chunk_ids arrays.

    Args:
        nodes: Chunked documents with section_id in metadata
        trees: Document tree structures to populate

    Returns:
        Updated trees with chunk_ids populated
    """
    # Build a map of (doc_id, section_id) -> tree_section for fast lookup
    section_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def register_section(tree_section: Dict[str, Any], doc_id: str):
        """Recursively register all sections in the map."""
        section_id = tree_section.get("section_id")
        if section_id:
            section_map[(doc_id, section_id)] = tree_section

        # Register children recursively
        for child in tree_section.get("children", []):
            register_section(child, doc_id)

    # Register all sections from all trees
    for tree in trees:
        doc_id = tree.get("doc_id", "")
        for section in tree.get("sections", []):
            register_section(section, doc_id)

    # Map chunks to sections
    for idx, node in enumerate(nodes):
        metadata = node.metadata
        source = metadata.get("source", "")
        section_id = metadata.get("section_id")

        if not section_id:
            continue

        # Use source filename without extension as doc_id
        doc_id = Path(source).stem if source else ""

        # Find matching section
        key = (doc_id, section_id)
        if key in section_map:
            tree_section = section_map[key]
            # Use node_id or create a unique ID
            chunk_id = getattr(node, 'node_id', f"chunk_{idx:04d}")
            tree_section["chunk_ids"].append(chunk_id)

    LOGGER.info("Mapped %d chunks to tree sections", len(nodes))
    return trees


def build_index_from_library() -> Tuple[List[Document], VectorStoreIndex]:
    """Original non-parallel version with corrections support."""
    config = AppConfig.get()
    paths = config.paths

    LOGGER.info("Processing library via Gemini...")
    documents: List[Document] = []
    files_index = current_files_index(paths.docs_path)
    cached_records = load_jsonl(paths.gemini_json_cache)

    for filename, fingerprint in files_index.items():
        doc_path = paths.docs_path / filename
        cached = cached_records.get(filename)
        needs_updates = not (
            cached and abs(cached.get("mtime", 0) - fingerprint["mtime"]) < 1 and cached.get("size") == fingerprint["size"]
        )

        if not needs_updates and cached and "gemini" in cached:
            meta = cached["gemini"]
            
            # *** NEW: Apply corrections if they exist ***
            if "corrections" in cached:
                meta = apply_corrections_to_gemini_record(meta, cached["corrections"])
                LOGGER.debug("Applied corrections to %s", filename)
        else:
            meta = gemini_extract_record(doc_path)

        if needs_updates:
            upsert_jsonl_record(
                paths.gemini_json_cache,
                {"filename": filename, "mtime": fingerprint["mtime"], "size": fingerprint["size"], "gemini": meta},
            )

        if "parse_error" in meta:
            LOGGER.error("Skipping %s due to extraction error: %s", filename, meta.get("parse_error"))
            continue

        documents.extend(to_documents_from_gemini(doc_path, meta))

    LOGGER.info("Loaded %s Gemini-derived document sections", len(documents))
    nodes = chunk_documents(documents)
    LOGGER.info("Created %s chunks", len(nodes))

    chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))
    collection = chroma_client.get_or_create_collection("maritime_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    LOGGER.info("Embedding %s chunks...", len(nodes))
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
    LOGGER.info("Embeddings created and stored.")
    
    # Cache nodes AFTER embedding so embeddings are preserved on reload
    # Note: VectorStoreIndex attaches embeddings to nodes during construction
    cache_nodes(nodes, len(documents), paths.nodes_cache_path, paths.cache_info_path)
    LOGGER.info("Cached nodes with embeddings")

    manager = IncrementalIndexManager(paths.docs_path, paths.gemini_json_cache, paths.nodes_cache_path, paths.cache_info_path, paths.chroma_path)
    manager.sync_cache["files_hash"] = manager._get_files_hash(paths.docs_path)
    manager._save_sync_cache()
    LOGGER.info("Initial sync cache saved.")
    return nodes, index


def build_index_from_library_parallel(
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    clear_gemini_cache: bool = False,
    tenant_id: str = None,  # None = all tenants
) -> Tuple[List[Document], VectorStoreIndex, ProcessingReport]:
    """Process the full document library with parallel extraction and embedding.
    
    Args:
        progress_callback: Optional callback(phase, current, total, item_desc)
        clear_gemini_cache: If True, force re-extraction of all files
        tenant_id: Process specific tenant only, or None for all tenants
    
    Returns:
        Tuple of (nodes, index, processing_report)
    """
    config = AppConfig.get()
    paths = config.paths
    
    start_time = time.time()
    LOGGER.info("=== Starting Parallel Processing ===")
    
    # Clear nodes for rebuild — scoped by tenant if specified
    if tenant_id:
        repo = NodeRepository(tenant_id=tenant_id)
        repo.clear_all()
        LOGGER.info("Cleared nodes for tenant '%s' before rebuild", tenant_id)
    else:
        clear_all_nodes_for_rebuild()
    
    # Determine which tenants to process
    if tenant_id:
        tenant_ids = [tenant_id]
    else:
        tenant_ids = [
            d.name for d in paths.docs_path.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ]
        if not tenant_ids:
            LOGGER.warning("No tenant folders found in %s", paths.docs_path)
            tenant_ids = ["shared"]
    
    LOGGER.info("Processing tenants: %s", tenant_ids)
    
    # Accumulate across all tenants
    all_documents: List[Document] = []
    all_file_statuses: Dict[str, DocumentProcessingStatus] = {}
    all_docs_to_extract: List[Path] = []
    file_tenant_map: Dict[str, str] = {}
    total_files_index: Dict[str, Dict[str, Any]] = {}
    all_cached_records: Dict[str, Any] = {}
    
    # Phase 0: Scan all tenant folders and check caches
    LOGGER.info("Phase 0: Analyzing document library across %d tenants...", len(tenant_ids))
    
    for tid in tenant_ids:
        tenant_docs_path = config.docs_path_for(tid)
        tenant_cache_path = config.gemini_cache_for(tid)
        
        files_index = current_files_index(tenant_docs_path)
        
        if clear_gemini_cache:
            cached_records = {}
        else:
            cached_records = load_jsonl(tenant_cache_path)
        
        LOGGER.info("Tenant '%s': %d files on disk, %d cached extractions",
                    tid, len(files_index), len(cached_records))
        
        for filename, fingerprint in files_index.items():
            doc_path = tenant_docs_path / filename
            cached = cached_records.get(filename)
            
            # Track file status
            status = DocumentProcessingStatus(
                filename=filename,
                file_size_bytes=fingerprint["size"],
            )
            all_file_statuses[filename] = status
            total_files_index[filename] = fingerprint
            file_tenant_map[filename] = tid
            
            status.parsing = StageResult(StageStatus.SUCCESS, "File parsed successfully")
            
            needs_extraction = clear_gemini_cache or not (
                cached 
                and abs(cached.get("mtime", 0) - fingerprint["mtime"]) < 1 
                and cached.get("size") == fingerprint["size"]
            )
            
            if not needs_extraction and cached and "gemini" in cached:
                meta = cached["gemini"]
                status.used_cache = True
                all_cached_records[filename] = cached
                
                # Apply corrections if they exist
                if "corrections" in cached:
                    meta = apply_corrections_to_gemini_record(meta, cached["corrections"])
                    LOGGER.debug("Applied corrections to %s", filename)
                
                if "parse_error" in meta or "extraction_error" in meta:
                    error_msg = meta.get("parse_error") or meta.get("extraction_error")
                    status.extraction = StageResult(StageStatus.FAILED, f"Extraction failed: {error_msg}")
                    LOGGER.error("Skipping %s due to cached extraction error", filename)
                    continue
                
                # Set validation status from cache
                if "validation_error" in meta:
                    validation_data = meta.get("validation", {})
                    coverage = validation_data.get("ngram_coverage", 0)
                    status.extraction = StageResult(StageStatus.SUCCESS, "Extraction succeeded (cached)")
                    status.validation = StageResult(
                        StageStatus.WARNING,
                        f"Low quality extraction (coverage: {coverage:.1%})",
                        details=validation_data
                    )
                else:
                    status.extraction = StageResult(StageStatus.SUCCESS, "Extraction succeeded (cached)")
                    validation_data = meta.get("validation", {})
                    if validation_data:
                        coverage = validation_data.get("ngram_coverage", 0)
                        status.validation = StageResult(
                            StageStatus.SUCCESS,
                            f"Validation passed (coverage: {coverage:.1%})",
                            details=validation_data
                        )
                    else:
                        status.validation = StageResult(StageStatus.SUCCESS, "Validation passed")
                
                status.embedding = StageResult(StageStatus.SUCCESS, "Embeddings cached")
                
                # Convert to documents
                documents_from_cache = to_documents_from_gemini(doc_path, meta)
                all_documents.extend(documents_from_cache)
            else:
                all_docs_to_extract.append(doc_path)
    
    LOGGER.info("Using %d cached extractions, need to extract %d files",
               len(total_files_index) - len(all_docs_to_extract), len(all_docs_to_extract))
    
    # Create processing report
    report = ProcessingReport(
        timestamp=datetime.now().isoformat(),
        total_files=len(total_files_index),
    )
    
    # Add cached file statuses to report
    for filename, status in all_file_statuses.items():
        if filename not in {p.name for p in all_docs_to_extract}:
            report.add_status(status)
    
    # Phase 1: Parallel extraction for uncached files
    if all_docs_to_extract:
        LOGGER.info("Phase 1: Parallel extraction of %d documents...", len(all_docs_to_extract))
        
        processor = ParallelDocumentProcessor(max_workers=10)
        extraction_results = processor.extract_batch(all_docs_to_extract, progress_callback)
        
        # Update per-tenant JSONL caches with results
        for result in extraction_results.successful + extraction_results.failed:
            filename = result.filename
            fingerprint = total_files_index.get(filename, {})
            status = all_file_statuses.get(filename)
            tid = file_tenant_map.get(filename, "shared")
            
            if result.record:
                # Preserve existing tenant_id from cache
                existing_tenant = tid
                
                # Upsert to the correct tenant's JSONL
                tenant_cache_path = config.gemini_cache_for(tid)
                upsert_jsonl_record(
                    tenant_cache_path,
                    {
                        "filename": filename,
                        "mtime": fingerprint.get("mtime", 0),
                        "size": fingerprint.get("size", 0),
                        "gemini": result.record,
                        "tenant_id": existing_tenant,
                    },
                )
                
                # Update status
                if status:
                    if "parse_error" in result.record or "extraction_error" in result.record:
                        error_msg = result.record.get("parse_error") or result.record.get("extraction_error")
                        status.extraction = StageResult(StageStatus.FAILED, f"Extraction failed: {error_msg}")
                    else:
                        status.extraction = StageResult(StageStatus.SUCCESS, "Extraction succeeded")
                        
                        if "validation_error" in result.record:
                            validation_data = result.record.get("validation", {})
                            coverage = validation_data.get("ngram_coverage", 0)
                            status.validation = StageResult(
                                StageStatus.WARNING,
                                f"Low quality extraction (coverage: {coverage:.1%})",
                                details=validation_data
                            )
                        else:
                            validation_data = result.record.get("validation", {})
                            if validation_data:
                                coverage = validation_data.get("ngram_coverage", 0)
                                status.validation = StageResult(
                                    StageStatus.SUCCESS,
                                    f"Validation passed (coverage: {coverage:.1%})",
                                    details=validation_data
                                )
                            else:
                                status.validation = StageResult(StageStatus.SUCCESS, "Validation passed")
        
        # Convert successful extractions to documents
        for result in extraction_results.successful:
            if result.record and "parse_error" not in result.record and "extraction_error" not in result.record:
                tid = file_tenant_map.get(result.filename, "shared")
                doc_path = config.docs_path_for(tid) / result.filename
                documents_from_extraction = to_documents_from_gemini(doc_path, result.record)
                all_documents.extend(documents_from_extraction)
        
        # Mark failures
        for result in extraction_results.failed:
            status = all_file_statuses.get(result.filename)
            if status:
                status.extraction = StageResult(StageStatus.FAILED, f"Extraction error: {result.error}")
            LOGGER.error("Skipping %s due to extraction error: %s", result.filename, result.error)
        
        # Add extraction statuses to report
        for filename in {p.name for p in all_docs_to_extract}:
            if filename in all_file_statuses:
                report.add_status(all_file_statuses[filename])
        
        LOGGER.info("Phase 1 complete: %d successful, %d failed",
                   extraction_results.success_count, extraction_results.failure_count)
    else:
        LOGGER.info("Phase 1 skipped: All documents cached")
    
    LOGGER.info("Total documents loaded: %d", len(all_documents))
    
    # Phase 2: Chunking (unchanged logic)
    LOGGER.info("Phase 2: Chunking documents...")
    nodes = chunk_documents(all_documents)
    LOGGER.info("Created %d chunks", len(nodes))
    
    # Assign tenant_id to chunks
    chunks_per_file: Dict[str, int] = {}
    for node in nodes:
        source = node.metadata.get("source", "")
        if source:
            filename = Path(source).name
            node.metadata["tenant_id"] = file_tenant_map.get(filename, "shared")
            chunks_per_file[filename] = chunks_per_file.get(filename, 0) + 1
        else:
            node.metadata["tenant_id"] = "shared"
    LOGGER.info("Assigned tenant_id to %d nodes", len(nodes))
    
    # Update chunk counts in statuses
    for filename, count in chunks_per_file.items():
        if filename in all_file_statuses:
            all_file_statuses[filename].chunks_created = count
    
    # Phase 3: Parallel embedding generation
    LOGGER.info("Phase 3: Parallel embedding generation...")
    
    # Extract texts for embedding
    chunk_texts = [node.get_content() for node in nodes]
    
    # Generate embeddings in parallel
    embedding_gen = ParallelEmbeddingGenerator(max_workers=5)
    embedding_batch = embedding_gen.generate_batch(chunk_texts, progress_callback)
    
    LOGGER.info("Generated %d embeddings in %.2fs", 
               len(embedding_batch.embeddings), embedding_batch.duration_sec)
    
    # Attach embeddings to nodes BEFORE caching
    for node, embedding in zip(nodes, embedding_batch.embeddings):
        node.embedding = embedding
    
    LOGGER.info("Attached embeddings to %d nodes", len(nodes))
    
    # Cache nodes WITH embeddings attached (so they're available on reload)
    #cache_nodes(nodes, len(all_documents), paths.nodes_cache_path, paths.cache_info_path)
    cache_nodes(
    nodes, 
    len(all_documents), 
    paths.nodes_cache_path,  # Kept for signature, not used
    paths.cache_info_path,
    tenant_id="shared",
    skip_clear=True
    )
    LOGGER.info("Cached nodes with embeddings")
    
    # Mark all files with chunks as having successful embedding
    for filename in chunks_per_file.keys():
        if filename in all_file_statuses:
            all_file_statuses[filename].embedding = StageResult(
                StageStatus.SUCCESS,
                f"Embedded {chunks_per_file[filename]} chunks"
            )
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))

    try:
        chroma_client.delete_collection("maritime_docs")
        LOGGER.info("Deleted old ChromaDB collection")
    except Exception:
        # Collection doesn't exist or other issue - that's fine for fresh start
        LOGGER.info("No existing ChromaDB collection to delete")

    collection = chroma_client.create_collection("maritime_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Write to ChromaDB (sequential - embeddings already attached)
    LOGGER.info("Writing %d chunks to ChromaDB (sequential)...", len(nodes))
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
    LOGGER.info("ChromaDB indexing complete")
    
    # Update sync cache
    manager = IncrementalIndexManager(
        paths.docs_path, 
        paths.gemini_json_cache, 
        paths.nodes_cache_path, 
        paths.cache_info_path, 
        paths.chroma_path
    )
    manager.sync_cache["files_hash"] = manager._get_files_hash(paths.docs_path)
    manager._save_sync_cache()
    LOGGER.info("Sync cache updated")
    
    # Finalize report
    elapsed_time = time.time() - start_time
    report.total_duration_sec = elapsed_time
    
    # Add all file statuses to report
    for status in all_file_statuses.values():
        report.add_status(status)
    
    # Save report
    report_path = paths.cache_dir / "last_processing_report.json"
    save_processing_report(report, report_path)
    LOGGER.info("Processing report saved to %s", report_path)
    
    LOGGER.info("Phase 4: Building document trees...")
    sys.stdout.flush()

    # Import here to avoid circular dependency
    from .extraction import build_document_tree

    try:
        # Reload cache to get all extractions (cached + newly extracted)
        all_cached_records = load_jsonl(paths.gemini_json_cache)
        
        # Build trees for all successfully processed files
        document_trees = []
        
        for filename in files_index.keys():
            if filename not in all_cached_records:
                LOGGER.debug("Skipping tree for %s (no cache entry)", filename)
                continue
            
            cached_record = all_cached_records[filename]
            gemini_meta = cached_record.get("gemini", {})
            
            # Skip files with extraction errors or no sections
            if "parse_error" in gemini_meta:
                LOGGER.debug("Skipping tree for %s (parse error)", filename)
                continue
                
            if "extraction_error" in gemini_meta:
                LOGGER.debug("Skipping tree for %s (extraction error)", filename)
                continue
                
            if not gemini_meta.get("sections"):
                LOGGER.debug("Skipping tree for %s (no sections)", filename)
                continue
            
            # Build tree
            doc_path = paths.docs_path / filename
            doc_id = doc_path.stem  # Filename without extension
            
            try:
                tree = build_document_tree(gemini_meta, doc_id)
                document_trees.append(tree)
                LOGGER.debug("Built tree for %s with %d root sections", 
                            doc_id, len(tree.get("sections", [])))
            except Exception as tree_exc:
                LOGGER.exception("Failed to build tree for %s: %s", filename, tree_exc)
                continue
        
        LOGGER.info("Built %d document trees", len(document_trees))
        
        # Map chunks to tree sections
        if document_trees:
            document_trees = map_chunks_to_tree_sections(nodes, document_trees)
            LOGGER.info("Mapped chunks to tree sections")
        
        # Save trees
        trees_path = paths.cache_dir / "document_trees.json"
        save_document_trees(document_trees, trees_path)
        
        LOGGER.info("✅ Phase 5 complete: %d document trees saved to %s", 
                    len(document_trees), trees_path)
        sys.stdout.flush()
        
    except Exception as exc:
        LOGGER.exception("Failed to build/save document trees: %s", exc)
        LOGGER.warning("⚠️  Continuing without document trees - hierarchical retrieval will be disabled")
        sys.stdout.flush()

    LOGGER.info("=== Parallel Processing Complete ===")
    LOGGER.info("Total time: %.2f seconds", elapsed_time)
    LOGGER.info("Results: %d successful, %d warnings, %d failed",
               report.successful, report.warnings, report.failed)
    
    return nodes, index, report


def load_cached_nodes_and_index() -> Tuple[Optional[List[Document]], Optional[VectorStoreIndex]]:
    """
    Connect to ChromaDB index. 
    
    Note: With FTS5, we don't load nodes into memory here.
    Nodes are queried directly from SQLite by the FTS5 retriever.
    
    Returns:
        Tuple of (empty nodes list, VectorStoreIndex) or (None, None) if no data
    """
    from .database import get_node_count
    
    config = AppConfig.get()
    paths = config.paths
    
    # Check if we have ANY nodes in SQLite (no tenant filter for existence check)
    node_count = get_node_count(None)
    
    if node_count == 0:
        LOGGER.info("No nodes found in SQLite")
        return None, None
    
    # Check ChromaDB exists
    if not paths.chroma_path.exists():
        LOGGER.info("ChromaDB path doesn't exist")
        return None, None
    
    LOGGER.info("Found %d nodes in SQLite, connecting to ChromaDB", node_count)
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))
    collection = chroma_client.get_or_create_collection("maritime_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    # Don't load nodes into memory - FTS5 queries SQLite directly
    # Return empty list; code needing nodes calls ensure_nodes_loaded()
    LOGGER.info("Connected to ChromaDB index")
    return [], index


@dataclass
class SyncResult:
    added: List[str]
    modified: List[str]
    deleted: List[str]


class IncrementalIndexManager:
    """Handles incremental updates for the document library."""

    def __init__(
        self,
        docs_path: Path,
        gemini_cache_path: Path,
        nodes_cache_path: Path, # DEPRECATED: unused after Phase 3 (SQLite migration)
        cache_info_path: Path,
        chroma_path: Path,
        tenant_id: str = "shared",
    ) -> None:
        self.docs_path = docs_path
        self.gemini_cache_path = gemini_cache_path
        self.nodes_cache_path = nodes_cache_path
        self.cache_info_path = cache_info_path
        self.tenant_id = tenant_id
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_or_create_collection("maritime_docs")
        self.sync_cache_file = cache_info_path.parent / f"sync_cache_{tenant_id}.json"
        self.sync_cache = self._load_sync_cache()
        self.gemini_cache = load_jsonl(gemini_cache_path)
        self.nodes: List[Document] = []

    def _load_sync_cache(self) -> Dict[str, Any]:
        if self.sync_cache_file.exists():
            return json.loads(self.sync_cache_file.read_text())
        return {"files_hash": {}}

    def _save_sync_cache(self) -> None:
        self.sync_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.sync_cache_file.write_text(json.dumps(self.sync_cache, indent=2))

    def _save_nodes_pickle(self, tenant_id: str = "shared") -> None:
        """
        Save current nodes to SQLite.
        
        Note: Method name kept for compatibility, but now saves to SQLite.
        """
        from .nodes import bulk_insert_nodes
        from .database import rebuild_fts_index
        
        if not self.nodes:
            LOGGER.warning("No nodes to save")
            return
        
        # Get existing doc_ids to determine what to update
        repo = NodeRepository(tenant_id=tenant_id)
        
        # Group nodes by doc_id for efficient upsert
        nodes_by_doc = {}
        for node in self.nodes:
            doc_id = node.metadata.get("source", "unknown")
            if doc_id not in nodes_by_doc:
                nodes_by_doc[doc_id] = []
            nodes_by_doc[doc_id].append(node)
        
        # Upsert nodes (this handles both new and existing)
        inserted = bulk_insert_nodes(self.nodes, tenant_id=tenant_id)
        
        # Rebuild FTS index
        rebuild_fts_index()
        
        # Update cache info
        cache_info = {
            "timestamp": datetime.now().isoformat(),
            "num_nodes": len(self.nodes),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "last_sync": datetime.now().isoformat(),
            "storage": "sqlite",
        }
        self.cache_info_path.write_text(json.dumps(cache_info, indent=2))
        
        LOGGER.info("Saved %d nodes to SQLite", inserted)

    def _get_files_hash(self, docs_path: Path) -> Dict[str, Dict[str, Any]]:
        return current_files_index(docs_path)

    def _remove_documents(self, filenames: Set[str], tenant_id: str = None) -> None:
        tenant_id = tenant_id or self.tenant_id
        """
        Remove documents from ChromaDB, SQLite, and Gemini cache.
        
        Args:
            filenames: Set of source filenames to remove
            tenant_id: Tenant identifier
        """
        if not filenames:
            return
        
        LOGGER.info("Removing %d documents: %s", len(filenames), filenames)
        
        # 1. Remove from ChromaDB
        for filename in filenames:
            try:
                # Get chunk IDs for this document
                results = self.collection.get(
                    where={"source": filename},
                    include=[]
                )
                if results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    LOGGER.debug("Removed %d chunks from ChromaDB for %s", 
                            len(results['ids']), filename)
            except Exception as exc:
                LOGGER.warning("Failed to remove %s from ChromaDB: %s", filename, exc)
        
        # 2. Remove from SQLite
        repo = NodeRepository(tenant_id=tenant_id)
        for filename in filenames:
            deleted = repo.delete_by_doc(filename)
            LOGGER.debug("Removed %d nodes from SQLite for %s", deleted, filename)
        
        # 3. Remove from Gemini cache (in-memory AND on disk)
        cache_modified = False
        for filename in filenames:
            if filename in self.gemini_cache:
                del self.gemini_cache[filename]
                cache_modified = True
                LOGGER.info("Removed %s from Gemini cache", filename)
        
        if cache_modified:
            write_jsonl(self.gemini_cache_path, self.gemini_cache.values())
            LOGGER.info("Persisted Gemini cache after removing %d entries", len(filenames))
        
        # 4. Update in-memory nodes list (for compatibility during transition)
        self.nodes = [
            node for node in self.nodes 
            if node.metadata.get("source") not in filenames
        ]
        
        LOGGER.info("Removed documents: %s", filenames)

    def _add_or_update_documents(self, filenames: Set[str], index: Optional[VectorStoreIndex]) -> None:
        """Sequential version - kept for compatibility."""
        for filename in filenames:
            doc_path = self.docs_path / filename
            try:
                if filename in self.gemini_cache:
                    cached_record = self.gemini_cache[filename]
                    gemini_meta = cached_record.get("gemini", {})
                    if "parse_error" not in gemini_meta and gemini_meta.get("sections"):
                        LOGGER.info("%s (cached extraction)", filename)
                        meta = gemini_meta
                    else:
                        LOGGER.info("%s (cache invalid, re-extracting)", filename)
                        meta = gemini_extract_record(doc_path)
                else:
                    LOGGER.info("%s (new file, extracting)", filename)
                    meta = gemini_extract_record(doc_path)

                if "parse_error" in meta:
                    LOGGER.warning("Skipping %s due to extraction error: %s", filename, meta.get("parse_error"))
                    continue

                docs = to_documents_from_gemini(doc_path, meta)
                new_nodes = chunk_documents(docs)
                if new_nodes:
                    if index is not None:
                        index.insert_nodes(new_nodes)
                        LOGGER.info("Added %s chunks from %s", len(new_nodes), filename)
                    
                    # Ensure embeddings are attached to nodes for caching
                    # (insert_nodes generates them but may not attach to objects)
                    from llama_index.core import Settings as LlamaSettings
                    embed_model = LlamaSettings.embed_model
                    for node in new_nodes:
                        node.metadata["tenant_id"] = self.tenant_id
                        if not getattr(node, 'embedding', None):
                            node.embedding = embed_model.get_text_embedding(node.get_content())
                    
                    self.nodes.extend(new_nodes)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.error("Error processing %s: %s", filename, exc)
        if filenames:
            self._save_nodes_pickle()

    def sync_library(
        self, 
        index: Optional[VectorStoreIndex], 
        force_retry_errors: bool = True,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    ) -> Tuple[SyncResult, Optional[ProcessingReport]]:
        """
        Sync library with parallel extraction and embedding generation.
        
        NEW: Now supports progress callbacks and returns processing report!
        
        Args:
            index: VectorStoreIndex to update
            force_retry_errors: If True, retry files with parse_error even if unchanged
            progress_callback: Optional callback(phase, current, total, item_desc)
        
        Returns:
            Tuple of (SyncResult, ProcessingReport)
        """
        start_time = time.time()
        
        # Reload cache
        self.gemini_cache = load_jsonl(self.gemini_cache_path)
        LOGGER.info("Reloaded Gemini cache from disk: %d entries", len(self.gemini_cache))

        # Detect changes
        current_files = self._get_files_hash(self.docs_path)
        cached_files = self.sync_cache.get("files_hash", {})

        new_files = set(current_files) - set(cached_files)
        deleted_files = set(cached_files) - set(current_files)
        modified_files = {fname for fname in (set(current_files) & set(cached_files)) 
                        if current_files[fname] != cached_files[fname]}
        
        # Include files with parse_error for retry
        if force_retry_errors:
            for filename in current_files:
                if filename in self.gemini_cache:
                    gemini_meta = self.gemini_cache[filename].get("gemini", {})
                    if "parse_error" in gemini_meta:
                        modified_files.add(filename)
                        LOGGER.info("Will retry %s (has parse_error in cache)", filename)

        # Create processing report
        total_files_to_process = len(new_files) + len(modified_files)
        report = ProcessingReport(
            timestamp=datetime.now().isoformat(),
            total_files=total_files_to_process,
        )
        
        # Initialize file statuses for files being processed
        file_statuses: Dict[str, DocumentProcessingStatus] = {}
        for filename in (new_files | modified_files):
            file_statuses[filename] = DocumentProcessingStatus(
                filename=filename,
                file_size_bytes=current_files[filename]["size"],
            )

        # Process deletions
        if deleted_files:
            self._remove_documents(deleted_files)
        
        # Process modifications (remove old versions first)
        if modified_files:
            self._remove_documents(modified_files)
        
        # Process additions/modifications WITH progress callback
        successfully_added_or_modified: Set[str] = set()
        if new_files or modified_files:
            successfully_added_or_modified, processing_statuses = self.(
                new_files | modified_files, 
                index,
                progress_callback  # NEW: Pass callback through!
            )
            
            # Merge processing statuses into report
            file_statuses.update(processing_statuses)
        
        # Update sync cache
        self.sync_cache["files_hash"] = current_files
        self._save_sync_cache()

        # Separate successful additions from modifications
        successful_additions = successfully_added_or_modified & new_files
        successful_modifications = successfully_added_or_modified & modified_files
        
        # Finalize report
        elapsed_time = time.time() - start_time
        report.total_duration_sec = elapsed_time
        
        # Add all file statuses to report
        for status in file_statuses.values():
            report.add_status(status)
        
        # Save report (same pattern as parallel build)
        from .config import AppConfig
        config = AppConfig.get()
        report_path = config.paths.cache_dir / "last_sync_report.json"
        from .processing_status import save_processing_report
        save_processing_report(report, report_path)
        LOGGER.info("Sync processing report saved to %s", report_path)
        
        LOGGER.info("=== Sync Complete ===")
        LOGGER.info("Total time: %.2f seconds", elapsed_time)
        LOGGER.info("Results: %d successful, %d warnings, %d failed",
                report.successful, report.warnings, report.failed)
        
        sync_result = SyncResult(
            list(successful_additions),
            list(successful_modifications),
            list(deleted_files)
        )
        
        return sync_result, report

    def _add_or_update_documents_parallel(
        self, 
        filenames: Set[str], 
        index: Optional[VectorStoreIndex],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    ) -> Tuple[Set[str], Dict[str, DocumentProcessingStatus]]:
        """
        Parallel version with parallel extraction and embedding generation.
        
        Args:
            filenames: Files to process
            index: VectorStoreIndex to update
            progress_callback: Optional callback(phase, current, total, item_desc)
        
        Returns:
            Tuple of (successfully_processed_filenames, file_statuses)
        """        
        config = AppConfig.get()
        paths = config.paths
        
        successfully_processed: Set[str] = set()
        all_documents: List[Document] = []
        
        # Track processing status for each file
        file_statuses: Dict[str, DocumentProcessingStatus] = {}
        for filename in filenames:
            file_statuses[filename] = DocumentProcessingStatus(
                filename=filename,
                file_size_bytes=0,
            )
        
        # Phase 1: Check cache and determine what needs extraction
        docs_to_extract: List[Path] = []
        
        for filename in filenames:
            doc_path = self.docs_path / filename
            status = file_statuses[filename]
            
            # Mark parsing as success (file exists)
            status.parsing = StageResult(StageStatus.SUCCESS, "File parsed successfully")
            
            # Check cache
            if filename in self.gemini_cache:
                cached_record = self.gemini_cache[filename]
                gemini_meta = cached_record.get("gemini", {})
                
                if "parse_error" not in gemini_meta and gemini_meta.get("sections"):
                    # Use cached extraction
                    LOGGER.info("%s (using cached extraction)", filename)
                    status.extraction = StageResult(StageStatus.SUCCESS, "Extraction succeeded (cached)")
                    status.used_cache = True
                    
                    # FIX: Set validation status from cached metadata
                    if "validation_error" in gemini_meta:
                        validation_data = gemini_meta.get("validation", {})
                        coverage = validation_data.get("ngram_coverage", 0)
                        status.validation = StageResult(
                            StageStatus.WARNING,
                            f"Low quality extraction (coverage: {coverage:.1%})",
                            details=validation_data
                        )
                    else:
                        validation_data = gemini_meta.get("validation", {})
                        if validation_data:
                            coverage = validation_data.get("ngram_coverage", 0)
                            status.validation = StageResult(
                                StageStatus.SUCCESS,
                                f"Validation passed (coverage: {coverage:.1%})",
                                details=validation_data
                            )
                        else:
                            status.validation = StageResult(StageStatus.SUCCESS, "Validation passed")
                    
                    try:
                        docs = to_documents_from_gemini(doc_path, gemini_meta)
                        all_documents.extend(docs)
                        successfully_processed.add(filename)
                    except Exception as exc:
                        LOGGER.exception("Failed to convert cached extraction for %s", filename)
                        status.extraction = StageResult(StageStatus.FAILED, f"Conversion error: {exc}")
                    continue
            
            # Need to extract
            docs_to_extract.append(doc_path)
        
        # Extract new/modified files in parallel
        if docs_to_extract:
            LOGGER.info("Extracting %d files in parallel...", len(docs_to_extract))
            sys.stdout.flush()
            
            processor = ParallelDocumentProcessor(max_workers=10)
            extraction_results = processor.extract_batch(docs_to_extract, progress_callback)
            
            # Update cache and process results
            for result in extraction_results.successful + extraction_results.failed:
                filename = result.filename
                status = file_statuses[filename]
                
                # Update Gemini cache
                files_hash = self._get_files_hash(self.docs_path)
                fingerprint = files_hash.get(filename, {})
                
                upsert_jsonl_record(
                    self.gemini_cache_path,
                    {
                        "filename": filename,
                        "mtime": fingerprint.get("mtime", 0),
                        "size": fingerprint.get("size", 0),
                        "gemini": result.record,
                        "tenant_id": self.tenant_id,
                    },
                )
            
            # Reload cache after updates
            self.gemini_cache = load_jsonl(self.gemini_cache_path)
            
            # Convert successful extractions to documents
            for result in extraction_results.successful:
                filename = result.filename
                status = file_statuses[filename]
                doc_path = self.docs_path / filename
                
                try:
                    status.extraction = StageResult(StageStatus.SUCCESS, "Extraction succeeded")
                    
                    # FIX: Set validation status from extraction result
                    gemini_meta = result.record
                    if "validation_error" in gemini_meta:
                        validation_data = gemini_meta.get("validation", {})
                        coverage = validation_data.get("ngram_coverage", 0)
                        status.validation = StageResult(
                            StageStatus.WARNING,
                            f"Low quality extraction (coverage: {coverage:.1%})",
                            details=validation_data
                        )
                    else:
                        validation_data = gemini_meta.get("validation", {})
                        if validation_data:
                            coverage = validation_data.get("ngram_coverage", 0)
                            status.validation = StageResult(
                                StageStatus.SUCCESS,
                                f"Validation passed (coverage: {coverage:.1%})",
                                details=validation_data
                            )
                        else:
                            status.validation = StageResult(StageStatus.SUCCESS, "Validation passed")
                    
                    docs = to_documents_from_gemini(doc_path, result.record)
                    all_documents.extend(docs)
                    successfully_processed.add(filename)
                except Exception as exc:
                    LOGGER.exception("Failed to convert %s to documents", filename)
                    status.extraction = StageResult(StageStatus.FAILED, f"Conversion error: {exc}")
            
            # Mark failures
            for result in extraction_results.failed:
                filename = result.filename
                status = file_statuses[filename]
                status.extraction = StageResult(StageStatus.FAILED, f"Extraction error: {result.error}")
                status.validation = StageResult(StageStatus.FAILED, "Extraction failed")
            
            sys.stdout.flush()
        
        if not all_documents:
            LOGGER.info("No documents to add")
            return successfully_processed, file_statuses
        
        # Phase 2: Chunking
        LOGGER.info("Chunking %d documents...", len(all_documents))
        sys.stdout.flush()
        
        new_nodes = chunk_documents(all_documents)
        LOGGER.info("Created %d chunks", len(new_nodes))
        
        # Track chunks per file
        chunks_per_file: Dict[str, int] = {}
        for node in new_nodes:
            node.metadata["tenant_id"] = self.tenant_id
            source = node.metadata.get("source", "")
            if source:
                filename = Path(source).name
                chunks_per_file[filename] = chunks_per_file.get(filename, 0) + 1
        
        # Update chunk counts
        for filename, count in chunks_per_file.items():
            if filename in file_statuses:
                file_statuses[filename].chunks_created = count
        
        # Phase 3: Parallel embedding generation
        LOGGER.info("Generating embeddings in parallel...")
        sys.stdout.flush()
        
        chunk_texts = [node.get_content() for node in new_nodes]
        
        embedding_gen = ParallelEmbeddingGenerator(max_workers=5)
        embedding_batch = embedding_gen.generate_batch(chunk_texts, progress_callback)
        
        LOGGER.info("Generated %d embeddings in %.2fs", 
                len(embedding_batch.embeddings), embedding_batch.duration_sec)
        sys.stdout.flush()
        
        # Attach embeddings to nodes
        for node, embedding in zip(new_nodes, embedding_batch.embeddings):
            node.embedding = embedding

        # Mark embedding success
        for filename in chunks_per_file.keys():
            if filename in file_statuses:
                file_statuses[filename].embedding = StageResult(
                    StageStatus.SUCCESS,
                    f"Embedded {chunks_per_file[filename]} chunks"
                )

        # Phase 4: Add to index sequentially
        if index is not None:
            LOGGER.info("Adding %d chunks to index...", len(new_nodes))
            sys.stdout.flush()
            
            for node in new_nodes:
                index.insert_nodes([node])
            
            LOGGER.info("✓ Added all chunks to index")
        
        # Update in-memory nodes list
        self.nodes.extend(new_nodes)
        
        # Save nodes pickle
        self._save_nodes_pickle()


        LOGGER.info("✓ Sync complete for %d files", len(successfully_processed))
        sys.stdout.flush()
        
        return successfully_processed, file_statuses


__all__ = [
    "chunk_documents",
    "cache_nodes",
    "build_index_from_library",
    "build_index_from_library_parallel",
    "load_cached_nodes_and_index",
    "IncrementalIndexManager",
    "SyncResult",
]
