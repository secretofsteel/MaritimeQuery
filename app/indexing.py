"""Index creation, chunking, and incremental updates."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import AppConfig
from .constants import CHUNK_OVERLAP, CHUNK_SIZE
from .extraction import gemini_extract_record, to_documents_from_gemini
from .files import current_files_index, load_jsonl, upsert_jsonl_record
from .logger import LOGGER
from .parallel_processing import ParallelDocumentProcessor, ParallelEmbeddingGenerator

def chunk_documents(documents: Iterable[Document]) -> List[Document]:
    """Split LlamaIndex documents into sentence-aware chunks with metadata headers."""
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

        for chunk_text in parser.split_text(document.text):
            full_text = f"{header}\n---\n{chunk_text}"
            chunk_metadata = {**metadata, "chunk_text_len": len(chunk_text)}
            nodes.append(Document(text=full_text, metadata=chunk_metadata))
    return nodes


def cache_nodes(nodes: List[Document], documents_count: int, cache_path: Path, info_path: Path) -> None:
    """Persist chunked nodes and audit metadata."""
    cache_info = {
        "timestamp": datetime.now().isoformat(),
        "num_nodes": len(nodes),
        "num_documents": documents_count,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with cache_path.open("wb") as file:
        pickle.dump(nodes, file)
    with info_path.open("w") as file:
        json.dump(cache_info, file, indent=2)
    LOGGER.info("Cached %s nodes", len(nodes))


def build_index_from_library() -> Tuple[List[Document], VectorStoreIndex]:
    """Process the full document library and build the vector index."""
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

    cache_nodes(nodes, len(documents), paths.nodes_cache_path, paths.cache_info_path)

    chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))
    collection = chroma_client.get_or_create_collection("maritime_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    LOGGER.info("Embedding %s chunks...", len(nodes))
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
    LOGGER.info("Embeddings created and stored.")

    manager = IncrementalIndexManager(paths.docs_path, paths.gemini_json_cache, paths.nodes_cache_path, paths.cache_info_path, paths.chroma_path)
    manager.sync_cache["files_hash"] = manager._get_files_hash(paths.docs_path)
    manager._save_sync_cache()
    LOGGER.info("Initial sync cache saved.")
    return nodes, index


def build_index_from_library_parallel(
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    clear_gemini_cache: bool = False
) -> Tuple[List[Document], VectorStoreIndex]:
    """Process the full document library with parallel extraction and embedding generation.
    
    This parallel version:
    1. WIPES existing ChromaDB collection, pickle cache, and sync cache (full rebuild)
    2. Optionally clears Gemini extraction cache (if clear_gemini_cache=True)
    3. Identifies uncached files that need Gemini extraction
    4. Extracts those files in parallel (10 workers)
    5. Generates embeddings in parallel (10 workers)  
    6. Writes to ChromaDB sequentially (required - not thread-safe)
    
    Args:
        progress_callback: Optional callback(phase, current, total, item_desc)
        clear_gemini_cache: If True, re-extract all files via Gemini (ignores cache)
    
    Returns:
        Tuple of (nodes, index)
    """
    config = AppConfig.get()
    paths = config.paths

    LOGGER.info("=== Starting Parallel Processing (FULL REBUILD) ===")
    
    # PHASE -1: WIPE EXISTING DATA
    LOGGER.info("Phase -1: Clearing existing data...")
    
    # 1. Delete ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))
    try:
        chroma_client.delete_collection("maritime_docs")
        LOGGER.info("✓ Deleted existing ChromaDB collection")
    except Exception:
        LOGGER.info("✓ No existing ChromaDB collection to delete")
    
    # 2. Clear pickle cache
    if paths.nodes_cache_path.exists():
        paths.nodes_cache_path.unlink()
        LOGGER.info("✓ Deleted nodes pickle cache")
    
    # 3. Clear sync cache
    sync_cache_file = paths.cache_info_path.parent / "sync_cache.json"
    if sync_cache_file.exists():
        sync_cache_file.unlink()
        LOGGER.info("✓ Deleted sync cache")
    
    # 4. Optionally clear Gemini extraction cache
    if clear_gemini_cache:
        if paths.gemini_json_cache.exists():
            paths.gemini_json_cache.unlink()
            LOGGER.info("✓ Deleted Gemini extraction cache (will re-extract all files)")
        cached_records = {}
    else:
        cached_records = load_jsonl(paths.gemini_json_cache)
        LOGGER.info("✓ Keeping Gemini extraction cache (%d files)", len(cached_records))
    
    # Phase 0: Determine which files need extraction
    LOGGER.info("Phase 0: Analyzing document library...")
    files_index = current_files_index(paths.docs_path)
    
    docs_to_extract: List[Path] = []
    all_documents: List[Document] = []
    
    # Check each file
    for filename, fingerprint in files_index.items():
        doc_path = paths.docs_path / filename
        cached = cached_records.get(filename)
        
        needs_extraction = not (
            cached 
            and abs(cached.get("mtime", 0) - fingerprint["mtime"]) < 1 
            and cached.get("size") == fingerprint["size"]
        )
        
        if not needs_extraction and cached and "gemini" in cached:
            meta = cached["gemini"]
        else:
            # Need to extract
            docs_to_extract.append(doc_path)
            continue
        
        # Use cached extraction
        if "parse_error" in meta:
            LOGGER.error("Skipping %s due to cached extraction error: %s", 
                        filename, meta.get("parse_error"))
            continue
        
        documents_from_cache = to_documents_from_gemini(doc_path, meta)
        all_documents.extend(documents_from_cache)
    
    LOGGER.info("Using %d cached extractions, need to extract %d files", 
               len(files_index) - len(docs_to_extract), len(docs_to_extract))
    
    # Phase 1: Parallel extraction for uncached files
    if docs_to_extract:
        LOGGER.info("Phase 1: Parallel extraction of %d documents...", len(docs_to_extract))
        
        processor = ParallelDocumentProcessor(max_workers=10)
        extraction_results = processor.extract_batch(docs_to_extract, progress_callback)
        
        # Update JSONL cache with results
        for result in extraction_results.successful + extraction_results.failed:
            filename = result.filename
            fingerprint = files_index[filename]
            
            if result.record:
                upsert_jsonl_record(
                    paths.gemini_json_cache,
                    {
                        "filename": filename,
                        "mtime": fingerprint["mtime"],
                        "size": fingerprint["size"],
                        "gemini": result.record,
                    },
                )
        
        # Convert successful extractions to documents
        for result in extraction_results.successful:
            doc_path = paths.docs_path / result.filename
            documents_from_extraction = to_documents_from_gemini(doc_path, result.record)
            all_documents.extend(documents_from_extraction)
        
        # Log failures
        for result in extraction_results.failed:
            LOGGER.error("Skipping %s due to extraction error: %s", 
                        result.filename, result.error)
        
        LOGGER.info("Phase 1 complete: %d successful, %d failed",
                   extraction_results.success_count, extraction_results.failure_count)
    else:
        LOGGER.info("Phase 1 skipped: All documents cached")
    
    LOGGER.info("Total documents loaded: %d", len(all_documents))
    
    # Phase 2: Chunking
    LOGGER.info("Phase 2: Chunking documents...")
    nodes = chunk_documents(all_documents)
    LOGGER.info("Created %d chunks", len(nodes))
    
    cache_nodes(nodes, len(all_documents), paths.nodes_cache_path, paths.cache_info_path)
    
    # Phase 3: Parallel embedding generation + sequential ChromaDB writes
    LOGGER.info("Phase 3: Parallel embedding generation...")
    
    # Extract texts for embedding
    chunk_texts = [node.get_content() for node in nodes]
    
    # Generate embeddings in parallel
    embedding_gen = ParallelEmbeddingGenerator(max_workers=10)
    embedding_batch = embedding_gen.generate_batch(chunk_texts, progress_callback)
    
    LOGGER.info("Generated %d embeddings in %.2fs", 
               len(embedding_batch.embeddings), embedding_batch.duration_sec)
    
    # Attach embeddings to nodes
    for node, embedding in zip(nodes, embedding_batch.embeddings):
        node.embedding = embedding
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))
    collection = chroma_client.get_or_create_collection("maritime_docs")
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
    
    LOGGER.info("=== Parallel Processing Complete ===")
    
    return nodes, index

def load_cached_nodes_and_index() -> Tuple[Optional[List[Document]], Optional[VectorStoreIndex]]:
    """Load cached nodes and rehydrate the index if possible."""
    config = AppConfig.get()
    paths = config.paths
    if paths.nodes_cache_path.exists() and paths.chroma_path.exists():
        LOGGER.info("Detected cached nodes and ChromaDB.")
        with paths.nodes_cache_path.open("rb") as file:
            cached_nodes: List[Document] = pickle.load(file)
        chroma_client = chromadb.PersistentClient(path=str(paths.chroma_path))
        collection = chroma_client.get_or_create_collection("maritime_docs")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        LOGGER.info("Loaded %s cached nodes and existing Chroma collection.", len(cached_nodes))
        return cached_nodes, index
    LOGGER.info("No existing cache found.")
    return None, None


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
        nodes_cache_path: Path,
        cache_info_path: Path,
        chroma_path: Path,
    ) -> None:
        self.docs_path = docs_path
        self.gemini_cache_path = gemini_cache_path
        self.nodes_cache_path = nodes_cache_path
        self.cache_info_path = cache_info_path
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_or_create_collection("maritime_docs")
        self.sync_cache_file = cache_info_path.parent / "sync_cache.json"
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

    def _save_nodes_pickle(self) -> None:
        if self.nodes_cache_path:
            with self.nodes_cache_path.open("wb") as file:
                pickle.dump(self.nodes, file)
        cache_info = {
            "timestamp": datetime.now().isoformat(),
            "num_nodes": len(self.nodes),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "last_sync": datetime.now().isoformat(),
        }
        self.cache_info_path.write_text(json.dumps(cache_info, indent=2))

    def _get_files_hash(self, docs_path: Path) -> Dict[str, Dict[str, Any]]:
        return current_files_index(docs_path)

    def _remove_documents(self, filenames: Set[str]) -> None:
        for filename in filenames:
            results = self.collection.get(where={"source": filename})
            if results.get("ids"):
                LOGGER.info("Removing %s chunks for %s", len(results["ids"]), filename)
                self.collection.delete(ids=results["ids"])
                self.nodes = [node for node in self.nodes if node.metadata.get("source") != filename]
        if filenames:
            self._save_nodes_pickle()

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
                    self.nodes.extend(new_nodes)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.error("Error processing %s: %s", filename, exc)
        if filenames:
            self._save_nodes_pickle()

    def _add_or_update_documents_parallel(self, filenames: Set[str], index: Optional[VectorStoreIndex]) -> None:
        """Parallel version with parallel extraction and embedding generation."""
        if not filenames:
            return
        
        LOGGER.info("Syncing %d files with parallel processing...", len(filenames))
        
        # Separate files that need extraction vs can use cache
        files_needing_extraction: List[Path] = []
        cached_documents: List[Document] = []
        
        for filename in filenames:
            doc_path = self.docs_path / filename
            
            if filename in self.gemini_cache:
                cached_record = self.gemini_cache[filename]
                gemini_meta = cached_record.get("gemini", {})
                
                if "parse_error" not in gemini_meta and gemini_meta.get("sections"):
                    # Use cached extraction
                    LOGGER.info("%s (using cached extraction)", filename)
                    try:
                        docs = to_documents_from_gemini(doc_path, gemini_meta)
                        cached_documents.extend(docs)
                    except Exception as exc:
                        LOGGER.error("Error converting cached %s: %s", filename, exc)
                else:
                    # Cache invalid or has parse_error - re-extract
                    files_needing_extraction.append(doc_path)
            else:
                # Not in cache - extract
                files_needing_extraction.append(doc_path)
        
        # Phase 1: Parallel extraction for files that need it
        all_documents = cached_documents.copy()
        
        if files_needing_extraction:
            LOGGER.info("Extracting %d files in parallel...", len(files_needing_extraction))
            
            processor = ParallelDocumentProcessor(max_workers=10)
            extraction_results = processor.extract_batch(files_needing_extraction)
            
            # Update JSONL cache with results
            files_index = current_files_index(self.docs_path)
            for result in extraction_results.successful + extraction_results.failed:
                filename = result.filename
                if filename in files_index:
                    fingerprint = files_index[filename]
                    
                    if result.record:
                        upsert_jsonl_record(
                            self.gemini_cache_path,
                            {
                                "filename": filename,
                                "mtime": fingerprint["mtime"],
                                "size": fingerprint["size"],
                                "gemini": result.record,
                            },
                        )
            
            # Reload cache after updates
            self.gemini_cache = load_jsonl(self.gemini_cache_path)
            
            # Convert successful extractions to documents
            for result in extraction_results.successful:
                doc_path = self.docs_path / result.filename
                try:
                    docs = to_documents_from_gemini(doc_path, result.record)
                    all_documents.extend(docs)
                except Exception as exc:
                    LOGGER.exception("Failed to convert %s to documents", result.filename)
        
        if not all_documents:
            LOGGER.info("No documents to add")
            return
        
        # Phase 2: Chunk all documents
        LOGGER.info("Chunking %d documents...", len(all_documents))
        new_nodes = chunk_documents(all_documents)
        LOGGER.info("Created %d chunks", len(new_nodes))
        
        # Phase 3: Parallel embedding generation
        LOGGER.info("Generating embeddings in parallel...")
        chunk_texts = [node.get_content() for node in new_nodes]
        
        embedding_gen = ParallelEmbeddingGenerator(max_workers=10)
        embedding_batch = embedding_gen.generate_batch(chunk_texts)
        
        LOGGER.info("Generated %d embeddings in %.2fs", 
                   len(embedding_batch.embeddings), embedding_batch.duration_sec)
        
        # Attach embeddings to nodes
        for node, embedding in zip(new_nodes, embedding_batch.embeddings):
            node.embedding = embedding
        
        # Phase 4: Add to index sequentially (embeddings pre-computed)
        if index is not None:
            LOGGER.info("Adding %d chunks to index...", len(new_nodes))
            for node in new_nodes:
                # Use insert_nodes with embeddings already attached
                index.insert_nodes([node])
            LOGGER.info("✓ Added all chunks to index")
        
        # Update in-memory nodes list
        self.nodes.extend(new_nodes)
        
        # Save nodes pickle
        self._save_nodes_pickle()
        
        LOGGER.info("✓ Sync complete for %d files", len(filenames))

    def sync_library(self, index: Optional[VectorStoreIndex], force_retry_errors: bool = True) -> SyncResult:
        """Sync library with parallel extraction and embedding generation.
        
        Args:
            index: VectorStoreIndex to update
            force_retry_errors: If True, retry files with parse_error even if unchanged
        
        Returns:
            SyncResult with added, modified, and deleted file counts
        """
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

        if deleted_files:
            self._remove_documents(deleted_files)
        if modified_files:
            self._remove_documents(modified_files)
        if new_files or modified_files:
            self._add_or_update_documents_parallel(new_files | modified_files, index)

        self.sync_cache["files_hash"] = current_files
        self._save_sync_cache()

        return SyncResult(list(new_files), list(modified_files), list(deleted_files))


__all__ = [
    "chunk_documents",
    "cache_nodes",
    "build_index_from_library",
    "build_index_from_library_parallel",
    "load_cached_nodes_and_index",
    "IncrementalIndexManager",
    "SyncResult",
]
