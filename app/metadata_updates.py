"""Metadata correction system for manual override of AI-extracted fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import AppConfig
from .files import load_jsonl, write_jsonl
from .logger import LOGGER

def _find_tenant_cache_for_file(filename: str) -> tuple[Path, str, dict]:
    """Find which per-tenant JSONL contains a given file's record.
    
    Searches all gemini_extract_cache_{tenant}.jsonl files.
    
    Args:
        filename: Source filename to look up
        
    Returns:
        Tuple of (cache_path, tenant_id, records_dict)
        
    Raises:
        FileNotFoundError: If file not found in any tenant cache
    """
    config = AppConfig.get()
    cache_dir = config.paths.cache_dir
    
    for jsonl_path in cache_dir.glob("gemini_extract_cache_*.jsonl"):
        records = load_jsonl(jsonl_path)
        if filename in records:
            # Extract tenant_id from filename pattern: gemini_extract_cache_{tenant}.jsonl
            tenant_id = jsonl_path.stem.replace("gemini_extract_cache_", "")
            return jsonl_path, tenant_id, records
    
    # Fallback: check legacy single cache (pre-migration)
    legacy = config.paths.gemini_json_cache
    if legacy.exists():
        records = load_jsonl(legacy)
        if filename in records:
            tenant_id = records[filename].get("tenant_id", "shared")
            return legacy, tenant_id, records
    
    raise FileNotFoundError(f"File '{filename}' not found in any Gemini cache")


def apply_corrections_to_gemini_record(
    gemini_record: Dict[str, Any],
    corrections: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply human corrections on top of Gemini extraction.
    
    Args:
        gemini_record: Original extraction from Gemini
        corrections: Dictionary of corrected fields (doc_type, title, form_number, etc.)
    
    Returns:
        Updated record with corrections applied
    """
    if not corrections:
        return gemini_record
    
    record = gemini_record.copy()
    
    # Apply each correction
    if "doc_type" in corrections:
        record["doc_type"] = corrections["doc_type"]
    
    if "title" in corrections:
        record["title"] = corrections["title"]
    
    if "form_number" in corrections:
        record["form_number"] = corrections["form_number"]
    
    if "form_category_name" in corrections or "category" in corrections:
        # Handle both field names for compatibility
        category = corrections.get("form_category_name") or corrections.get("category")
        record["category"] = category
        record["form_category_name"] = category
    
    return record


def save_correction(filename: str, field: str, value: Any) -> bool:
    """
    Save a metadata correction to the gemini_json_cache.
    
    Args:
        filename: Source filename (e.g., "Ballast_Water.docx")
        field: Field to correct ("doc_type", "title", "form_number", "form_category_name")
        value: New value for the field
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_path, tenant_id, records = _find_tenant_cache_for_file(filename)
    except FileNotFoundError:
        LOGGER.error("File not found in any tenant cache: %s", filename)
        return False
    
    try:
        # Load existing cache
        records = load_jsonl(cache_path)
        
        # Find the record for this file
        if filename not in records:
            LOGGER.error("File not found in cache: %s", filename)
            return False
        
        record = records[filename]
        
        # Initialize corrections dict if not present
        if "corrections" not in record:
            record["corrections"] = {}
        
        # Save the correction
        record["corrections"][field] = value
        
        LOGGER.info("✏️  Corrected %s.%s = %s", filename, field, value)
        
        # Write back to cache
        ordered_records = [records[key] for key in sorted(records.keys())]
        write_jsonl(cache_path, ordered_records)
        
        return True
        
    except Exception as exc:
        LOGGER.error("Failed to save correction: %s", exc)
        return False


def save_tenant_id(filename: str, tenant_id: str) -> bool:
    """
    Save tenant_id to the gemini_json_cache at record top level.
    
    Unlike corrections (which override Gemini fields), tenant_id is a 
    file assignment property stored at the record's top level.
    
    Args:
        filename: Source filename
        tenant_id: Tenant identifier
    
    Returns:
        True if successful
    """
    try:
        cache_path, current_tenant, records = _find_tenant_cache_for_file(filename)
    except FileNotFoundError:
        LOGGER.error("File not found in any tenant cache: %s", filename)
        return False
    
    try:
        records = load_jsonl(cache_path)
        
        if filename not in records:
            LOGGER.error("File not found in cache: %s", filename)
            return False
        
        record = records[filename]
        record["tenant_id"] = tenant_id
        
        LOGGER.info("✏️  Set tenant_id for %s = %s", filename, tenant_id)
        
        ordered_records = [records[key] for key in sorted(records.keys())]
        write_jsonl(cache_path, ordered_records)
        
        return True
        
    except Exception as exc:
        LOGGER.error("Failed to save tenant_id: %s", exc)
        return False

def update_metadata_everywhere(
    filename: str,
    corrections: Dict[str, Any],
    nodes: List[Document],
    chroma_path: Path
) -> bool:
    """
    Apply metadata corrections to all three storage locations:
    1. Gemini JSON cache
    2. Pickled nodes cache
    3. ChromaDB vector store
    
    Args:
        filename: Source filename
        corrections: Dictionary of fields to correct
        nodes: List of Document nodes (will be modified in place)
        chroma_path: Path to ChromaDB storage
    
    Returns:
        True if all updates succeeded
    """
    try:
        # 1. Update gemini_json_cache
        for field, value in corrections.items():
            if not save_correction(filename, field, value):
                LOGGER.warning("Failed to save correction to cache: %s.%s", filename, field)
                return False
        
        # 2. Update nodes in memory (caller should re-pickle)
        updated_count = 0
        for node in nodes:
            metadata = node.metadata
            if metadata.get("source") == filename:
                for field, value in corrections.items():
                    # Handle special case for form_category_name
                    if field == "form_category_name":
                        metadata["form_category_name"] = value
                        # Also update category field if it exists
                        if "category" in metadata:
                            metadata["category"] = value
                    else:
                        metadata[field] = value
                updated_count += 1
        
        LOGGER.info("Updated %d nodes in memory", updated_count)
        
        # 3. Update ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        collection = chroma_client.get_or_create_collection("maritime_docs")
        
        # Get all document IDs for this source
        results = collection.get(where={"source": filename})
        doc_ids = results["ids"]
        
        if not doc_ids:
            LOGGER.warning("No documents found in ChromaDB for: %s", filename)
            return False
        
        # Update metadata for each document
        for doc_id in doc_ids:
            update_dict = {}
            for field, value in corrections.items():
                # ChromaDB expects flat metadata
                if field == "form_category_name":
                    update_dict["form_category_name"] = value
                else:
                    update_dict[field] = value
            
            collection.update(
                ids=[doc_id],
                metadatas=[update_dict]
            )
        
        LOGGER.info("✅ Updated %d chunks in ChromaDB", len(doc_ids))
        return True
        
    except Exception as exc:
        LOGGER.error("Failed to update metadata: %s", exc)
        return False


def get_correction_status(filename: str) -> Optional[Dict[str, Any]]:
    """
    Check if a file has any corrections applied.
    
    Args:
        filename: Source filename
    
    Returns:
        Dictionary of corrections if any exist, None otherwise
    """
    try:
        cache_path, tenant_id, records = _find_tenant_cache_for_file(filename)
        record = records.get(filename)
    except FileNotFoundError:
        return None
    except Exception as exc:
        LOGGER.error("Failed to read corrections: %s", exc)
        return None

    if record and "corrections" in record:
        return record["corrections"]
    
    return None

def transfer_document_ownership(
    filename: str,
    new_tenant_id: str,
) -> bool:
    """Transfer a document's physical file and JSONL record to a new tenant.
    
    Performs three operations:
    1. Moves the file from old tenant folder to new tenant folder
    2. Removes the JSONL record from old tenant's cache
    3. Adds the JSONL record to new tenant's cache (with updated tenant_id)
    
    Does NOT update ChromaDB or SQLite — the caller handles that separately
    via update_metadata_everywhere() and direct SQL updates (existing flow).
    
    Args:
        filename: Source filename (e.g. 'ISM_Code.pdf')
        new_tenant_id: Target tenant identifier
        
    Returns:
        True if all file operations succeeded
    """
    import shutil
    
    config = AppConfig.get()
    
    try:
        # 1. Find current location
        old_cache_path, old_tenant_id, old_records = _find_tenant_cache_for_file(filename)
    except FileNotFoundError:
        LOGGER.error("Cannot transfer %s — not found in any tenant cache", filename)
        return False
    
    if old_tenant_id == new_tenant_id:
        LOGGER.info("File %s already belongs to tenant '%s', skipping transfer", filename, new_tenant_id)
        return True
    
    try:
        # 2. Move physical file
        old_file_path = config.docs_path_for(old_tenant_id) / filename
        new_file_path = config.docs_path_for(new_tenant_id) / filename  # docs_path_for auto-creates dir
        
        if old_file_path.exists():
            shutil.move(str(old_file_path), str(new_file_path))
            LOGGER.info("Moved file: %s → %s/", filename, new_tenant_id)
        else:
            # File might already be in new location (partial previous transfer)
            if new_file_path.exists():
                LOGGER.warning("File already at destination: %s/%s", new_tenant_id, filename)
            else:
                LOGGER.warning("Source file not found: %s/%s", old_tenant_id, filename)
        
        # 3. Move JSONL record: remove from old cache, add to new cache
        record = old_records.pop(filename)
        record["tenant_id"] = new_tenant_id

        # 4. Update sync caches so next sync doesn't re-index this file
        from .files import file_fingerprint
        
        # Add to new tenant's sync cache
        new_sync_file = config.paths.cache_dir / f"sync_cache_{new_tenant_id}.json"
        new_sync = json.loads(new_sync_file.read_text()) if new_sync_file.exists() else {"files_hash": {}}
        if new_file_path.exists():
            new_sync["files_hash"][filename] = file_fingerprint(new_file_path)
        new_sync_file.write_text(json.dumps(new_sync, indent=2))
        
        # Remove from old tenant's sync cache
        old_sync_file = config.paths.cache_dir / f"sync_cache_{old_tenant_id}.json"
        if old_sync_file.exists():
            old_sync = json.loads(old_sync_file.read_text())
            old_sync.get("files_hash", {}).pop(filename, None)
            old_sync_file.write_text(json.dumps(old_sync, indent=2))
        
        LOGGER.info("Updated sync caches for ownership transfer: %s → %s", old_tenant_id, new_tenant_id)

        # Write old cache without this record
        ordered_old = [old_records[key] for key in sorted(old_records.keys())]
        write_jsonl(old_cache_path, ordered_old)
        LOGGER.info("Removed %s from %s", filename, old_cache_path.name)
        
        # Add to new tenant's cache
        new_cache_path = config.gemini_cache_for(new_tenant_id)
        new_records = load_jsonl(new_cache_path) if new_cache_path.exists() else {}
        new_records[filename] = record
        ordered_new = [new_records[key] for key in sorted(new_records.keys())]
        write_jsonl(new_cache_path, ordered_new)
        LOGGER.info("Added %s to %s", filename, new_cache_path.name)
        
        LOGGER.info("✅ Transferred %s: %s → %s", filename, old_tenant_id, new_tenant_id)
        return True
        
    except Exception as exc:
        LOGGER.exception("Failed to transfer ownership for %s", filename)
        return False
