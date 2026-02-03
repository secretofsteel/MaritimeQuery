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
    config = AppConfig.get()
    cache_path = config.paths.gemini_json_cache
    
    if not cache_path.exists():
        LOGGER.error("Gemini cache not found: %s", cache_path)
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
    config = AppConfig.get()
    cache_path = config.paths.gemini_json_cache
    
    if not cache_path.exists():
        LOGGER.error("Gemini cache not found: %s", cache_path)
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
    config = AppConfig.get()
    cache_path = config.paths.gemini_json_cache
    
    if not cache_path.exists():
        return None
    
    try:
        records = load_jsonl(cache_path)
        record = records.get(filename)
        
        if record and "corrections" in record:
            return record["corrections"]
        
        return None
        
    except Exception as exc:
        LOGGER.error("Failed to read corrections: %s", exc)
        return None
