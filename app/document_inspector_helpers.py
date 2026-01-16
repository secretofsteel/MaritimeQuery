"""
Helper functions for Document Inspector in admin panel.

Provides utilities to:
- Load cached text extracts from pickle files
- Load document trees from JSON
- Load Gemini extraction data from JSONL cache
- Format trees visually for display
- Identify problem documents
"""

from __future__ import annotations

import pickle
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from app.config import AppConfig
from app.logger import LOGGER


@dataclass
class DocumentMetrics:
    """Metrics for a single document."""
    filename: str
    source_path: str
    text_length: int  # chars
    file_size_bytes: int
    num_sections: int
    num_chunks: int
    validation_coverage: float  # 0-1
    has_extraction: bool
    has_tree: bool
    extraction_status: str  # "✅", "⚠️", or "❌"


def _get_text_cache_path(file_path: Path, cache_dir: Path) -> Path:
    """
    Generate cache path for text extract pickle.
    Matches the logic in files.py.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Original file not found: {file_path}")
    
    file_stat = file_path.stat()
    cache_key = f"{file_stat.st_mtime}_{file_stat.st_size}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    
    base_name = file_path.stem
    safe_name = re.sub(r'[^\w\-]', '_', base_name)
    
    return cache_dir / "text_extracts" / f"{safe_name}_{cache_hash}.pkl"


def load_cached_text(source_filename: str) -> Optional[str]:
    """
    Load cached text extract for a document.
    
    Args:
        source_filename: Original filename (e.g., "Chapter 7.5.pdf")
    
    Returns:
        Extracted text or None if not found
    """
    config = AppConfig.get()
    docs_path = config.paths.docs_path
    cache_dir = config.paths.cache_dir
    
    # Find the file in docs directory
    file_path = docs_path / source_filename
    if not file_path.exists():
        LOGGER.warning("File not found: %s", source_filename)
        return None
    
    try:
        cache_path = _get_text_cache_path(file_path, cache_dir)
        
        if not cache_path.exists():
            LOGGER.warning("No cached text for: %s", source_filename)
            return None
        
        with open(cache_path, 'rb') as f:
            text = pickle.load(f)
        
        return text
    
    except Exception as exc:
        LOGGER.error("Failed to load cached text for %s: %s", source_filename, exc)
        return None


def load_document_tree(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Load document tree structure for a specific document.
    
    Args:
        doc_id: Document ID (typically filename without extension)
    
    Returns:
        Tree structure dict or None if not found
    """
    config = AppConfig.get()
    trees_path = config.paths.cache_dir / "document_trees.json"
    
    if not trees_path.exists():
        LOGGER.warning("Document trees file not found")
        return None
    
    try:
        with open(trees_path, 'r', encoding='utf-8') as f:
            trees = json.load(f)
        
        # Find tree for this doc_id
        for tree in trees:
            if tree.get("doc_id") == doc_id:
                return tree
        
        LOGGER.warning("No tree found for doc_id: %s", doc_id)
        return None
    
    except Exception as exc:
        LOGGER.error("Failed to load document tree for %s: %s", doc_id, exc)
        return None


def load_extraction_data(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load Gemini extraction data for a document.
    
    Args:
        filename: Original filename
    
    Returns:
        Extraction dict (the gemini nested object) or None if not found
    """
    config = AppConfig.get()
    gemini_cache = config.paths.cache_dir / "gemini_extract_cache.jsonl"
    
    if not gemini_cache.exists():
        LOGGER.warning("Gemini cache not found")
        return None
    
    try:
        with open(gemini_cache, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get('filename') == filename:
                    # Return the gemini nested object, not the wrapper
                    gemini_data = data.get('gemini', {})
                    # Also merge corrections if present
                    corrections = data.get('corrections', {})
                    if corrections:
                        gemini_data['corrections'] = corrections
                    # Add top-level validation error if present
                    if data.get('validation_error'):
                        gemini_data['validation_error'] = data['validation_error']
                    return gemini_data
        
        LOGGER.warning("No extraction found for: %s", filename)
        return None
    
    except Exception as exc:
        LOGGER.error("Failed to load extraction for %s: %s", filename, exc)
        return None


def format_tree_visually(tree: Dict[str, Any], max_depth: int = 5) -> str:
    """
    Format document tree as indented text for display.
    
    Args:
        tree: Document tree structure
        max_depth: Maximum recursion depth
    
    Returns:
        Formatted tree string
    """
    lines = []
    
    doc_id = tree.get("doc_id", "Unknown")
    lines.append(f"Document: {doc_id}")
    lines.append("=" * 80)
    lines.append("")
    
    sections = tree.get("sections", [])
    if not sections:
        lines.append("⚠️  No sections found")
        return "\n".join(lines)
    
    def format_section(section: Dict[str, Any], depth: int = 0, prefix: str = ""):
        """Recursively format a section."""
        if depth > max_depth:
            return
        
        section_id = section.get("section_id", "???")
        # Try multiple field names for section title
        section_title = section.get("title") or section.get("section_name") or section.get("original_name") or "Unnamed"
        chunk_ids = section.get("chunk_ids", [])
        
        # Calculate indent
        indent = "  " * depth
        
        # Section line with chunk count
        chunk_info = f"[{len(chunk_ids)} chunks]" if chunk_ids else "[no chunks]"
        lines.append(f"{indent}├─ {section_id}: {section_title} {chunk_info}")
        
        # Recursively format children
        children = section.get("children", [])
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            child_prefix = prefix + ("  " if is_last else "│ ")
            format_section(child, depth + 1, child_prefix)
    
    # Format each top-level section
    for section in sections:
        format_section(section)
        lines.append("")
    
    return "\n".join(lines)


def calculate_section_token_sizes(tree: Dict[str, Any], nodes: List[Any]) -> Dict[str, int]:
    """
    Calculate token count for each section by looking up chunks.
    
    Args:
        tree: Document tree
        nodes: List of all nodes (app_state.nodes)
    
    Returns:
        Dict mapping section_id -> token_count
    """
    # Build node text lookup by node_id
    node_texts = {}
    for node in nodes:
        node_id = node.id_ if hasattr(node, 'id_') else node.node_id
        node_texts[node_id] = node.text if hasattr(node, 'text') else ""
    
    section_tokens = {}
    
    def count_tokens_recursive(section: Dict[str, Any]):
        """Recursively count tokens for section and children."""
        section_id = section.get("section_id")
        if not section_id:
            return
        
        total_tokens = 0
        
        # Count tokens from direct chunks
        chunk_ids = section.get("chunk_ids", [])
        for chunk_id in chunk_ids:
            text = node_texts.get(chunk_id, "")
            total_tokens += len(text.split())
        
        # Recursively count children
        for child in section.get("children", []):
            child_tokens = count_tokens_recursive(child)
            total_tokens += child_tokens
        
        section_tokens[section_id] = total_tokens
        return total_tokens
    
    # Process all sections
    for section in tree.get("sections", []):
        count_tokens_recursive(section)
    
    return section_tokens


def identify_problem_documents(app_state) -> List[Tuple[str, List[str]]]:
    """
    Scan all documents and identify problems.
    
    Returns:
        List of (filename, [issue descriptions])
    """
    config = AppConfig.get()
    problems = []
    
    # Get all unique sources from nodes
    sources = set()
    for node in app_state.nodes:
        source = node.metadata.get("source")
        if source:
            sources.add(source)
    
    for source in sorted(sources):
        issues = []
        doc_id = Path(source).stem
        
        # Check 1: Load tree and check for mega-sections
        tree = load_document_tree(doc_id)
        if tree:
            section_tokens = calculate_section_token_sizes(tree, app_state.nodes)
            mega_sections = {sid: tokens for sid, tokens in section_tokens.items() if tokens > 15000}
            
            if mega_sections:
                for sid, tokens in mega_sections.items():
                    issues.append(f"⚠️ Mega-section {sid}: {tokens:,} tokens")
        else:
            issues.append("❌ No document tree found")
        
        # Check 2: Load extraction and check validation coverage
        extraction = load_extraction_data(source)
        if extraction:
            # Get doc_type to check if it's a form/checklist
            corrections = extraction.get("corrections", {})
            doc_type = corrections.get("doc_type") or extraction.get("doc_type")
            is_form_or_checklist = doc_type and doc_type.upper() in ("FORM", "CHECKLIST")
            
            validation = extraction.get("validation", {})
            # Use ngram_coverage (0-1 scale)
            coverage = validation.get("ngram_coverage", 0)
            
            # Only flag low coverage for non-form documents
            if not is_form_or_checklist and coverage < 0.85:
                issues.append(f"⚠️ Low validation coverage: {coverage * 100:.1f}%")
            
            # Check for validation errors (but skip coverage-related errors for forms/checklists)
            validation_error = extraction.get("validation_error")
            if validation_error:
                # Skip coverage-related validation errors for forms/checklists
                is_coverage_error = "coverage" in validation_error.lower()
                if not (is_form_or_checklist and is_coverage_error):
                    issues.append(f"⚠️ Validation error: {validation_error}")
        else:
            issues.append("❌ No Gemini extraction found")
        
        # Check 3: Check if cached text exists
        cached_text = load_cached_text(source)
        if not cached_text:
            issues.append("❌ No cached text extract")
        
        if issues:
            problems.append((source, issues))
    
    return problems


def get_document_metrics(source: str, app_state) -> Optional[DocumentMetrics]:
    """
    Get comprehensive metrics for a document.
    
    Args:
        source: Source filename
        app_state: AppState instance
    
    Returns:
        DocumentMetrics or None
    """
    config = AppConfig.get()
    doc_id = Path(source).stem
    
    # Get file path and size
    file_path = config.paths.docs_path / source
    if not file_path.exists():
        return None
    
    file_size = file_path.stat().st_size
    
    # Load cached text
    cached_text = load_cached_text(source)
    text_length = len(cached_text) if cached_text else 0
    
    # Load tree
    tree = load_document_tree(doc_id)
    num_sections = len(tree.get("sections", [])) if tree else 0
    has_tree = tree is not None
    
    # Load extraction
    extraction = load_extraction_data(source)
    has_extraction = extraction is not None
    
    # Try to get validation coverage from multiple possible locations
    validation_coverage = 0.0
    if extraction:
        # Validation data is directly in the extraction (since we unwrapped it)
        validation = extraction.get("validation")
        if validation and isinstance(validation, dict):
            # Use ngram_coverage as the primary metric (it's already 0-1 scale)
            ngram_cov = validation.get("ngram_coverage")
            if ngram_cov is not None:
                validation_coverage = float(ngram_cov)
            # Fallback to word_coverage if ngram not available
            elif validation.get("word_coverage") is not None:
                validation_coverage = float(validation.get("word_coverage"))
    
    # Count chunks from app_state
    num_chunks = sum(1 for node in app_state.nodes if node.metadata.get("source") == source)
    
    # Determine extraction status
    # Get doc_type to check if it's a form/checklist
    doc_type = None
    if extraction:
        corrections = extraction.get("corrections", {})
        doc_type = corrections.get("doc_type") or extraction.get("doc_type")
    
    # For forms/checklists, don't penalize low coverage (they have lots of empty fields)
    is_form_or_checklist = doc_type and doc_type.upper() in ("FORM", "CHECKLIST")
    
    if not has_extraction or not has_tree or not cached_text:
        extraction_status = "❌"
    elif not is_form_or_checklist and validation_coverage < 0.85:
        # Only flag low coverage for non-form documents
        extraction_status = "⚠️"
    else:
        extraction_status = "✅"
    
    return DocumentMetrics(
        filename=source,
        source_path=str(file_path),
        text_length=text_length,
        file_size_bytes=file_size,
        num_sections=num_sections,
        num_chunks=num_chunks,
        validation_coverage=validation_coverage,
        has_extraction=has_extraction,
        has_tree=has_tree,
        extraction_status=extraction_status,
    )