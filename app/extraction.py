"""Gemini extraction with boundary detection fallback and integrated validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core import Document
from google.genai import types
from rapidfuzz import fuzz

from .config import AppConfig
from .constants import (
    EXTRACT_SYSTEM_PROMPT,
    FORM_CATEGORIES,
    GEMINI_SCHEMA,
    PYTHON_SLICING_FUZZY_THRESHOLD,
)
from .files import clean_text_for_llm, read_doc_for_llm, is_scanned_pdf
from .logger import LOGGER
from .profiling import time_function


# ============================================================================
# HELPERS
# ============================================================================


def _context_contains_anchor(context: str, anchor: str, min_ratio: int = 75) -> bool:
    """
    Check if a context window likely contains the anchor text.
    We do a simple fuzzy match over a normalized context.
    """
    if not anchor:
        return False

    ctx = _normalize_whitespace(context.lower())
    anc = _normalize_whitespace(anchor.lower())

    # Exact / substring match first
    if anc in ctx:
        return True

    # If anchor is long, take first ~80 chars as probe
    probe = anc[:80]
    score = fuzz.partial_ratio(probe, ctx)
    return score >= min_ratio


def _get_section_anchor_text(section_info: Dict[str, Any]) -> Optional[str]:
    """
    Get a robust anchor string for validating that a slice is real content and not TOC.

    Priority:
    1. first_sentence (if provided by Gemini)
    2. section name without the numeric id prefix
    """
    # 1) Prefer first_sentence if present
    first_sentence = section_info.get("first_sentence") or section_info.get("summary")
    if first_sentence:
        # Use only the first ~200 chars to avoid over-noise
        return first_sentence.strip()[:200]

    # 2) Fallback: title without numeric id prefix
    name = section_info.get("name") or ""
    if not name:
        return None

    # Remove leading "3", "11.3.2", "15.2.4.1.1", "A.", etc.
    m = re.match(r"^\s*(?:[0-9]+(?:\.[0-9]+)*|[A-Z]\.)\s*(.*)$", name)
    title_without_id = m.group(1).strip() if m and m.group(1) else name.strip()

    # Don't anchor on a 1-word title, it's too weak.
    if len(title_without_id.split()) < 2:
        return None

    return title_without_id[:200]



def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace for robust text matching.
    
    Collapses newlines, tabs, and multiple spaces into single spaces.
    This handles cases where section headers are split across lines like:
        "7.2.2.8\nRPM Posting" becomes "7.2.2.8 RPM Posting"
    
    Args:
        text: Text to normalize
    
    Returns:
        Text with normalized whitespace
    """
    return re.sub(r'\s+', ' ', text).strip()

def _index_lines_for_slicing(raw_text: str) -> Tuple[List[str], List[int]]:
    """
    Split raw text into lines and compute starting character offsets for each line.
    Used for efficient heading-based slicing.
    """
    lines = raw_text.splitlines(keepends=True)
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    return lines, offsets


def _find_heading_candidates(lines: List[str]) -> List[int]:
    """
    Heuristically identify lines that look like headings.

    This is intentionally generic:
    - Numbered headings like "1", "1.1", "3.2.1 Title"
    - Lettered like "A. IHO Standards"
    - All-caps blocks like "SAFETY PROCEDURES"
    """
    heading_indices: List[int] = []
    pattern = re.compile(
        r"^\s*("            # start of line
        r"\d+(?:\.\d+)*\.?\s+.+|"   # 1.2 or 1.2.3 Title...
        r"[A-Z]\.\s+.+|"            # A. Something
        r"[A-Z][A-Z0-9\-\s]{4,}"    # LONG UPPERCASE THING
        r")"
    )

    for i, line in enumerate(lines):
        if pattern.match(line):
            heading_indices.append(i)

    return heading_indices


def _find_section_start_by_id(
    raw_text: str,
    section_info: Dict[str, Any],
    next_section_info: Optional[Dict[str, Any]] = None,
    toc_start: int = -1,
    toc_end: int = -1,
) -> int:
    """Locate section, with TOC range to exclude."""
    section_name = section_info.get("name", "") or section_info.get("section_id", "")
    sec_id = _extract_section_id(section_name)
    
    if sec_id:
        result = _find_by_section_id(raw_text, section_info, next_section_info, sec_id, toc_start, toc_end)
        if result >= 0:
            return result
    
    LOGGER.debug("No section ID found for '%s', trying fuzzy name match", section_name)
    return _find_by_fuzzy_name(raw_text, section_info, next_section_info, toc_start, toc_end)

def _find_by_section_id(
    raw_text: str,
    section_info: Dict[str, Any],
    next_section_info: Optional[Dict[str, Any]],
    sec_id: str,
    toc_start: int = -1,      # ADD THIS
    toc_end: int = -1,        # ADD THIS
) -> int:
    """Find section using its numeric/letter ID with TOC filtering."""
    
    # Build anchor set
    anchors: List[str] = []
    cur_anchor = _get_section_anchor_text(section_info)
    if cur_anchor:
        anchors.append(cur_anchor)
    if next_section_info:
        next_anchor = _get_section_anchor_text(next_section_info)
        if next_anchor:
            anchors.append(next_anchor)

    # Look for all lines that start with the ID
    pattern = rf"^\s*{re.escape(sec_id)}[^\n]*$"
    matches = list(re.finditer(pattern, raw_text, flags=re.MULTILINE))
    
    if not matches:
        LOGGER.warning("No text matches found for section ID '%s'", sec_id)
        return -1
    
    LOGGER.debug("Found %d potential matches for section ID '%s'", len(matches), sec_id)
    
    # STRATEGY: Filter TOC first, then validate with anchors
    non_toc_matches = []
    for idx, m in enumerate(matches):
        start = m.start()
        
        # Check if this line is in a TOC context
        if _is_in_toc_range(start, toc_start, toc_end):
            LOGGER.debug("Match at position %d is in TOC range, skipping", start)
            continue
        
        non_toc_matches.append((idx, start))
    
    # If no anchors, return first non-TOC match
    if not anchors:
        if non_toc_matches:
            LOGGER.debug("No anchors available, returning first non-TOC match")
            return non_toc_matches[0][1]
        else:
            LOGGER.warning("No anchors and all matches marked as TOC for '%s'", sec_id)
            return -1
    
    # Try to validate with anchors
    for idx, start in non_toc_matches:
        # Take a context window after this line
        ctx_start = start
        ctx_end = min(len(raw_text), start + 800)
        context = raw_text[ctx_start:ctx_end]

        # Check if at least one anchor appears in this context
        has_anchor = any(_context_contains_anchor(context, anc) for anc in anchors)
        if has_anchor:
            LOGGER.debug("Match %d at position %d validated with anchor", idx, start)
            return start
        else:
            LOGGER.debug("Match %d at position %d failed anchor validation", idx, start)
    
    # All matches failed anchor validation
    # FALLBACK: If we have non-TOC matches but anchors failed, return first non-TOC anyway
    if non_toc_matches:
        LOGGER.warning("All non-TOC matches for '%s' failed anchor validation, using first match anyway", sec_id)
        return non_toc_matches[0][1]
    
    LOGGER.warning("All %d matches for '%s' failed validation", len(matches), sec_id)
    return -1

def _find_by_fuzzy_name(
    raw_text: str,
    section_info: Dict[str, Any],
    next_section_info: Optional[Dict[str, Any]],
    toc_start: int = -1,      # ADD THIS
    toc_end: int = -1,        # ADD THIS
) -> int:
    """
    Find section by fuzzy matching its full name (for unnumbered sections).
    
    Strategy:
    1. Search for the section name in the text
    2. Validate with anchors (if available)
    3. Exclude matches within TOC range
    
    Returns -1 if not found.
    """
    section_name = section_info.get("name", "")
    if not section_name:
        return -1
    
    # Build anchors for validation
    anchors: List[str] = []
    cur_anchor = _get_section_anchor_text(section_info)
    if cur_anchor:
        anchors.append(cur_anchor)
    if next_section_info:
        next_anchor = _get_section_anchor_text(next_section_info)
        if next_anchor:
            anchors.append(next_anchor)
    
    # Normalize for searching
    text_lower = raw_text.lower()
    name_lower = section_name.lower().strip()
    
    # Pattern: Look for the name at start of line (likely a heading)
    pattern = rf"^\s*{re.escape(name_lower)}[^\n]*$"
    matches = list(re.finditer(pattern, text_lower, flags=re.MULTILINE))
    
    if not matches:
        # Fallback: Try line-by-line fuzzy matching
        lines = raw_text.splitlines()
        candidates = []
        pos = 0
        
        for line in lines:
            line_norm = _normalize_whitespace(line.lower())
            name_norm = _normalize_whitespace(name_lower)
            
            # Use fuzzy ratio
            score = fuzz.ratio(name_norm, line_norm)
            if score >= 85:  # 85% similarity threshold
                # NEW: Skip if in TOC range
                if not _is_in_toc_range(pos, toc_start, toc_end):
                    candidates.append(pos)
            
            pos += len(line) + 1  # +1 for newline
        
        if not candidates:
            return -1
        
        # Create simple match objects
        class SimpleMatch:
            def __init__(self, position):
                self._pos = position
            def start(self):
                return self._pos
        
        matches = [SimpleMatch(p) for p in candidates]
    
    if not matches:
        return -1
    
    # Filter out TOC matches first
    non_toc_matches = []
    for m in matches:
        start = m.start()
        
        # NEW: Check if in TOC range
        if _is_in_toc_range(start, toc_start, toc_end):
            LOGGER.debug("Fuzzy match at position %d is in TOC range, skipping", start)
            continue
        
        non_toc_matches.append((start, m))
    
    if not non_toc_matches:
        LOGGER.warning("All fuzzy matches for '%s' were in TOC range", section_name)
        return -1
    
    # If we have anchors, validate candidates
    if anchors:
        for start, m in non_toc_matches:
            # Validate with anchors
            ctx_start = start
            ctx_end = min(len(raw_text), start + 800)
            context = raw_text[ctx_start:ctx_end]
            
            if any(_context_contains_anchor(context, anc) for anc in anchors):
                return start
    
    # No anchors available, or anchor validation failed
    # Return first non-TOC match (best guess)
    return non_toc_matches[0][0]


def _find_toc_boundaries(raw_text: str) -> Tuple[int, int]:
    """
    Find the start and end positions of the Table of Contents.
    
    Returns:
        (start_pos, end_pos) tuple, or (-1, -1) if no TOC found
    """
    text_lower = raw_text.lower()
    
    # Look for TOC header
    toc_patterns = [
        r'table\s+of\s+contents',
        r'\bcontents\b',
        r'\bindex\b'
    ]
    
    toc_start = -1
    for pattern in toc_patterns:
        match = re.search(pattern, text_lower)
        if match:
            toc_start = match.start()
            break
    
    if toc_start < 0:
        return (-1, -1)
    
    # Find where TOC ends - look for first section that starts actual content
    # TOC typically ends when we see a section number followed by substantial content
    # Strategy: Find first occurrence of section pattern with > 500 chars before next section
    
    section_pattern = r'^\s*(\d+\.?\d*)\s+[A-Z]'
    matches = list(re.finditer(section_pattern, raw_text[toc_start:], flags=re.MULTILINE))
    
    if len(matches) < 2:
        # Couldn't find clear TOC end, estimate it
        return (toc_start, min(toc_start + 5000, len(raw_text)))
    
    # Check each match pair to find where content sections start
    for i in range(len(matches) - 1):
        current_pos = toc_start + matches[i].start()
        next_pos = toc_start + matches[i + 1].start()
        distance = next_pos - current_pos
        
        # If distance between sections is > 500 chars, we're past TOC
        if distance > 500:
            toc_end = current_pos
            LOGGER.info("Detected TOC range: %d to %d", toc_start, toc_end)
            return (toc_start, toc_end)
    
    # Fallback: estimate TOC is about 2-3 pages (~5000 chars)
    toc_end = min(toc_start + 5000, len(raw_text))
    LOGGER.info("Using estimated TOC range: %d to %d", toc_start, toc_end)
    return (toc_start, toc_end)


def _is_in_toc_range(position: int, toc_start: int, toc_end: int) -> bool:
    """Check if a position falls within the TOC range."""
    if toc_start < 0 or toc_end < 0:
        return False
    return toc_start <= position <= toc_end


def _extract_section_id(name: str) -> Optional[str]:
    """
    Extract a section id prefix like:
    - "3"
    - "11.3.2"
    - "15.2.4.1.1"
    - "A." or "B."
    
    Returns None for unnumbered sections like "Bibliography", "APPENDIX"
    """
    if not name:
        return None
    # Numbered like 3, 11.3, 15.2.4.1.1 or lettered A., B.
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*|[A-Z]\.)\b", name)
    if not m:
        return None
    return m.group(1).strip()


# ============================================================================
# PUBLIC API
# ============================================================================

def build_extract_prompt(filename: str, file_text: str) -> str:
    """Construct the exact prompt used in the original notebook."""
    return (
        f"{EXTRACT_SYSTEM_PROMPT}\n\nSchema: {json.dumps(GEMINI_SCHEMA, indent=2)}"
        f"\nCategory map: {json.dumps(FORM_CATEGORIES, indent=2)}\n\nFilename: {filename}\nDocument preview:\n{file_text}"
    )


@time_function
def gemini_extract_record(path: Path, max_retries: int = 0) -> Dict[str, Any]:
    """
    Extract structured information from documents using Gemini.

    Automatically routes to appropriate extraction strategy:
    - Scanned PDFs ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ OCR-based vision extraction
    - Large documents (>200k chars) ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Pass 0 structure + boundary detection + validation
    - Normal documents ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Single-pass JSON extraction (with fallback) + validation

    Args:
        path: Path to document file
        max_retries: Number of retry attempts on failure

    Returns:
        Extraction record with metadata, sections, and validation status
    """
    # Handle scanned PDFs separately
    if path.suffix.lower() == ".pdf" and is_scanned_pdf(path):
        LOGGER.info("Detected scanned PDF: %s", path.name)
        return gemini_extract_from_scanned_pdf(path, max_retries)

    # Read full document text (keep for validation)
    full_text = clean_text_for_llm(read_doc_for_llm(path))
    char_count = len(full_text)

    # Route based on document size
    if char_count > 200_000:
        LOGGER.info("Large document detected (%d chars), using Pass 0 + boundary detection for %s",
                   char_count, path.name)
        result = _extract_large_document_with_python_slicing(path, full_text, max_retries)
    else:
        LOGGER.debug("Normal-sized document (%d chars), using single-pass extraction for %s",
                    char_count, path.name)
        result = _gemini_extract_single_pass_with_fallback(path, full_text, max_retries)
    
    # Run validation if extraction succeeded (no parse_error or extraction_error)
    if not result.get("parse_error") and not result.get("extraction_error"):
        result = _validate_extraction(result, full_text, path)
    
    return result


def format_references_for_metadata(references: Dict[str, List[str]]) -> str:
    """Format the reference dictionary into a metadata string suitable for Chroma."""
    parts: List[str] = []
    if not isinstance(references, dict):
        return ""
    if references.get("forms"):
        parts.append(f"FORMS: {', '.join(references['forms'][:5])}")
    if references.get("regulations"):
        parts.append(f"REGS: {', '.join(references['regulations'][:3])}")
    if references.get("procedures"):
        parts.append(f"PROCS: {', '.join(references['procedures'][:3])}")
    if references.get("policies"):
        parts.append(f"POLICIES: {', '.join(references['policies'][:3])}")
    if references.get("reports"):
        parts.append(f"REPORTS: {', '.join(references['reports'][:3])}")
    if references.get("chapters"):
        parts.append(f"CHAPTERS: {', '.join(references['chapters'][:3])}")
    if references.get("sections"):
        parts.append(f"SECTIONS: {', '.join(references['sections'][:5])}")
    return "| ".join(parts) if parts else ""


def to_documents_from_gemini(path: Path, meta: Dict[str, Any]) -> List[Document]:
    """Transform Gemini output into section-level LlamaIndex Documents."""
    base_meta = {
        "source": path.name,
        "doc_type": meta.get("doc_type", "DOCUMENT"),
        "title": meta.get("title", path.stem),
        "topic": meta.get("normalized_topic") or meta.get("title", path.stem),
        "sections_found_by_gemini_str": ", ".join(
            [sec["name"] for sec in meta.get("sections", []) if isinstance(sec, dict) and "name" in sec]
        ),
    }
    if meta.get("category"):
        base_meta["form_category_name"] = meta["category"]
    if meta.get("form_number"):
        base_meta["form_number"] = meta["form_number"]
    if meta.get("hierarchy") and isinstance(meta.get("hierarchy"), list):
        base_meta["hierarchy"] = " > ".join([str(item) for item in meta["hierarchy"]])

    sections = [
        section for section in meta.get("sections", []) if isinstance(section, dict) and {"name", "content"} <= section.keys()
    ]
    documents: List[Document] = []
    references = meta.get("references", {}) if isinstance(meta.get("references"), dict) else {}
    references_str = format_references_for_metadata(references)

    if sections:
        for section in sections:
            name = section.get("name", "").strip()
            content = section.get("content", "").strip()
            if content:
                metadata = {**base_meta, "section": name, "references": references_str}
                documents.append(Document(text=content, metadata=metadata))

    if not documents:
        LOGGER.warning("No sections processed for %s, falling back to full document.", path.name)
        fallback_text = read_doc_for_llm(path)
        metadata = {**base_meta, "section": "Full Document Content", "references": references_str}
        documents.append(Document(text=fallback_text, metadata=metadata))

    return documents


def build_document_tree(meta: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """
    Build a hierarchical document tree from Gemini extraction metadata.

    Captures section structure with parent-child relationships for hierarchical retrieval.
    Each section node contains: section_id, title, level, parent_id, children, chunk_ids.

    Args:
        meta: Gemini extraction record with sections
        doc_id: Document identifier (typically filename without extension)

    Returns:
        Document tree structure
    """
    sections_raw = meta.get("sections", [])

    if not sections_raw:
        return {
            "doc_id": doc_id,
            "doc_type": meta.get("doc_type", "UNKNOWN"),
            "title": meta.get("title", doc_id),
            "sections": []
        }

    # Parse all sections and build flat list first
    parsed_sections = []
    for idx, section in enumerate(sections_raw):
        if not isinstance(section, dict):
            continue

        section_name = section.get("name", "").strip()
        if not section_name:
            continue

        parsed = _parse_section_identifier(section_name)
        if parsed:
            parsed["original_name"] = section_name
            parsed["chunk_ids"] = []
            parsed["children"] = []
            parsed["parent_id"] = None
            parsed_sections.append(parsed)

    # Build parent-child relationships based on section_id hierarchy
    for i, section in enumerate(parsed_sections):
        section_id = section["section_id"]

        # Skip non-numeric section IDs (can't determine hierarchy)
        if '.' not in section_id and not section_id.replace('.', '').isdigit():
            continue

        # Find parent: e.g., for "3.1.2", parent is "3.1"
        parts = section_id.split('.')
        if len(parts) > 1:
            parent_id = '.'.join(parts[:-1])

            # Find parent in parsed_sections
            for parent in parsed_sections:
                if parent["section_id"] == parent_id:
                    section["parent_id"] = parent_id
                    parent["children"].append(section)
                    break

    # Build tree structure: only top-level sections (those without parents) at root
    root_sections = [s for s in parsed_sections if s["parent_id"] is None]

    return {
        "doc_id": doc_id,
        "doc_type": meta.get("doc_type", "UNKNOWN"),
        "title": meta.get("title", doc_id),
        "sections": root_sections
    }


# ============================================================================
# VALIDATION
# ============================================================================

def _normalize_for_validation(text: str) -> str:
    """Normalize text for n-gram comparison."""
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove formatting artifacts but keep content
    text = re.sub(r'[ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ÃƒÂ¢Ã¢â‚¬â€Ã‚Â¦ÃƒÂ¢Ã¢â‚¬â€œÃ‚Âª\-ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ÂÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦]', '', text)
    # Remove special chars that might vary
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()


def _get_ngrams(text: str, n: int) -> Set[str]:
    """Extract n-grams from text."""
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _validate_extraction_ngram(raw_text: str, extracted_data: dict, n: int = 3) -> dict:
    """
    Validate extraction using n-gram matching to detect missing/hallucinated content.
    
    Args:
        raw_text: Original document text
        extracted_data: Extraction result with sections
        n: N-gram size (default 3)
    
    Returns:
        Validation metrics dict with 'passed' boolean and coverage stats
    """
    # Extract all section content
    sections = extracted_data.get('sections', [])
    extracted_content = " ".join(
        section.get("content", "") 
        for section in sections
        if isinstance(section, dict)
    )
    
    # Normalize both texts
    raw_norm = _normalize_for_validation(raw_text)
    extracted_norm = _normalize_for_validation(extracted_content)
    
    # Generate n-grams
    raw_ngrams = _get_ngrams(raw_norm, n)
    extracted_ngrams = _get_ngrams(extracted_norm, n)
    
    # Calculate metrics
    ngrams_in_both = raw_ngrams & extracted_ngrams
    missing_ngrams = raw_ngrams - extracted_ngrams
    hallucinated_ngrams = extracted_ngrams - raw_ngrams
    
    coverage = len(ngrams_in_both) / len(raw_ngrams) if raw_ngrams else 0
    hallucination_rate = len(hallucinated_ngrams) / len(extracted_ngrams) if extracted_ngrams else 0
    
    # Word-level stats for context
    raw_words = set(raw_norm.split())
    extracted_words = set(extracted_norm.split())
    word_coverage = len(raw_words & extracted_words) / len(raw_words) if raw_words else 0
    
    # Character count comparison
    length_ratio = len(extracted_norm) / len(raw_norm) if raw_norm else 0
    
    # Determine issues
    issues = []
    if coverage < 0.85:  # Using 85% threshold as specified
        issues.append(f"Low n-gram coverage: {coverage:.1%} - significant content missing")
    if hallucination_rate > 0.20:
        issues.append(f"High hallucination rate: {hallucination_rate:.1%}")
    if length_ratio < 0.60:
        issues.append(f"Extracted text too short: {length_ratio:.1%} of original")
    if length_ratio > 1.40:
        issues.append(f"Extracted text too long: {length_ratio:.1%} - possible duplication")
    
    return {
        'passed': coverage >= 0.85 and hallucination_rate <= 0.20,
        'ngram_coverage': coverage,
        'hallucination_rate': hallucination_rate,
        'word_coverage': word_coverage,
        'length_ratio': length_ratio,
        'missing_ngram_count': len(missing_ngrams),
        'hallucinated_ngram_count': len(hallucinated_ngrams),
        'issues': issues,
        'raw_char_count': len(raw_norm),
        'extracted_char_count': len(extracted_norm),
    }


def _validate_extraction(result: Dict[str, Any], raw_text: str, path: Path) -> Dict[str, Any]:
    """
    Run validation on successful extraction and flag if quality is insufficient.
    
    Args:
        result: Extraction result dict
        raw_text: Original document text
        path: Path to document file
    
    Returns:
        Result dict with validation metadata added
    """
    try:
        validation = _validate_extraction_ngram(raw_text, result, n=3)
        
        # Add validation metadata
        result['validation'] = {
            'ngram_coverage': validation['ngram_coverage'],
            'hallucination_rate': validation['hallucination_rate'],
            'word_coverage': validation['word_coverage'],
            'length_ratio': validation['length_ratio'],
        }
        
        # Flag if validation fails
        if not validation['passed']:
            error_msg = f"Validation failed: {', '.join(validation['issues'])}"
            result['validation_error'] = error_msg
            LOGGER.error("Validation failed for %s: %s", path.name, error_msg)
            LOGGER.error("Coverage: %.1f%%, Hallucination: %.1f%%",
                        validation['ngram_coverage'] * 100,
                        validation['hallucination_rate'] * 100)
        else:
            LOGGER.info("Validation passed for %s (coverage: %.1f%%)",
                       path.name, validation['ngram_coverage'] * 100)
        
    except Exception as exc:
        LOGGER.exception("Validation failed with exception for %s", path.name)
        result['validation_error'] = f"Validation exception: {exc}"
    
    return result


# ============================================================================
# OCR EXTRACTION
# ============================================================================

def gemini_extract_from_scanned_pdf(path: Path, max_retries: int = 2) -> Dict[str, Any]:
    """Extract from scanned PDFs using Gemini vision."""
    config = AppConfig.get()
    
    if not config.ocr_enabled:
        LOGGER.warning("OCR disabled, skipping: %s", path.name)
        return {
            "filename": path.name,
            "doc_type": "UNKNOWN",
            "title": path.stem,
            "sections": [],
            "references": {},
            "raw_output": "{}",
            "parse_error": "OCR disabled",
        }
    
    LOGGER.info("Processing scanned PDF with OCR: %s", path.name)
    
    try:
        with path.open("rb") as f:
            file_bytes = f.read()
        
        prompt = f"""{EXTRACT_SYSTEM_PROMPT}

**IMPORTANT**: This is a SCANNED PDF. You must:
1. OCR all text from the images
2. Extract structure per schema

Schema: {json.dumps(GEMINI_SCHEMA, indent=2)}
Category map: {json.dumps(FORM_CATEGORIES, indent=2)}

Filename: {path.name}"""
        
        response = config.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=GEMINI_SCHEMA,
            ),
        )
        
        raw = response.text or "{}"
        data = _parse_json_response(raw)
        data["ocr_used"] = True
        
    except Exception as exc:
        LOGGER.exception("OCR failed for %s", path.name)
        return {
            "filename": path.name,
            "doc_type": "UNKNOWN",
            "title": path.stem,
            "sections": [],
            "references": {},
            "raw_output": "{}",
            "parse_error": f"OCR failed: {exc}",
            "ocr_used": True,
        }
    
    data.setdefault("filename", path.name)
    data.setdefault("doc_type", "UNKNOWN")
    data.setdefault("title", path.stem)
    data.setdefault("sections", [])
    data.setdefault("references", {})
    data["raw_output"] = raw
    data["ocr_used"] = True
    
    return data


# ============================================================================
# BOUNDARY DETECTION
# ============================================================================

def _extract_with_boundary_detection(
    raw_text: str,
    structure: dict,
    file_path: Path,
) -> dict:
    """
    Extract section content using heading-based slicing.

    New strategy (simpler, less cursed):
    1. Use Pass 0 structure only for section *names* (no content).
    2. Split raw text into lines and compute their offsets.
    3. Detect heading-like lines via generic regex.
    4. For each section, independently fuzzy-match its name against the heading lines.
    5. Sort all matched sections by their character offsets.
    6. Slice content between consecutive section starts.
    7. Merge empty/header-only sections using existing _merge_empty_sections().
    """
    LOGGER.info("Using heading-based boundary detection for content extraction: %s", file_path.name)

    section_data = structure.get("_section_data", [])
    if not section_data:
        LOGGER.error("No section data in structure for boundary detection")
        return {
            **structure,
            "sections": [],
            "extraction_error": "No section data in Pass 0 structure",
        }

    #Pre-identify TOC range
    toc_start, toc_end = _find_toc_boundaries(raw_text)
    if toc_start >= 0:
        LOGGER.info("Found TOC from position %d to %d, will exclude from search", toc_start, toc_end)


    # 1) Index lines and offsets
    lines, offsets = _index_lines_for_slicing(raw_text)
    lines_norm = [_normalize_whitespace(l).lower() for l in lines]

    # 2) Find candidate heading lines
    heading_indices = _find_heading_candidates(lines)
    if not heading_indices:
        LOGGER.error("No heading-like lines detected in %s during boundary detection", file_path.name)
        return {
            **structure,
            "sections": [],
            "extraction_error": "No heading candidates detected in document text",
        }

    # 3) For each section, find best heading line
    section_starts: List[Tuple[int, Dict[str, Any]]] = []
    failed_sections: List[str] = []

    for i, section_info in enumerate(section_data):
        section_name = section_info.get("name", "") or section_info.get("section_id", "")
        if not section_name:
            LOGGER.warning("Missing section name for section %d in %s", i, file_path.name)
            failed_sections.append(f"Section {i}")
            continue

        next_info = section_data[i + 1] if i + 1 < len(section_data) else None

        start_pos = _find_section_start_by_id(raw_text, section_info, next_info, toc_start, toc_end)
        if start_pos < 0:
            LOGGER.error(
                "Could not locate section id for %r in %s",
                section_name,
                file_path.name,
            )
            failed_sections.append(section_name)
            continue

        section_starts.append((start_pos, {"name": section_name}))


    if not section_starts:
        LOGGER.error("Heading-based slicing could not locate any sections for %s", file_path.name)
        return {
            **structure,
            "sections": [],
            "extraction_error": "Could not locate any section headings in document text",
            "failed_sections": failed_sections,
        }

    # 4) Sort by actual position in document
    section_starts.sort(key=lambda x: x[0])

    # 5) Slice content between consecutive section starts
    extracted_sections: List[Dict[str, str]] = []

    for idx, (start_pos, section_info) in enumerate(section_starts):
        section_name = section_info.get("name", f"Section {idx}")
        end_pos = section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(raw_text)

        if start_pos < 0 or end_pos <= start_pos:
            LOGGER.warning(
                "Invalid slice for section %r in %s: start=%d end=%d",
                section_name,
                file_path.name,
                start_pos,
                end_pos,
            )
            failed_sections.append(section_name)
            continue

        content = raw_text[start_pos:end_pos].strip()
        if not content:
            LOGGER.warning(
                "Empty content slice for section %r in %s (start=%d, end=%d)",
                section_name,
                file_path.name,
                start_pos,
                end_pos,
            )
            failed_sections.append(section_name)
            continue

        # Remove header line if it matches the section name
        lines_content = content.split("\n", 1)
        if lines_content:
            header_norm = _normalize_whitespace(lines_content[0].lower())
            name_norm = _normalize_whitespace(section_name.lower())
            if header_norm.startswith(name_norm):
                content = lines_content[1] if len(lines_content) > 1 else ""

        LOGGER.debug(
            "Section '%s': start=%d end=%d preview=%r",
            section_name,
            start_pos,
            end_pos,
            raw_text[start_pos : start_pos + 120].replace("\n", " "),
        )

        extracted_sections.append(
            {
                "name": section_name,
                "content": content,
            }
        )

    if not extracted_sections:
        LOGGER.error("Heading-based slicing produced no populated sections for %s", file_path.name)
        return {
            **structure,
            "sections": [],
            "extraction_error": "No sections successfully sliced from document text",
            "failed_sections": failed_sections,
        }

    # 6) Merge header-only / tiny sections using your existing logic
    merged_sections = _merge_empty_sections(extracted_sections)

    # Failure rate: how many expected sections we totally missed
    expected_names = {s.get("name", "") for s in section_data if s.get("name")}
    found_names = {s["name"] for s in extracted_sections}
    missing = expected_names - found_names
    failure_rate = len(missing) / len(expected_names) if expected_names else 0

    if failure_rate > 0.5:
        LOGGER.error(
            "Heading-based detection failed: %d/%d sections not found",
            len(missing),
            len(expected_names),
        )
        return {
            **structure,
            "sections": merged_sections,
            "extraction_error": f"Failed to extract {len(missing)}/{len(expected_names)} sections",
        }

    result = {
        **structure,
        "sections": merged_sections,
    }

    if failed_sections:
        LOGGER.warning(
            "Heading-based slicing for %s completed with %d failed section(s): %s",
            file_path.name,
            len(failed_sections),
            failed_sections[:10],
        )
        result["failed_sections"] = failed_sections

    # Strip internal fields before returning
    result.pop("_section_data", None)
    result.pop("section_names", None)

    LOGGER.info(
        "Heading-based boundary detection complete: %d sections extracted (after merging)",
        len(merged_sections),
    )
    return result


def _merge_empty_sections(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge empty or header-only sections into the next non-empty *child-like* section.
    Rules:
    - An empty section is one with < 80 chars of content.
    - Consecutive empty sections all merge forward into the first non-empty descendant.
    - If no non-empty descendant exists, the empty section is dropped.
    - Never merge across different top-level section prefixes (e.g. "3" → "4" prohibited).
    """
    MERGE_THRESHOLD = 80  # chars
    merged: List[Dict[str, str]] = []
    i = 0
    total = len(sections)

    while i < total:
        sec = sections[i]
        name = sec["name"]
        content = sec["content"].strip()
        is_empty = len(content) < MERGE_THRESHOLD

        # Extract section prefix: e.g. "3", "3.1", "15.2.4"
        m = re.match(r"^([0-9]+(?:\.[0-9]+)*)", name)
        prefix = m.group(1) if m else None

        if not is_empty:
            # Normal section, keep as-is
            merged.append(sec)
            i += 1
            continue

        # Collect consecutive empty sections
        empty_chain = [sec]
        j = i + 1

        while j < total:
            next_sec = sections[j]
            next_content = next_sec["content"].strip()
            next_empty = len(next_content) < MERGE_THRESHOLD

            # Next one is also empty → keep chaining
            if next_empty:
                empty_chain.append(next_sec)
                j += 1
                continue

            # Next one is non-empty: check prefix compatibility
            m2 = re.match(r"^([0-9]+(?:\.[0-9]+)*)", next_sec["name"])
            next_prefix = m2.group(1) if m2 else None

            if prefix and next_prefix and next_prefix.startswith(prefix):
                # Valid merge target
                break
            else:
                # Not a child → can't merge → drop empty chain
                empty_chain = []
                break

        if empty_chain and j < total:
            # Merge the chain into the next section
            target = sections[j]
            merged_name = ": ".join([s["name"] for s in empty_chain] + [target["name"]])
            merged_content = target["content"]

            merged.append({"name": merged_name, "content": merged_content})
            i = j + 1
        else:
            # Nothing to merge into → drop the empties
            i = j

    return merged


# ============================================================================



def _extract_large_document_with_python_slicing(
    path: Path,
    full_text: str,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Extract large documents using Pass 0 structure + boundary detection.
    
    Strategy:
    1. Pass 0: Extract structure (metadata + section names + first sentences) via Gemini
    2. Boundary detection: Extract content by finding section boundaries in raw text
    3. Validation: Check extraction quality
    4. If Pass 0 fails OR boundary detection fails, flag as error
    
    Args:
        path: Path to document file
        full_text: Complete document text
        max_retries: Retry attempts for Pass 0
    
    Returns:
        Complete extraction record with all sections and validation status
    """
    LOGGER.info("=== Starting Pass 0 + boundary detection extraction for %s ===", path.name)
    
    # Pass 0: Get document structure (with section names + first sentences)
    LOGGER.info("Pass 0: Extracting structure...")
    structure = _gemini_get_structure(path.name, full_text, max_retries)
    
    # Check for structure extraction failure
    if structure.get("parse_error"):
        LOGGER.error("Pass 0 failed for %s: %s. Flagging as parse_error, will not embed.",
                    path.name, structure.get("parse_error"))
        return structure
    
    section_data = structure.get("_section_data", [])
    
    if not section_data:
        LOGGER.warning("No sections detected in Pass 0 for %s. Flagging as parse_error.",
                      path.name)
        structure["parse_error"] = "No sections identified in structure extraction"
        return structure
    
    LOGGER.info("Pass 0 successful: %d sections identified", len(section_data))
    
    # Use boundary detection for content extraction
    LOGGER.info("Extracting content via boundary detection...")
    result = _extract_with_boundary_detection(full_text, structure, path)
    
    # Check for slicing failure
    if result.get("extraction_error"):
        LOGGER.error("boundary detection failed for %s, will not embed.", path.name)
        return result
    
    LOGGER.info("=== Extraction complete: %d sections extracted ===",
               len(result.get("sections", [])))
    
    return result


# ============================================================================
# SINGLE-PASS EXTRACTION WITH FALLBACK
# ============================================================================

def _gemini_extract_single_pass_with_fallback(
    path: Path,
    full_text: str,
    max_retries: int
) -> Dict[str, Any]:
    """
    Single-pass JSON extraction with boundary detection fallback.
    
    Strategy:
    1. Try standard Gemini JSON extraction
    2. If JSON parsing fails, fall back to Pass 0 + boundary detection
    3. If Pass 0 also fails, flag as parse_error
    
    Args:
        path: Path to document file
        full_text: Complete document text
        max_retries: Number of retry attempts
    
    Returns:
        Extraction record (either from JSON or boundary detection)
    """
    config = AppConfig.get()
    preview = clean_text_for_llm(full_text)
    prompt = build_extract_prompt(path.name, preview)

    raw = "{}"
    last_error: Optional[str] = None

    # Attempt 1: Standard JSON extraction
    for attempt in range(1, max_retries + 2):
        try:
            response = config.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=GEMINI_SCHEMA,
                ),
            )
            raw = response.text or "{}"
            data = _parse_json_response(raw)
            
            # Success! Return the data
            data.setdefault("filename", path.name)
            data.setdefault("doc_type", "UNKNOWN")
            data.setdefault("title", path.stem)
            data.setdefault("sections", [])
            data.setdefault("references", {})
            data["raw_output"] = raw
            data.setdefault("ocr_used", False)
            
            LOGGER.info("Single-pass JSON extraction successful for %s", path.name)
            return data
            
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("Gemini JSON extraction failed for %s (attempt %d): %s", 
                          path.name, attempt, exc)
            
            if attempt > max_retries:
                # All JSON attempts failed - fall back to boundary detection
                LOGGER.warning("All JSON extraction attempts failed for %s, falling back to boundary detection",
                              path.name)
                break
            continue
    
    # Attempt 2: Fallback to Pass 0 + boundary detection
    LOGGER.info("Attempting boundary detection fallback for %s", path.name)
    
    try:
        # Extract structure first
        structure = _gemini_get_structure(path.name, full_text, max_retries=2)
        
        # Check if structure extraction failed
        if structure.get("parse_error"):
            LOGGER.error("Pass 0 fallback also failed for %s: %s",
                        path.name, structure.get("parse_error"))
            # Return with combined error message
            return {
                "filename": path.name,
                "doc_type": "UNKNOWN",
                "title": path.stem,
                "sections": [],
                "references": {},
                "raw_output": raw,
                "parse_error": f"JSON extraction failed ({last_error}), Pass 0 fallback also failed ({structure.get('parse_error')})"
            }
        
        # Structure OK, proceed with boundary detection
        result = _extract_with_boundary_detection(full_text, structure, path)
        
        # Check for extraction failure
        if result.get("extraction_error"):
            LOGGER.error("Boundary detection fallback also failed for %s", path.name)
            return {
                **result,
                "parse_error": f"Both JSON and boundary detection failed"
            }
        
        LOGGER.info("Boundary detection fallback successful for %s", path.name)
        return result
        
    except Exception as exc:
        LOGGER.exception("boundary detection fallback also failed for %s", path.name)
        return {
            "filename": path.name,
            "doc_type": "UNKNOWN",
            "title": path.stem,
            "sections": [],
            "references": {},
            "raw_output": raw,
            "parse_error": f"Both JSON extraction and boundary detection failed: {exc}"
        }


# ============================================================================
# STRUCTURE EXTRACTION (PASS 0)
# ============================================================================

def _gemini_get_structure(filename: str, text_preview: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Pass 0: Extract only document structure without section content.
    
    This lightweight extraction identifies all sections with names + first sentences
    for use as anchors in boundary detection. First sentences are stored in internal
    _section_data field and not included in final cached schema.
    
    Args:
        filename: Original document filename
        text_preview: Full document text for structure detection
        max_retries: Number of retry attempts on failure
    
    Returns:
        Dict with doc metadata, section_names array, and internal _section_data
    """
    # Schema for structure-only extraction with first sentences
    structure_schema = {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "doc_type": {"type": "string"},
            "title": {"type": "string"},
            "category": {"type": "string"},
            "form_number": {"type": "string"},
            "normalized_topic": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "first_sentence": {"type": "string"}
                    },
                    "required": ["name", "first_sentence"]
                },
                "description": "Complete ordered list of sections with name + first sentence"
            },
            "references": {
                "type": "object",
                "properties": {
                    "forms": {"type": "array", "items": {"type": "string"}},
                    "procedures": {"type": "array", "items": {"type": "string"}},
                    "regulations": {"type": "array", "items": {"type": "string"}},
                    "policies": {"type": "array", "items": {"type": "string"}},
                    "reports": {"type": "array", "items": {"type": "string"}},
                },
            }
        },
        "required": ["filename", "doc_type", "title", "sections"]
    }
    
    structure_prompt = f"""You are a maritime document analysis model. Your task is to extract ONLY document structure and metadata.

**CRITICAL INSTRUCTIONS:**
- Return ONLY the JSON object. Do not include any surrounding text, markdown code blocks (like ```json), or explanations.
- Extract ONLY document metadata and section names WITH FIRST SENTENCE
- DO NOT extract full section content
- For each section, provide: name AND first sentence (for text anchoring)
- Adhere strictly to the provided JSON schema

**Your Task:**
1. Identify document type, title, category, form number
2. List ALL sections with: name + first sentence (just first sentence, not full content!)
3. Identify any document references (form numbers, procedure names, regulations). Each entry to appear only once.

**What to output for each section:**
- name: Complete section name with number (e.g., "3.1 Safety Procedures")
- first_sentence: The first sentence of that section (for anchoring, ~10-20 words)

**Example:**
{{
  "sections": [
    {{
      "name": "3.1 Safety Procedures",
      "first_sentence": "All personnel must wear appropriate PPE when working on deck."
    }},
    {{
      "name": "3.2 Emergency Response",
      "first_sentence": "In case of emergency, activate the general alarm immediately."
    }}
  ]
}}

**What NOT to output:**
- Full section content (forbidden in this pass)
- Multiple sentences per section (just the first one!)
- Detailed descriptions

**Schema (follow exactly):**
{json.dumps(structure_schema, indent=2)}

**Category map:**
{json.dumps(FORM_CATEGORIES, indent=2)}

**Title extraction rules:**
- The filename should be included as-is in the 'filename' field"
- If it is a Form or Checklist, the 'title' must start with the form/checklist code and number, followed by a hyphen and the title
- For procedures, policies, regulations, and manuals, the 'title' should be the main title as prominently displayed
- Rule of Thumb: forms/checklists use filename, procedures/policies/regulations/manuals use extracted title

**doc_type options:**
- Form, Checklist, Procedure, Regulation, Policy, Manual
- Use 'Manual' only for equipment/instruction manuals
- Company IMS documents are 'Procedure'

**Document to analyze:**
Filename: {filename}

**REMINDER:**
- section name must be COMPLETE: number (if present) + full title
- Each reference should appear ONLY ONCE
- first_sentence should be ~10-20 words, just enough for text anchoring

Full document text:
{text_preview}
"""
    
    config = AppConfig.get()
    last_error: Optional[str] = None
    
    for attempt in range(1, max_retries + 2):
        try:
            response = config.client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=structure_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=structure_schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                ),
            )
            
            raw = response.text or "{}"

            # Check if response is suspiciously large
            if len(raw) > 50000:
                LOGGER.warning("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â  Structure extraction returned %d chars - likely contains section content!", len(raw))
            
            data = _parse_json_response(raw, allow_repair=False)

            # DEFENSIVE: Deduplicate and cap references
            refs = data.get("references", {})
            for key in refs:
                if isinstance(refs[key], list):
                    original_count = len(refs[key])
                    data["references"][key] = list(dict.fromkeys(refs[key]))
                    deduped_count = len(data["references"][key])
                    
                    if original_count != deduped_count:
                        LOGGER.warning("Deduped references.%s: %d ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ %d items", 
                                      key, original_count, deduped_count)
                    
                    if deduped_count > 100:
                        LOGGER.error("Reference spam detected in %s: %d items! Capping at 100.", 
                                    key, deduped_count)
                        data["references"][key] = data["references"][key][:100]
            
            # Process sections: extract name+first_sentence, store in internal field
            sections = data.get("sections", [])
            if not sections:
                LOGGER.warning("No sections found in structure extraction for %s", filename)
                data["sections"] = []
                data["_section_data"] = []
            else:
                # Store full section data (with first_sentence) internally
                data["_section_data"] = sections
                
                # For backward compatibility, also provide section_names array
                data["section_names"] = [s.get("name", "") for s in sections if s.get("name")]
            
            LOGGER.info("Structure extracted: %d sections identified for %s", 
                       len(sections), filename)
            return data
            
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("Structure extraction failed for %s (attempt %d): %s", 
                          filename, attempt, exc)
            
            if attempt > max_retries:
                return {
                    "filename": filename,
                    "doc_type": "UNKNOWN",
                    "title": Path(filename).stem,
                    "section_names": [],
                    "_section_data": [],
                    "references": {},
                    "parse_error": f"Structure extraction failed: {last_error}"
                }
            continue
    
    return {
        "filename": filename,
        "doc_type": "UNKNOWN",
        "title": Path(filename).stem,
        "section_names": [],
        "_section_data": [],
        "references": {},
        "parse_error": "Structure extraction failed after all retries"
    }


# ============================================================================
# JSON PARSING UTILITIES
# ============================================================================

def _parse_json_response(raw: str, *, allow_repair: bool = False) -> Dict[str, Any]:
    """
    Parse a JSON response from Gemini with extra robustness.
    """
    if not raw:
        raise ValueError("Empty JSON response from Gemini")

    # Prefer ```json fences if present
    fence_match = re.search(r"```(?:json)?\s*\n*(.*?)\s*```", raw, re.DOTALL)
    json_str = fence_match.group(1) if fence_match else raw
    json_str = json_str.strip()
    if not json_str:
        raise ValueError("Empty JSON after stripping fences")

    # Trim to the main JSON object between first '{' and last '}'
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = json_str[start : end + 1]

    # Strip non-printable control characters (keep \n, \t)
    json_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", json_str)

    # First parse attempt
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Primary JSON parse failed: %s", exc)

    # Try trailing-comma fix
    try:
        corrected = re.sub(r",\s*([\]}])", r"\1", json_str)
        return json.loads(corrected)
    except json.JSONDecodeError as exc:
        LOGGER.warning("JSON parse failed after comma fix: %s", exc)

    # Optional repair with Gemini itself
    if allow_repair:
        LOGGER.warning("Attempting Gemini-based JSON repair")
        repaired = _repair_broken_json_with_gemini(json_str)
        return repaired

    # Still broken ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ hard fail
    raise ValueError("Unable to parse Gemini JSON response after all repair attempts")


def _repair_broken_json_with_gemini(raw: str) -> Dict[str, Any]:
    """Ask Gemini to repair its own broken JSON."""
    config = AppConfig.get()
    snippet = raw[:8000]

    prompt = (
        "You previously attempted to output a JSON object matching this schema:\n\n"
        f"{json.dumps(GEMINI_SCHEMA, indent=2)}\n\n"
        "However, the output contained JSON syntax errors or was corrupted.\n\n"
        "Task:\n"
        "- Repair the JSON so that it is syntactically valid.\n"
        "- Ensure it conforms to the schema as closely as possible.\n"
        "- Return ONLY a single JSON object.\n\n"
        "Broken JSON output follows:\n"
        f"{snippet}\n"
    )

    response = config.client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=GEMINI_SCHEMA,
        ),
    )

    fixed = response.text or "{}"

    try:
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        LOGGER.error("Repair response was not valid JSON: %s", exc)
        raise


# ============================================================================
# SECTION PARSING UTILITIES
# ============================================================================

def _parse_section_identifier(section_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse section name to extract section_id, title, and level.

    Examples:
        "3. Discharge Procedure" → {section_id: "3", title: "Discharge Procedure", level: 1}
        "3.1 Pre-Discharge" → {section_id: "3.1", title: "Pre-Discharge", level: 2}
    """
    if not section_name:
        return None

    # Try to match numbered sections
    match = re.match(r'^([\d\.]+)\.?\s+(.+)$', section_name.strip())

    if match:
        section_id = match.group(1).rstrip('.')
        title = match.group(2).strip()
        level = section_id.count('.') + 1

        return {
            "section_id": section_id,
            "title": title,
            "level": level
        }

    # No numbering found - treat whole name as title
    return {
        "section_id": section_name.strip(),
        "title": section_name.strip(),
        "level": 1
    }

__all__ = [
    "build_extract_prompt",
    "gemini_extract_record",
    "gemini_extract_from_scanned_pdf",
    "format_references_for_metadata",
    "to_documents_from_gemini",
    "build_document_tree",
    "summarize_extraction_results",
]


def summarize_extraction_results(extraction_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of extraction results showing successes and failures.
    
    Args:
        extraction_records: List of extraction records from gemini_extract_record
    
    Returns:
        Summary dict with success/failure counts and details
    """
    total = len(extraction_records)
    successes = []
    failures = []
    
    for record in extraction_records:
        filename = record.get('filename', 'unknown')
        
        # Determine success/failure
        has_parse_error = bool(record.get('parse_error'))
        has_extraction_error = bool(record.get('extraction_error'))
        has_validation_issues = bool(record.get('validation_issues'))
        sections_count = len(record.get('sections', []))
        
        if has_parse_error or has_extraction_error:
            # Hard failure
            reason = record.get('parse_error') or record.get('extraction_error')
            failures.append({
                'filename': filename,
                'reason': reason,
                'type': 'parse_error' if has_parse_error else 'extraction_error'
            })
        elif sections_count == 0:
            # No sections extracted
            failures.append({
                'filename': filename,
                'reason': 'No sections extracted',
                'type': 'empty_result'
            })
        else:
            # Success (possibly with warnings)
            successes.append({
                'filename': filename,
                'sections': sections_count,
                'has_warnings': has_validation_issues,
                'extraction_method': 'boundary detection' if record.get('python_extraction') else 'Gemini JSON',
                'ocr_used': record.get('ocr_used', False)
            })
    
    summary = {
        'total': total,
        'succeeded': len(successes),
        'failed': len(failures),
        'success_rate': f"{len(successes)/total*100:.1f}%" if total > 0 else "N/A",
        'successes': successes,
        'failures': failures
    }
    
    return summary


def print_extraction_summary(extraction_records: List[Dict[str, Any]]) -> None:
    """
    Print a formatted extraction summary to console.
    
    Args:
        extraction_records: List of extraction records from gemini_extract_record
    """
    summary = summarize_extraction_results(extraction_records)
    
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total documents: {summary['total']}")
    print(f"Succeeded: {summary['succeeded']} ({summary['success_rate']})")
    print(f"Failed: {summary['failed']}")
    print()
    
    if summary['successes']:
        print("-" * 80)
        print("SUCCESSFUL EXTRACTIONS:")
        print("-" * 80)
        for item in summary['successes']:
            warnings = " [HAS WARNINGS]" if item['has_warnings'] else ""
            method = f" ({item['extraction_method']})"
            ocr = " [OCR]" if item['ocr_used'] else ""
            print(f"Ã¢Å“â€œ {item['filename']}: {item['sections']} sections{method}{ocr}{warnings}")
    
    if summary['failures']:
        print()
        print("-" * 80)
        print("FAILED EXTRACTIONS:")
        print("-" * 80)
        for item in summary['failures']:
            print(f"Ã¢Å“â€” {item['filename']}")
            print(f"  Type: {item['type']}")
            print(f"  Reason: {item['reason'][:150]}")  # Truncate long errors
            print()
    
    print("="*80 + "\n")