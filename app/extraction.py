"""Gemini extraction helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Document
from google.genai import types

from .config import AppConfig
from .constants import EXTRACT_SYSTEM_PROMPT, FORM_CATEGORIES, GEMINI_SCHEMA
from .files import clean_text_for_llm, read_doc_for_llm, is_scanned_pdf
from .logger import LOGGER
from .profiling import time_function

def build_extract_prompt(filename: str, file_text: str) -> str:
    """Construct the exact prompt used in the original notebook."""
    return (
        f"{EXTRACT_SYSTEM_PROMPT}\n\nSchema: {json.dumps(GEMINI_SCHEMA, indent=2)}"
        f"\nCategory map: {json.dumps(FORM_CATEGORIES, indent=2)}\n\nFilename: {filename}\nDocument preview:\n{file_text}"
    )

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


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse a JSON string, removing code fences and trailing commas if necessary."""
    json_match = re.search(r"```(?:json)?\s*\n*(.*?)\s*\n*```", raw, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else raw.strip()
    if not json_str:
        raise ValueError("Empty JSON response")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        corrected = re.sub(r",\s*([\]}])", r"\1", json_str)
        return json.loads(corrected)

@time_function
@time_function
def gemini_extract_record(path: Path, max_retries: int = 0) -> Dict[str, Any]:
    """
    Extract structured information from documents using Gemini.
    
    Automatically routes to appropriate extraction strategy:
    - Scanned PDFs → OCR-based vision extraction
    - Large documents (>250k chars) → Multi-pass extraction
    - Normal documents → Single-pass extraction
    
    Args:
        path: Path to document file
        max_retries: Number of retry attempts on failure
    
    Returns:
        Extraction record with metadata and sections
    """
    # Handle scanned PDFs separately
    if path.suffix.lower() == ".pdf" and is_scanned_pdf(path):
        LOGGER.info("Detected scanned PDF: %s", path.name)
        return gemini_extract_from_scanned_pdf(path, max_retries)
    
    # Read full document text
    full_text = clean_text_for_llm(read_doc_for_llm(path))
    char_count = len(full_text)
    
    # Route based on document size
    if char_count > 250_000:
        LOGGER.info("Large document detected (%d chars), using multi-pass extraction for %s", 
                   char_count, path.name)
        return _gemini_extract_large_document(path, full_text, max_retries)
    else:
        LOGGER.debug("Normal-sized document (%d chars), using single-pass extraction for %s",
                    char_count, path.name)
        return _gemini_extract_single_pass(path, full_text, max_retries)


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




def _create_overlapping_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks for multi-pass processing.
    
    Args:
        text: Full document text
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks with specified overlap
    
    Example:
        With chunk_size=150k, overlap=10k:
        Chunk 1: [0:150k]
        Chunk 2: [140k:290k]  (starts 10k before end of chunk 1)
        Chunk 3: [280k:430k]
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        
        if end >= len(text):
            break
        
        # Next chunk starts `overlap` chars before this one ends
        start = end - overlap
    
    return chunks


def _gemini_get_structure(filename: str, text_preview: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Pass 0: Extract only document structure without section content.
    
    This lightweight extraction identifies all sections and metadata,
    returning just section names (no content) to guide subsequent passes.
    
    Args:
        filename: Original document filename
        text_preview: First ~100k chars of document for structure detection
        max_retries: Number of retry attempts on failure
    
    Returns:
        Dict with doc metadata and list of section names (no content)
    """
    # Schema for structure-only extraction (no section content)
    structure_schema = {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "doc_type": {"type": "string"},
            "title": {"type": "string"},
            "category": {"type": "string"},
            "form_number": {"type": "string"},
            "normalized_topic": {"type": "string"},
            "hierarchy": {"type": "array", "items": {"type": "string"}},
            "references": {
                "type": "object",
                "properties": {
                    "forms": {"type": "array", "items": {"type": "string"}},
                    "procedures": {"type": "array", "items": {"type": "string"}},
                    "regulations": {"type": "array", "items": {"type": "string"}},
                    "policies": {"type": "array", "items": {"type": "string"}},
                    "reports": {"type": "array", "items": {"type": "string"}},
                    "chapters": {"type": "array", "items": {"type": "string"}},
                    "sections": {"type": "array", "items": {"type": "string"}},
                },
            },
            "section_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Complete ordered list of ALL section names in the document"
            }
        },
        "required": ["filename", "doc_type", "title", "section_names"]
    }
    
    # Create a structure-only prompt that explicitly forbids content extraction
    structure_prompt = f"""You are a maritime document analysis model performing STRUCTURE EXTRACTION ONLY.

**CRITICAL INSTRUCTIONS:**
- Extract ONLY document metadata and section names
- DO NOT extract any section content
- DO NOT include full text of sections
- Return ONLY a list of section names (strings), not content

**Your Task:**
1. Identify document type, title, category, form number
2. List ALL section/chapter names in order (just the names!)
3. Identify any document references (form numbers, procedure names, regulations)
4. Build document hierarchy if present

**What to output:**
- section_names: Array of strings (section names only, NO content)
- doc_type: One of: Form, Checklist, Procedure, Regulation, Policy, Manual
- title: Document title
- Optional: category, form_number, normalized_topic, hierarchy, references

**What NOT to output:**
- Section content (forbidden in this pass)
- Full text paragraphs
- Detailed descriptions

**Schema (follow exactly):**
{json.dumps(structure_schema, indent=2)}

**Category map:**
{json.dumps(FORM_CATEGORIES, indent=2)}

**Document to analyze:**
Filename: {filename}
Full document text:
{text_preview}
"""
    
    config = AppConfig.get()
    last_error: Optional[str] = None
    
    for attempt in range(1, max_retries + 2):
        try:
            response = config.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=structure_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=structure_schema,
                ),
            )
            
            raw = response.text or "{}"

            # DEBUG: Check what we actually got
            LOGGER.debug("Response object type: %s", type(response))
            LOGGER.debug("Response.text length: %d", len(response.text) if response.text else 0)
            
            # DEBUG: Check if response is suspiciously large (might contain content)
            if len(raw) > 50000:
                LOGGER.warning("⚠️  Structure extraction returned %d chars - likely contains section content!", len(raw))
                #LOGGER.debug("Response preview: %s...", raw[:200])
                LOGGER.error("Response starts with: %s", raw[:500])
                LOGGER.error("Response ends with: %s", raw[-500:])
            else:
                LOGGER.debug("Structure extraction response: %d chars (good)", len(raw))
            
            data = _parse_json_response(raw)
            
            # Ensure section_names exists
            if not data.get("section_names"):
                LOGGER.warning("No section names found in structure extraction for %s", filename)
                data["section_names"] = []
            
            LOGGER.info("Structure extracted: %d sections identified for %s", 
                       len(data["section_names"]), filename)
            return data
            
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("Structure extraction failed for %s (attempt %d): %s", 
                          filename, attempt, exc)
            
            # DEBUG: If JSON parse error, log the problematic response
            if "Unterminated string" in str(exc) or "JSONDecodeError" in str(exc):
                LOGGER.error("JSON parse error in structure extraction!")
                LOGGER.error("Response length: %d chars", len(raw) if 'raw' in locals() else 0)
                if 'raw' in locals() and raw:
                    LOGGER.error("Response preview (first 1000 chars): %s", raw[:1000])
                    LOGGER.error("Response preview (last 1000 chars): %s", raw[-1000:])
            
            if attempt > max_retries:
                # Return minimal structure on complete failure
                return {
                    "filename": filename,
                    "doc_type": "UNKNOWN",
                    "title": Path(filename).stem,
                    "section_names": [],
                    "references": {},
                    "parse_error": f"Structure extraction failed: {last_error}"
                }
            continue
    
    # Should never reach here, but just in case
    return {
        "filename": filename,
        "doc_type": "UNKNOWN",
        "title": Path(filename).stem,
        "section_names": [],
        "references": {}
    }


def _gemini_extract_chunk_with_resume(
    filename: str,
    chunk_text: str,
    all_section_names: List[str],
    already_completed_sections: List[str],  # NEW PARAMETER
    last_completed: Optional[str],
    is_first_pass: bool,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Extract section content from a text chunk with resume capability.
    
    Args:
        filename: Original document filename
        chunk_text: Text chunk to process
        all_section_names: Complete list of section names from Pass 0
        already_completed_sections: List of section names already extracted in previous passes
        last_completed: Name of last section completed in previous pass (None for first pass)
        is_first_pass: Whether this is the first content extraction pass
        max_retries: Number of retry attempts on failure
    
    Returns:
        Dict with sections array containing name+content pairs
    """
    # Build context about where we are in the document
    if is_first_pass:
        context = "This is the FIRST content extraction pass. Start from the beginning of the document."
    else:
        # Find position in section list
        try:
            last_idx = all_section_names.index(last_completed)
            next_section = all_section_names[last_idx + 1] if last_idx + 1 < len(all_section_names) else "END"
        except (ValueError, IndexError):
            next_section = "UNKNOWN"
        
        context = f"""This is a CONTINUATION pass.

**ALREADY COMPLETED (DO NOT RE-EXTRACT THESE):**
{json.dumps(already_completed_sections, indent=2)}

Last completed section: "{last_completed}"
Next expected section: "{next_section}"

**CRITICAL INSTRUCTIONS:**
1. Skip ALL sections in the "ALREADY COMPLETED" list above
2. Resume from section "{next_section}"
3. Extract ONLY sections that are NOT in the completed list
4. Even if you see text from completed sections in this chunk, DO NOT extract them again
"""
    
    prompt = f"""{EXTRACT_SYSTEM_PROMPT}

**MULTI-PASS CONTENT EXTRACTION MODE**

{context}

Complete section list YOU identified (for reference):
{json.dumps(all_section_names, indent=2)}

**Instructions:**
1. Extract COMPLETE content for each section you process
2. Process sections IN ORDER from the section list
3. Skip any sections already in the "ALREADY COMPLETED" list
4. Stop naturally at a section boundary when approaching output limits
5. Ensure every section you include has its full content

Schema: {json.dumps(GEMINI_SCHEMA, indent=2)}
Category map: {json.dumps(FORM_CATEGORIES, indent=2)}

Filename: {filename}
Document chunk:
{chunk_text}
"""
    
    config = AppConfig.get()
    last_error: Optional[str] = None
    
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
            
            sections_extracted = len(data.get("sections", []))
            LOGGER.info("Extracted %d sections from chunk", sections_extracted)
            
            return data
            
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("Chunk extraction failed (attempt %d): %s", attempt, exc)
            if attempt > max_retries:
                return {
                    "filename": filename,
                    "doc_type": "UNKNOWN",
                    "title": Path(filename).stem,
                    "sections": [],
                    "references": {},
                    "parse_error": f"Chunk extraction failed: {last_error}"
                }
            continue
    
    return {"sections": [], "references": {}}


def _stitch_multipass_results(
    structure: Dict[str, Any],
    chunk_results: List[Dict[str, Any]],
    filename: str
) -> Dict[str, Any]:
    """
    Merge structure and content extraction results into final record.
    
    Args:
        structure: Result from Pass 0 (metadata + section names)
        chunk_results: Results from Pass 1-N (section content)
        filename: Original document filename
    
    Returns:
        Complete extraction record with all metadata and section content
    """
    # Start with structure metadata
    merged = {
        "filename": filename,
        "doc_type": structure.get("doc_type", "UNKNOWN"),
        "title": structure.get("title", Path(filename).stem),
        "hierarchy": structure.get("hierarchy", []),
        "references": structure.get("references", {}),
        "sections": [],
        "multi_pass_extraction": True,
        "num_passes": len(chunk_results),
        "ocr_used": False,
    }
    
    # Copy optional fields if present
    if structure.get("category"):
        merged["category"] = structure["category"]
    if structure.get("form_number"):
        merged["form_number"] = structure["form_number"]
    if structure.get("normalized_topic"):
        merged["normalized_topic"] = structure["normalized_topic"]
    
    # Merge references from all passes (union)
    all_refs = dict(structure.get("references", {}))
    for result in chunk_results:
        refs = result.get("references", {})
        if isinstance(refs, dict):
            for key, values in refs.items():
                if key not in all_refs:
                    all_refs[key] = []
                if isinstance(values, list):
                    all_refs[key].extend(values)
    
    # Deduplicate reference lists
    for key in all_refs:
        if isinstance(all_refs[key], list):
            all_refs[key] = list(dict.fromkeys(all_refs[key]))  # Preserve order while deduping
    
    merged["references"] = all_refs
    
    # Collect all sections from chunk results
    all_sections = []
    seen_section_names = set()
    
    for result in chunk_results:
        sections = result.get("sections", [])
        
        for section in sections:
            if not isinstance(section, dict):
                continue
            
            name = section.get("name", "").strip()
            content = section.get("content", "").strip()
            
            if not name or not content:
                continue
            
            # Check for duplicate sections (shouldn't happen, but defensive)
            if name in seen_section_names:
                LOGGER.warning("Duplicate section detected in stitching: %s", name)
                continue
            
            seen_section_names.add(name)
            all_sections.append({
                "name": name,
                "content": content
            })
    
    merged["sections"] = all_sections
    
    # Validation: check coverage against structure
    expected_sections = set(structure.get("section_names", []))
    extracted_sections = set(s["name"] for s in all_sections)
    
    missing_sections = expected_sections - extracted_sections
    if missing_sections:
        LOGGER.warning("Multi-pass extraction missed %d sections for %s: %s",
                      len(missing_sections), filename, list(missing_sections)[:5])
        merged["extraction_warning"] = f"Missing {len(missing_sections)} sections"
    
    LOGGER.info("Stitched %d passes into %d sections for %s (expected %d)",
               len(chunk_results), len(all_sections), filename, len(expected_sections))
    
    return merged


def _gemini_extract_large_document(path: Path, full_text: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Multi-pass extraction for large documents exceeding output token limits.
    
    Strategy:
    - Pass 0: Extract structure (metadata + section names only)
    - Pass 1-N: Extract content in chunks with state-aware resumption
    - Stitch: Merge all results into final extraction record
    
    Args:
        path: Path to document file
        full_text: Complete document text
        max_retries: Retry attempts per pass
    
    Returns:
        Complete extraction record with all sections
    """
    LOGGER.info("=== Starting multi-pass extraction for %s ===", path.name)
    
    # Pass 0: Get document structure (no content)
    LOGGER.info("Pass 0: Extracting structure...")
    structure = _gemini_get_structure(path.name, full_text, max_retries)  # FIXED: full_text instead of [:100_000]
    
    section_names = structure.get("section_names", [])
    
    if not section_names:
        LOGGER.warning("No sections detected in structure pass, falling back to single-pass")
        return _gemini_extract_single_pass(path, full_text, max_retries)
    
    LOGGER.info("Structure extracted: %d sections identified", len(section_names))
    
    # Create overlapping chunks
    chunks = _create_overlapping_chunks(full_text, chunk_size=150_000, overlap=10_000)
    LOGGER.info("Document split into %d chunks for processing", len(chunks))
    
    # Multi-pass content extraction
    chunk_results = []
    last_completed_section = None
    already_completed_sections = []  # NEW: Track all completed sections
    total_sections_extracted = 0
    
    for i, chunk_text in enumerate(chunks):
        LOGGER.info("Pass %d/%d: Extracting content from chunk...", i + 1, len(chunks))
        
        result = _gemini_extract_chunk_with_resume(
            filename=path.name,
            chunk_text=chunk_text,
            all_section_names=section_names,
            already_completed_sections=already_completed_sections,  # NEW: Pass completed list
            last_completed=last_completed_section,
            is_first_pass=(i == 0),
            max_retries=max_retries
        )
        
        extracted_sections = result.get("sections", [])
        
        if not extracted_sections:
            LOGGER.warning("No sections extracted in pass %d, stopping", i + 1)
            break
        
        chunk_results.append(result)
        
        # NEW: Add all newly extracted section names to the completed list
        for section in extracted_sections:
            section_name = section.get("name", "").strip()
            if section_name and section_name not in already_completed_sections:
                already_completed_sections.append(section_name)
        
        total_sections_extracted = len(already_completed_sections)  # CHANGED: Use unique count
        
        # Track progress
        last_completed_section = extracted_sections[-1].get("name")
        LOGGER.info("Pass %d complete: extracted %d NEW sections (total unique so far: %d/%d)",
                   i + 1, len(extracted_sections), total_sections_extracted, len(section_names))
        LOGGER.info("Last completed section: '%s'", last_completed_section)
        
        # Early termination if we've extracted all expected sections
        if total_sections_extracted >= len(section_names):
            LOGGER.info("All expected sections extracted, stopping early")
            break
    
    # Stitch everything together
    LOGGER.info("Stitching %d passes into final result...", len(chunk_results))
    final_result = _stitch_multipass_results(structure, chunk_results, path.name)
    
    LOGGER.info("=== Multi-pass extraction complete: %d sections extracted ===",
               len(final_result.get("sections", [])))
    
    return final_result


def _gemini_extract_single_pass(path: Path, full_text: str, max_retries: int) -> Dict[str, Any]:
    """
    Original single-pass extraction for normal-sized documents.
    
    This is the existing extraction logic, separated out for clarity.
    """
    config = AppConfig.get()
    preview = clean_text_for_llm(full_text)
    prompt = build_extract_prompt(path.name, preview)

    raw = "{}"
    last_error: Optional[str] = None

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
            last_error = None
            break
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("Gemini extraction failed for %s (attempt %s): %s", path.name, attempt, exc)
            if attempt > max_retries:
                data = {"raw_output": raw, "parse_error": last_error}
            continue

    data.setdefault("filename", path.name)
    data.setdefault("doc_type", "UNKNOWN")
    data.setdefault("title", path.stem)
    data.setdefault("sections", [])
    data.setdefault("references", {})
    data["raw_output"] = raw
    if last_error:
        data["parse_error"] = last_error
    data.setdefault("ocr_used", False)
    return data


__all__ = [
    "build_extract_prompt",
    "gemini_extract_record",
    "gemini_extract_from_scanned_pdf", 
    "format_references_for_metadata",
    "to_documents_from_gemini",
]