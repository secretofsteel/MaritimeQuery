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
from .files import clean_text_for_llm, read_doc_for_llm
from .logger import LOGGER


def build_extract_prompt(filename: str, file_text: str) -> str:
    """Construct the exact prompt used in the original notebook."""
    return (
        f"{EXTRACT_SYSTEM_PROMPT}\n\nSchema: {json.dumps(GEMINI_SCHEMA, indent=2)}"
        f"\nCategory map: {json.dumps(FORM_CATEGORIES, indent=2)}\n\nFilename: {filename}\nDocument preview:\n{file_text}"
    )


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


def gemini_extract_record(path: Path, max_retries: int = 2) -> Dict[str, Any]:
    """Call Gemini with retries; preserve the exact schema and keys."""
    config = AppConfig.get()
    preview = clean_text_for_llm(read_doc_for_llm(path))
    prompt = build_extract_prompt(path.name, preview)

    raw = "{}"
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 2):
        try:
            response = config.client.models.generate_content(
                model="gemini-flash-lite-latest",
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
        except Exception as exc:  # pragma: no cover - defensive path
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
    return data


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


__all__ = [
    "build_extract_prompt",
    "gemini_extract_record",
    "format_references_for_metadata",
    "to_documents_from_gemini",
]
