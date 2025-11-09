"""File helpers: hashing, JSONL cache, document reading and cleaning."""

from __future__ import annotations

import json
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .logger import LOGGER

def _df_to_markdown(df, max_rows: int = 100, max_cols: int = 12) -> str:
    """Convert DataFrame to markdown table for LLM consumption."""
    if df is None or df.empty:
        return ""
    
    df = df.iloc[:max_rows, :max_cols].copy()
    
    # Heuristic: if first row looks like headers, promote it
    try:
        first_row = df.iloc[0]
        looks_like_headers = first_row.apply(
            lambda x: isinstance(x, str) and len(str(x).strip()) > 0 and len(str(x)) < 40
        ).mean() >= 0.6
        
        if looks_like_headers:
            df.columns = first_row
            df = df[1:].reset_index(drop=True)
    except Exception:
        pass
    
    df = df.fillna("")
    headers = [str(col).strip() or f"Col{i}" for i, col in enumerate(df.columns)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" --- "] * len(headers)) + "|",
    ]
    
    for _, row in df.iterrows():
        cells = [str(val).strip().replace("\n", " ") for val in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)


def _pymupdf_page_to_text(page) -> str:
    """Extract text from PyMuPDF page with layout awareness."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return page.get_text("text") or ""
    
    text_blocks = [b for b in blocks if len(b) >= 5 and (b[4] or "").strip()]
    text_blocks.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right
    
    return "\n\n".join(str(b[4]).strip() for b in text_blocks)


def is_scanned_pdf(path: Path) -> bool:
    """Detect if PDF is scanned (needs OCR) vs born-digital."""
    try:
        import fitz
    except ImportError:
        return False
    
    try:
        with fitz.open(str(path)) as doc:
            pages_to_check = min(3, len(doc))
            text_chars_total = 0
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text("text") or ""
                text_chars_total += len(text.strip())
            
            return text_chars_total < 100
    except Exception as exc:
        LOGGER.warning("Failed to detect if %s is scanned: %s", path.name, exc)
        return False

def file_fingerprint(path: Path) -> Dict[str, Any]:
    """Return a simple fingerprint for change detection."""
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime": stat.st_mtime}


def iter_library_files(root: Path) -> Iterable[Path]:
    """Yield supported document paths from the library root."""
    for ext in ("*.docx", "*.doc", "*.xlsx", "*.xls", "*.pdf", "*.txt"):
        yield from root.glob(ext)


def current_files_index(root: Path) -> Dict[str, Dict[str, Any]]:
    """Build an index of current files keyed by filename."""
    return {p.name: file_fingerprint(p) for p in iter_library_files(root)}


def load_jsonl(path: Path) -> Dict[str, Any]:
    """Read a JSONL cache file into a dict keyed by filename."""
    data: Dict[str, Any] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                record = json.loads(line)
                filename = record.get("filename")
                if filename:
                    data[filename] = record
            except json.JSONDecodeError:
                continue
    return data


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Overwrite a JSONL file with records."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def upsert_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    """Update or append a record in a JSONL cache."""
    data = load_jsonl(path)
    data[record["filename"]] = record
    write_jsonl(path, data.values())


def collapse_repeated_lines(text: str, max_repeats: int = 3) -> str:
    """Collapse repeated lines to avoid redundant table headers."""
    lines = text.splitlines()
    cleaned: List[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        repeat_count = 1
        while (
            index + repeat_count < len(lines)
            and lines[index + repeat_count].strip() == current.strip()
            and repeat_count < max_repeats
        ):
            repeat_count += 1
        cleaned.extend(lines[index : index + min(repeat_count, max_repeats)])
        if repeat_count > max_repeats:
            index += repeat_count
        else:
            index += repeat_count
    return "\n".join(cleaned)


def clean_text_for_llm(text: str) -> str:
    """Aggressive text normalisation to reduce parsing noise."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r"([\\-_=]{4,})", "----", text)
    text = re.sub(r"(\.{4,})", "...", text)
    text = re.sub(r"([,.;:!?])\1{2,}", r"\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def read_doc_for_llm(path: Path, max_chars: Optional[int] = None) -> str:
    """Extract text from various document formats."""
    ext = path.suffix.lower()
    text = ""
    try:
        if ext == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".docx":
            from docx import Document as DocxDocument

            document = DocxDocument(str(path))
            parts: List[str] = []

            # Extract header tables (where your form metadata is!)
            for section in document.sections:
                header = section.header
                for table in header.tables:
                    for row in table.rows:
                        row_parts = []
                        for cell in row.cells:
                            cell_text = cell.text.strip()
                            if cell_text and cell_text not in row_parts:  # Avoid duplicates
                                row_parts.append(cell_text)
                        if row_parts:
                            parts.append(' | '.join(row_parts))
                
                # Extract header paragraphs too (some forms use these)
                for para in header.paragraphs:
                    if para.text.strip():
                        parts.append(para.text)


            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    parts.append(paragraph.text)

            for table in document.tables:
                table_text_parts = []
                for row in table.rows:
                    row_text_parts = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text_parts.append(cell_text)
                    if row_text_parts:
                        table_text_parts.append(" | ".join(row_text_parts))
                table_full_text = "\n".join(table_text_parts)
                max_table_chars = 10000
                if len(table_full_text) > max_table_chars:
                    parts.append(table_full_text[:max_table_chars])
                else:
                    parts.append(table_full_text)

            text = collapse_repeated_lines("\n".join(parts))

        elif ext == ".doc":
            # Legacy format - will be handled by Gemini vision in extraction.py
            # Just return empty string here, actual extraction happens at Gemini level
            LOGGER.debug("Legacy .doc detected: %s - deferring to LLM", path.name)
            return ""

        elif ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                
                with fitz.open(str(path)) as doc:
                    page_texts: List[str] = []
                    for page in doc:
                        page_text = _pymupdf_page_to_text(page)
                        if page_text.strip():
                            page_texts.append(page_text)
                    
                    text = "\n\n".join(page_texts)
                    
                    if len(text.strip()) < 100 and len(doc) > 0:
                        LOGGER.warning("PDF %s yielded minimal text - may be scanned", path.name)
            
            except ImportError:
                LOGGER.debug("PyMuPDF not available, using PyPDF2 for %s", path.name)
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                pages = []
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        pages.append(extracted)
                text = "\n".join(pages)
            
            except Exception as exc:
                LOGGER.warning("PyMuPDF failed for %s: %s", path.name, exc)
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(path))
                    pages = []
                    for page in reader.pages:
                        extracted = page.extract_text() or ""
                        if extracted.strip():
                            pages.append(extracted)
                    text = "\n".join(pages)
                except Exception:
                    text = ""

        elif ext in {".xlsx", ".xls"}:
            excel = pd.ExcelFile(str(path))
            chunks: List[str] = []
            
            for sheet_name in excel.sheet_names[:5]:
                try:
                    df = pd.read_excel(str(path), sheet_name=sheet_name, nrows=100, header=None)
                    df.dropna(axis=1, how="all", inplace=True)
                    df.dropna(thresh=2, inplace=True)
                    df = df[df.iloc[:, 0].notna() | df.any(axis=1)].copy()
                    
                    if not df.empty:
                        md = _df_to_markdown(df)
                        if md:
                            chunks.append(f"[Sheet: {sheet_name}]\n{md}")
                
                except Exception as exc:
                    LOGGER.warning("Could not read sheet '%s' from %s: %s", sheet_name, path.name, exc)
                    continue
            
            text = "\n\n".join(chunks)

    except Exception as exc:  # pragma: no cover - defensive path
        text = f"[READ_ERROR] {exc}"
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


__all__ = [
    "file_fingerprint",
    "iter_library_files",
    "current_files_index",
    "load_jsonl",
    "write_jsonl",
    "upsert_jsonl_record",
    "collapse_repeated_lines",
    "clean_text_for_llm",
    "read_doc_for_llm",
    "is_scanned_pdf",
]
