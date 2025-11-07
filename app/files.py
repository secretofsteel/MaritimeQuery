"""File helpers: hashing, JSONL cache, document reading and cleaning."""

"""
Drop-in upgrades for MaritimeQuery/app/files.py
- Adds PyMuPDF-first PDF extraction with structure preservation
- Auto-detects scanned pages and (best-effort) OCR fallback via OCRmyPDF
- Safer DOCX/Excel handling and Markdown-friendly outputs
- Truncation marker and hyphenation cleanup
- Recursive file iterator for supported types
"""
from __future__ import annotations

import io
import os
import re
import json
import csv
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from .logger import LOGGER
from hashlib import md5

# Your existing imports likely include these
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # handled at runtime

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import pandas as pd
except Exception:
    pd = None

SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}

# ---------------------------------------------------------------------------
# Helper utilities for file hashing and JSONL cache
# ---------------------------------------------------------------------------

def _convert_doc_with_soffice(src: Path, target_ext: str, timeout: int = 120) -> Optional[Path]:
    """
    Convert legacy .doc using headless LibreOffice.
    target_ext: 'docx' or 'pdf'
    Returns path to converted file or None on failure.
    """
    assert target_ext in {"docx", "pdf"}
    try:
        outdir = src.parent
        # LibreOffice picks output type from --convert-to
        # docx filter: "docx:MS Word 2007 XML"
        fmt = "docx:MS Word 2007 XML" if target_ext == "docx" else "pdf:writer_pdf_Export"
        cmd = [
            "soffice", "--headless", "--nologo", "--nodefault", "--nofirststartwizard",
            "--convert-to", fmt, "--outdir", str(outdir), str(src)
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        cand = src.with_suffix(f".{target_ext}")
        return cand if cand.exists() else None
    except Exception:
        return None

# light check that the binary exists
def _soffice_available() -> bool:
    return shutil.which("soffice") is not None

def _pymupdf_is_scanned_page(page) -> bool:
    """Heuristic: page with near-zero text and mostly image blocks.
    Not perfect, good enough to trigger OCR.
    """
    try:
        txt = page.get_text("text") or ""
        if len(txt.strip()) >= 32:
            return False
        j = page.get_text("rawdict") or {}
        blocks = j.get("blocks", [])
        if not blocks and not txt.strip():
            return True
        img_blocks = sum(1 for b in blocks if b.get("type") == 1)
        return img_blocks >= max(1, len(blocks) - 1)
    except Exception:
        return False


def _pymupdf_page_to_markdown(page) -> str:
    """Lightweight, layout-aware text using block reading order.
    Upgrade path: inspect spans and map larger fonts to #/## headings,
    convert bullet glyphs to '-', etc.
    """
    try:
        blocks = page.get_text("blocks")  # [x0,y0,x1,y1,txt,...]
    except Exception:
        return page.get_text("text") or ""
    blocks = [b for b in blocks if len(b) >= 5 and (b[4] or "").strip()]
    blocks.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, then left-to-right
    out: List[str] = []
    for b in blocks:
        out.append(str(b[4]).strip())
    return "\n\n".join(out)


def _df_to_markdown(df, max_rows: int = 30, max_cols: int = 12) -> str:
    if df is None or df.empty:
        return ""
    df = df.iloc[:max_rows, :max_cols].copy()
    # Try to promote first row to header if it looks like headers
    try:
        if df.iloc[0].apply(lambda x: isinstance(x, str) and len(str(x)) < 40).mean() >= 0.6:
            df.columns = df.iloc[0]
            df = df[1:]
    except Exception:
        pass
    df = df.fillna("")
    headers = list(map(str, df.columns))
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" --- "] * len(headers)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(map(lambda x: str(x), row.values)) + " |")
    return "\n".join(lines)


def file_fingerprint(path: Path) -> Dict[str, Any]:
    """Return a simple fingerprint for change detection."""
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime": stat.st_mtime}


def iter_library_files(root: Path) -> Iterator[Path]:
    """Yield supported files under root (recursively)."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


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


# ---------------------------------------------------------------------------
# Text cleaning for LLM consumption
# ---------------------------------------------------------------------------

def clean_text_for_llm(text: str) -> str:
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks produced by PDF extraction
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing spaces per line
    text = "\n".join(s.rstrip() for s in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# Main dispatcher (REPLACE your read_doc_for_llm with this)
# ---------------------------------------------------------------------------

def read_doc_for_llm(path: Path, max_chars: Optional[int] = None) -> str:
    """Unified text extraction for PDF/DOCX/Excel/CSV/TXT.
    - PDF: PyMuPDF first; if scanned, best-effort OCR via OCRmyPDF, fallback to PyPDF2
    - DOCX: paragraphs + tables, no over-eager de-duplication
    - Excel/CSV: Markdown-friendly tables
    """
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        return ""

    text = ""

    # -------------------- PDF --------------------
    if ext == ".pdf":
        try:
            if fitz is not None:
                with fitz.open(str(path)) as doc:
                    page_texts: List[str] = []
                    scanned_pages = 0
                    for page in doc:
                        if _pymupdf_is_scanned_page(page):
                            scanned_pages += 1
                        md = _pymupdf_page_to_markdown(page)
                        if md.strip():
                            page_texts.append(md)
                    text = "\n\n".join(page_texts)

                    # If looks scanned or extraction was weak, try OCR fallback
                    if (not text.strip()) or scanned_pages >= max(1, len(doc) // 2):
                        try:
                            with tempfile.TemporaryDirectory() as td:
                                ocr_out = Path(td) / "ocr.pdf"
                                cmd = [
                                    "ocrmypdf",
                                    "--skip-text",
                                    "--tesseract-timeout",
                                    "300",
                                    "--optimize",
                                    "1",
                                    "--output-type",
                                    "pdf",
                                    str(path),
                                    str(ocr_out),
                                ]
                                subprocess.run(cmd, check=True, capture_output=True)
                                with fitz.open(str(ocr_out)) as ocr_doc:
                                    page_texts = []
                                    for page in ocr_doc:
                                        md = _pymupdf_page_to_markdown(page)
                                        if md.strip():
                                            page_texts.append(md)
                                    if page_texts:
                                        text = "\n\n".join(page_texts)
                        except Exception:
                            # OCR fallback unavailable or failed; keep whatever we had
                            pass
            else:
                # PyMuPDF not available: last-resort PyPDF2
                if PdfReader is None:
                    return ""
                reader = PdfReader(str(path))
                pages: List[str] = []
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        pages.append(extracted)
                text = "\n".join(pages)
        except Exception:
            # As an ultimate fallback, try PyPDF2 even if PyMuPDF failed mid-run
            if PdfReader is not None:
                try:
                    reader = PdfReader(str(path))
                    pages = []
                    for page in reader.pages:
                        extracted = page.extract_text() or ""
                        if extracted.strip():
                            pages.append(extracted)
                    text = "\n".join(pages)
                except Exception:
                    text = ""
            else:
                text = ""

    # -------------------- DOCX --------------------
    elif ext == ".docx":
        if docx is None:
            return ""
        d = docx.Document(str(path))
        parts: List[str] = []
        # paragraphs
        for p in d.paragraphs:
            t = (p.text or "").strip()
            if t:
                parts.append(t)
        # tables (no over-eager de-dup)
        for tbl in d.tables:
            for row in tbl.rows:
                row_text_parts: List[str] = []
                for cell in row.cells:
                    cell_text = (cell.text or "").strip()
                    if cell_text:
                        row_text_parts.append(cell_text)
                if row_text_parts:
                    parts.append(" | ".join(row_text_parts))
        text = "\n".join(parts)

    # -------------------- DOC (legacy Word) --------------------
    elif ext == ".doc":
        if not _soffice_available():
            # No LibreOffice in this environment; nothing to do
            return ""
        docx_path = _convert_doc_with_soffice(path, "docx")
        if docx_path and docx_path.exists():
            return read_doc_for_llm(docx_path, max_chars=max_chars)

        pdf_path = _convert_doc_with_soffice(path, "pdf")
        if pdf_path and pdf_path.exists():
            return read_doc_for_llm(pdf_path, max_chars=max_chars)

        return ""


    # -------------------- Excel --------------------
    elif ext in {".xlsx", ".xls"}:
        if pd is None:
            return ""
        try:
            xls = pd.ExcelFile(str(path))
        except Exception:
            return ""
        chunks: List[str] = []
        for sheet_name in xls.sheet_names:
            try:
                df = xls.parse(sheet_name)
            except Exception:
                continue
            md = _df_to_markdown(df)
            if md:
                chunks.append(f"# Sheet: {sheet_name}\n{md}")
        text = "\n\n".join(chunks)

    # -------------------- CSV --------------------
    elif ext == ".csv":
        try:
            if pd is not None:
                df = pd.read_csv(str(path))
                text = _df_to_markdown(df, max_rows=200, max_cols=24)
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception:
            text = ""

    # -------------------- TXT --------------------
    elif ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""

    # Post-process and truncate with marker
    text = clean_text_for_llm(text)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
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
    "SUPPORTED_EXTS",
]
