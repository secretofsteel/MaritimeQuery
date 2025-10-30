"""File helpers: hashing, JSONL cache, document reading and cleaning."""

from __future__ import annotations

import json
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .logger import LOGGER


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
            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    parts.append(paragraph.text)

            for table in document.tables:
                table_text_parts = []
                for row in table.rows:
                    row_text_parts = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text and cell_text not in row_text_parts:
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
            import textract

            text = textract.process(str(path)).decode("utf-8")

        elif ext == ".pdf":
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages.append(extracted)
            text = "\n".join(pages)

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
                        chunks.append(f"[Sheet: {sheet_name}]\n" + df.to_string(index=False, header=False))
                except Exception as exc:  # pragma: no cover - defensive path
                    LOGGER.warning("Could not read sheet '%s' from %s: %s", sheet_name, path.name, exc)
                    continue
            text = "\n\n".join(chunks)

        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

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
]
