"""File helpers: hashing, JSONL cache, document reading and cleaning."""

from __future__ import annotations

import io
import json
import re
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

def extract_with_pymupdf4llm(path: Path) -> Optional[str]:
    """
    Try high-level extraction via PyMuPDF Layout + PyMuPDF4LLM.

    Returns:
        Markdown string on success, or None on any failure.
    """
    try:
        import fitz  # PyMuPDF
        import pymupdf.layout  # enable layout engine
        import pymupdf4llm
    except ImportError:
        return None

    try:
        # Use Document object so it also works with non-PDF formats that fitz.open supports.
        doc = fitz.open(str(path))
        try:
            md = pymupdf4llm.to_markdown(
                doc,
                header=False,
                footer=False,
                table_strategy="lines_strict",
                page_separators=False,
            )
        finally:
            doc.close()

        if isinstance(md, str):
            text = md.strip()
        elif isinstance(md, list):
            # page_chunks mode etc. Join "text" fields if present.
            parts = [
                (chunk.get("text") or "").strip()
                for chunk in md
                if isinstance(chunk, dict)
            ]
            text = "\n\n".join(p for p in parts if p)
        else:
            return None

        return text or None

    except Exception as e:
        LOGGER.warning(
            "pymupdf4llm extraction failed for %s: %s", path.name, e
        )
        return None


def _describe_visual_with_gemini(image_bytes: bytes, context: str = "") -> Optional[str]:
    """
    Send a visual element to Gemini vision model for description.

    Args:
        image_bytes: PNG image bytes of the visual element
        context: Optional context about where this visual appears

    Returns:
        Description text or None if failed
    """
    try:
        from .config import AppConfig

        config = AppConfig.get()
        client = config.client

        # Create a prompt for technical manual visual extraction
        prompt = (
            "You are reading a technical manual. Describe this diagram, chart, image, or drawing "
            "in a few concise sentences. Preserve all labels, units, measurements, conditions, "
            "table-like information, and key visual relationships. Be specific and technical."
        )

        if context:
            prompt += f"\n\nContext: {context}"

        # Use Gemini Flash for fast, cost-effective vision analysis
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                prompt,
                {"mime_type": "image/png", "data": image_bytes}
            ]
        )

        if response and response.text:
            return response.text.strip()

        return None

    except Exception as exc:
        LOGGER.warning("Gemini vision API failed: %s", exc)
        return None


def _extract_visual_content_from_page(page, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract images and significant drawings from a PDF page.

    Args:
        page: PyMuPDF page object
        page_num: Page number for reference

    Returns:
        List of visual items with rect, type, and description
    """
    try:
        import fitz
    except ImportError:
        return []

    visual_items = []
    page_area = page.rect.width * page.rect.height
    min_area_threshold = page_area * 0.10  # 10% of page area
    min_dimension = 100  # minimum 100px in either dimension

    # 1. Extract images
    try:
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            try:
                xref = img_info[0]

                # Get image bounding box
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue

                rect = img_rects[0]  # Use first occurrence

                # Filter by size
                img_width = rect.width
                img_height = rect.height
                img_area = img_width * img_height

                if img_width < min_dimension or img_height < min_dimension:
                    continue

                if img_area < min_area_threshold:
                    continue

                # Render the image region to pixmap
                mat = fitz.Matrix(2, 2)  # 2x scale for better quality
                pix = page.get_pixmap(matrix=mat, clip=rect)

                # Convert to PNG bytes
                img_bytes = pix.tobytes("png")

                # Get description from Gemini
                context = f"Page {page_num + 1}, Image {img_index + 1}"
                description = _describe_visual_with_gemini(img_bytes, context)

                if description:
                    visual_items.append({
                        "kind": "visual",
                        "type": "image",
                        "rect": rect,
                        "y0": float(rect.y0),
                        "x0": float(rect.x0),
                        "description": description,
                        "context": context,
                    })
                    LOGGER.info("Extracted image on page %d: %dx%d px", page_num + 1, int(img_width), int(img_height))

            except Exception as exc:
                LOGGER.debug("Failed to extract image %d on page %d: %s", img_index, page_num + 1, exc)
                continue

    except Exception as exc:
        LOGGER.debug("Failed to get images from page %d: %s", page_num + 1, exc)

    # 2. Extract significant drawings/vector graphics
    try:
        drawings = page.get_drawings()
        if drawings:
            # Group drawings by proximity to identify significant graphic regions
            # For now, we'll render the entire page and let smaller drawings be filtered
            # A more sophisticated approach would cluster drawings by bounding box

            # Calculate bounding box of all drawings
            if len(drawings) > 5:  # Only process if there are enough drawings to be significant
                all_rects = []
                for drawing in drawings:
                    if "rect" in drawing:
                        all_rects.append(fitz.Rect(drawing["rect"]))

                if all_rects:
                    # Find the union of all drawing rects
                    combined_rect = all_rects[0]
                    for r in all_rects[1:]:
                        combined_rect |= r

                    # Filter by size
                    draw_width = combined_rect.width
                    draw_height = combined_rect.height
                    draw_area = draw_width * draw_height

                    if (draw_width >= min_dimension and draw_height >= min_dimension and
                        draw_area >= min_area_threshold):

                        # Render the drawing region
                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat, clip=combined_rect)
                        img_bytes = pix.tobytes("png")

                        context = f"Page {page_num + 1}, Drawing/Diagram"
                        description = _describe_visual_with_gemini(img_bytes, context)

                        if description:
                            visual_items.append({
                                "kind": "visual",
                                "type": "drawing",
                                "rect": combined_rect,
                                "y0": float(combined_rect.y0),
                                "x0": float(combined_rect.x0),
                                "description": description,
                                "context": context,
                            })
                            LOGGER.info("Extracted drawing on page %d: %dx%d px", page_num + 1, int(draw_width), int(draw_height))

    except Exception as exc:
        LOGGER.debug("Failed to extract drawings from page %d: %s", page_num + 1, exc)

    return visual_items


def _pymupdf_page_to_text(page, page_num: int = 0, extract_visuals: bool = False) -> str:
    """
    Extract text from a PyMuPDF page with layout awareness + table preservation + visual extraction.

    - Uses page.get_text("blocks") for normal text.
    - Uses page.find_tables() for real tables.
    - Optionally extracts images and drawings with AI descriptions.
    - Converts tables to Markdown.
    - Inserts tables and visual descriptions in correct reading order based on geometry.
    - Avoids duplicating table text from the block output.

    Args:
        page: PyMuPDF page object
        page_num: Page number for reference (0-indexed)
        extract_visuals: Whether to extract and describe visual content

    Returns:
        Extracted text with inline visual descriptions
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Fallback: should basically never happen in your env
        return page.get_text("text") or ""

    # 1. Get raw blocks
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return page.get_text("text") or ""

    # Normalize text blocks
    text_items = []
    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if not txt or not str(txt).strip():
            continue
        rect = fitz.Rect(x0, y0, x1, y1)
        text_items.append(
            {
                "kind": "text",
                "rect": rect,
                "y0": float(rect.y0),
                "x0": float(rect.x0),
                "text": str(txt).strip(),
            }
        )

    # 2. Detect tables (PyMuPDF >= table API)
    table_items = []
    table_rects = []

    try:
        tf = page.find_tables()
        tables = list(getattr(tf, "tables", [])) if tf else []
    except Exception:
        tables = []

    for idx, tbl in enumerate(tables, start=1):
        # 2a. Convert table to Markdown
        md = ""
        try:
            md = tbl.to_markdown(clean=False, fill_empty=True)
        except Exception:
            # Fallback if to_markdown not available / fails
            try:
                rows = tbl.extract()
                md_lines = []
                for r in rows:
                    # r is list of cell values
                    md_lines.append(" | ".join((c or "").strip() for c in r))
                md = "\n".join(md_lines)
            except Exception:
                continue  # skip broken table gracefully

        # 2b. Compute bounding box to place table in flow
        rect = None
        try:
            cells = getattr(tbl, "cells", None)
            if cells:
                x0 = min(c.x0 for c in cells)
                y0 = min(c.y0 for c in cells)
                x1 = max(c.x1 for c in cells)
                y1 = max(c.y1 for c in cells)
                rect = fitz.Rect(x0, y0, x1, y1)
            else:
                bbox = getattr(tbl, "bbox", None)
                if bbox is not None:
                    rect = fitz.Rect(*bbox)
        except Exception:
            rect = None

        if rect is None:
            # If we have no geometry, append after last text as a last resort
            y0 = max((ti["y0"] for ti in text_items), default=0.0) + 1.0
            rect = fitz.Rect(0, y0, page.rect.x1, y0 + 1.0)

        table_rects.append(rect)
        table_items.append(
            {
                "kind": "table",
                "rect": rect,
                "y0": float(rect.y0),
                "x0": float(rect.x0),
                # Wrapped in blank lines so it's clearly separated for the LLM
                "text": md.strip(),
            }
        )

    # 3. Remove text blocks that belong to tables (to avoid duplication)
    def is_inside_table(block_rect: "fitz.Rect") -> bool:
        if not table_rects:
            return False
        for tr in table_rects:
            inter = tr & block_rect
            if inter.is_empty:
                continue
            # If most of the block is inside a table area, drop it
            if inter.get_area() >= 0.6 * block_rect.get_area():
                return True
        return False

    filtered_items = []
    for item in text_items:
        if not is_inside_table(item["rect"]):
            filtered_items.append(item)

    # 4. Extract visual content (images and drawings) if enabled
    visual_items = []
    if extract_visuals:
        try:
            visual_items = _extract_visual_content_from_page(page, page_num)
            LOGGER.debug("Extracted %d visual items from page %d", len(visual_items), page_num + 1)
        except Exception as exc:
            LOGGER.warning("Visual extraction failed for page %d: %s", page_num + 1, exc)

    # 5. Merge text + tables + visuals in reading order
    all_items = filtered_items + table_items + visual_items
    if not all_items:
        return ""

    all_items.sort(key=lambda i: (round(i["y0"], 1), i["x0"]))

    # 6. Build final markdown-ish text with inline visual descriptions
    output_parts: List[str] = []
    for item in all_items:
        if item["kind"] == "text":
            output_parts.append(item["text"])
        elif item["kind"] == "table":
            # Tables already markdown; wrap with spacing for clarity.
            output_parts.append(item["text"])
        elif item["kind"] == "visual":
            # Format visual descriptions with clear markers
            visual_type = item.get("type", "visual").upper()
            context = item.get("context", "")
            description = item.get("description", "")
            visual_block = f"[{visual_type} - {context}]\n{description}\n[/{visual_type}]"
            output_parts.append(visual_block)

    return "\n\n".join(part.strip() for part in output_parts if part.strip())



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
    """Normalize text for LLMs without breaking basic Markdown structures."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Track fenced code blocks ``` ```
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(line.rstrip())
            continue

        if in_code_block:
            # Do not touch code fences content
            cleaned_lines.append(line.rstrip())
            continue

        # For markdown table rows, keep pipes & hyphens as-is, just trim edges
        if stripped.startswith("|") and stripped.endswith("|"):
            # Collapse internal tabs/spaces a bit but keep structure
            norm = re.sub(r"[ \t]+", " ", line)
            cleaned_lines.append(norm.rstrip())
            continue

        # Generic normalization for normal text
        line = re.sub(r"[ \t]+", " ", line)          # collapse spaces
        line = re.sub(r"([\-_=]{4,})", "----", line) # normalize long ruler-ish runs
        line = re.sub(r"(\.{4,})", "...", line)      # normalize dot spam
        line = re.sub(r"([,.;:!?])\1{2,}", r"\1", line)  # kill stupid !!!??? spam

        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines)

    # Collapse excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()

def read_doc_for_llm(path: Path, max_chars: Optional[int] = None) -> str:
    """Extract text from various document formats."""
    ext = path.suffix.lower()
    
    # Dispatch to format-specific handler
    handlers = {
        '.txt': _extract_txt,
        '.docx': _extract_docx,
        '.doc': _extract_legacy_doc,
        '.pdf': _extract_pdf,
        '.xlsx': _extract_excel,
        '.xls': _extract_excel,
    }
    
    handler = handlers.get(ext)
    if not handler:
        LOGGER.warning("Unsupported file type: %s", ext)
        return ""
    
    try:
        text = handler(path)
    except Exception as exc:
        LOGGER.error("Failed to extract %s: %s", path.name, exc)
        text = f"[READ_ERROR] {exc}"
    
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    
    return text


def _extract_txt(path: Path) -> str:
    """Extract from plain text files."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_docx(path: Path) -> str:
    """Extract from modern Word documents, preserving table positions."""
    from docx import Document as DocxDocument
    
    document = DocxDocument(str(path))
    parts: List[str] = []
    
    # Header tables and paragraphs
    for section in document.sections:
        header = section.header
        for table in header.tables:
            for row in table.rows:
                row_parts = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text and cell_text not in row_parts:
                        row_parts.append(cell_text)
                if row_parts:
                    parts.append(' | '.join(row_parts))
        
        for para in header.paragraphs:
            if para.text.strip():
                parts.append(para.text)
    
    # FIXED: Extract body elements in document order (paragraphs AND tables inline)
    for element in document.element.body:
        # Paragraph
        if element.tag.endswith('p'):
            for para in document.paragraphs:
                if para._element == element:
                    if para.text.strip():
                        parts.append(para.text)
                    break
        
        # Table (inline with paragraphs)
        elif element.tag.endswith('tbl'):
            for table in document.tables:
                if table._element == element:
                    table_text_parts = []
                    for row in table.rows:
                        row_text_parts = []
                        for cell in row.cells:
                            cell_text = cell.text.strip()
                            if cell_text:
                                row_text_parts.append(cell_text)
                        if row_text_parts:
                            table_text_parts.append(" | ".join(row_text_parts))
                    
                    if table_text_parts:
                        table_full_text = "\n".join(table_text_parts)
                        max_table_chars = 10000
                        if len(table_full_text) > max_table_chars:
                            parts.append(table_full_text[:max_table_chars])
                        else:
                            parts.append(table_full_text)
                    break
    
    return collapse_repeated_lines("\n".join(parts))


def _extract_legacy_doc(path: Path) -> str:
    """
    Extract from legacy .doc files by converting to PDF then using PDF extraction.
    
    IMPORTANT: Cleans up the temporary converted PDF after extraction.
    """
    LOGGER.info("Legacy .doc detected: %s - attempting local conversion", path.name)
    
    # Use a temp file in system temp dir, NOT in the library
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        converted_path = Path(tmp.name)
    
    try:
        # Try Spire.Doc first
        try:
            from spire.doc import Document, FileFormat
            doc = Document()
            doc.LoadFromFile(str(path))
            doc.SaveToFile(str(converted_path), FileFormat.PDF)
            doc.Close()
            LOGGER.info("Converted %s to PDF via Spire.Doc", path.name)
        except ImportError:
            # Fall back to Aspose
            try:
                import aspose.words as aw
                doc = aw.Document(str(path))
                doc.save(str(converted_path))
                LOGGER.info("Converted %s to PDF via Aspose.Words", path.name)
            except Exception as exc:
                LOGGER.warning("Aspose conversion failed for %s: %s", path.name, exc)
                raise  # Re-raise to trigger outer except
        
        if not converted_path.exists():
            raise FileNotFoundError("Conversion produced no output file")
        
        # Now extract from the converted PDF using the standard PDF handler
        text = _extract_pdf(converted_path)
        LOGGER.info("Extracted text from converted PDF: %s", path.name)
        return text
        
    finally:
        # CRITICAL: Always clean up the temporary PDF
        if converted_path.exists():
            try:
                converted_path.unlink()
                LOGGER.debug("Cleaned up temporary PDF: %s", converted_path.name)
            except Exception as exc:
                LOGGER.warning("Failed to delete temporary PDF %s: %s", converted_path.name, exc)


def _extract_pdf(path: Path) -> str:
    """
    Extract from PDF using markdown-first approach with optional visual extraction.

    Tries pymupdf4llm first (best for LLMs), falls back to custom parser.
    If visual extraction is enabled, processes images and drawings with AI descriptions.
    """

    # Check if visual extraction is enabled
    from .config import AppConfig
    try:
        config = AppConfig.get()
        extract_visuals = getattr(config, 'visual_extraction_enabled', False)
    except Exception:
        extract_visuals = False

    # Try high-level markdown extraction first
    #text = extract_with_pymupdf4llm(path)
    #if text:
    #    return text

    # Fallback to custom extraction
    try:
        import fitz

        with fitz.open(str(path)) as doc:
            page_texts: List[str] = []
            for page_num, page in enumerate(doc):
                page_text = _pymupdf_page_to_text(page, page_num=page_num, extract_visuals=extract_visuals)
                if page_text.strip():
                    page_texts.append(page_text)

            text = "\n\n".join(page_texts)

            if len(text.strip()) < 100 and len(doc) > 0:
                LOGGER.warning("PDF %s yielded minimal text - may be scanned", path.name)

            return text

    except ImportError:
        LOGGER.debug("PyMuPDF not available, using PyPDF2 for %s", path.name)
        return _extract_pdf_pypdf2(path)

    except Exception as exc:
        LOGGER.warning("PyMuPDF failed for %s: %s", path.name, exc)
        return _extract_pdf_pypdf2(path)


def _extract_pdf_pypdf2(path: Path) -> str:
    """Fallback PDF extraction using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages.append(extracted)
        return "\n".join(pages)
    except Exception:
        return ""


def _extract_excel(path: Path) -> str:
    """Extract from Excel files (both .xlsx and .xls)."""
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
    
    return "\n\n".join(chunks)


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
