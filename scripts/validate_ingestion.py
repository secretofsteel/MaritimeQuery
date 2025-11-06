# scripts/validate_ingestion.py
import time
from pathlib import Path
from typing import List, Tuple

from app.files import read_doc_for_llm
from app.extraction import gemini_extract_record

SAMPLES_DIR = Path("samples")   # drop a few nasty PDFs/DOCX here
MAX_PREVIEW = 12000

def check_nonempty_text(p: Path) -> Tuple[bool, int]:
    t0 = time.time()
    txt = read_doc_for_llm(p, max_chars=MAX_PREVIEW)
    dt = time.time() - t0
    ok = bool(txt.strip())
    return ok, int(dt * 1000)

def check_gemini_json(p: Path) -> Tuple[bool, int]:
    t0 = time.time()
    rec = gemini_extract_record(p)
    dt = time.time() - t0
    # minimal assertions
    has_title = bool(rec.get("Title"))
    has_sections = isinstance(rec.get("Sections"), list)
    return has_title and has_sections, int(dt * 1000)

def main():
    pdfs = list(SAMPLES_DIR.glob("**/*.pdf"))
    docs = list(SAMPLES_DIR.glob("**/*.docx")) + list(SAMPLES_DIR.glob("**/*.doc"))
    files = pdfs + docs
    if not files:
        print("No sample files found under ./samples")
        return
    failed_text: List[Path] = []
    failed_json: List[Path] = []
    for p in files:
        ok_text, ms = check_nonempty_text(p)
        print(f"[TEXT] {p.name}: {'OK' if ok_text else 'FAIL'} in {ms}ms")
        if not ok_text:
            failed_text.append(p)
            continue
        ok_json, ms2 = check_gemini_json(p)
        print(f"[JSON] {p.name}: {'OK' if ok_json else 'FAIL'} in {ms2}ms")
        if not ok_json:
            failed_json.append(p)
    if failed_text:
        print("\nFiles with EMPTY extraction:")
        for p in failed_text: print(" -", p)
    if failed_json:
        print("\nFiles with BAD JSON:")
        for p in failed_json: print(" -", p)

if __name__ == "__main__":
    main()
