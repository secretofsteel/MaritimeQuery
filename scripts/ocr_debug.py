# debug_pdf.py
from pathlib import Path
import json

import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.extraction import gemini_extract_record
from app.files import read_doc_for_llm, clean_text_for_llm
from app.config import AppConfig

# Initialize config
config = AppConfig.get()

# Point to your PDF
path = Path("data/temp/ECDIS Procedure.pdf")

print(f"Extracting: {path.name}")
print("="*60)

#full_text = clean_text_for_llm(read_doc_for_llm(path))

# Run extraction
result = gemini_extract_record(path)

print("\n" + "="*60)
print("EXTRACTION RESULTS:")
print("="*60)

print(f"Doc Type: {result.get('doc_type')}")
print(f"Title: {result.get('title')}")
print(f"Sections: {len(result.get('sections', []))}")

if result.get('sections'):
    print("\nSections:")
    for i, section in enumerate(result['sections'], 1):
        print(f"  {i}. {section.get('name', 'UNNAMED')}")

'''
if "The hardware/software failures have as a consequence" in full_text:
    idx = full_text.find("The hardware/software failures have as a consequence")
    if idx != -1:
        print(f"\nThe ENC cells are named using an 8-character identifier' (next 5000 chars):")
        print(full_text[idx:idx+5000])
'''