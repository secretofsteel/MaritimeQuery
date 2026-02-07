#!/usr/bin/env python3
"""Debug script to test structure extraction in isolation."""

import json
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import AppConfig
from app.files import read_doc_for_llm, clean_text_for_llm
from app.extraction import _gemini_get_structure


def test_structure_extraction(file_path: str, output_file: str = None, preview_length: int = None):
    """
    Test structure extraction on a specific file and output results.
    
    Args:
        file_path: Path to document to test
        output_file: Optional output file path (default: prints to console)
        preview_length: Optional - limit input text length (chars)
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"File not found: {file_path}")
        return
    
    # Determine output
    if output_file:
        output = open(output_file, 'w', encoding='utf-8')
        print(f"Writing output to: {output_file}")
    else:
        output = sys.stdout
    
    try:
        output.write(f"\n{'='*80}\n")
        output.write(f"STRUCTURE EXTRACTION TEST\n")
        output.write(f"{'='*80}\n")
        output.write(f"File: {path.name}\n")
        output.write(f"Size: {path.stat().st_size:,} bytes\n")
        output.write(f"\n")
        
        # Initialize config
        config = AppConfig.get()
        output.write(f"Plaintext mode: {config.use_plaintext_extraction}\n")
        output.write(f"\n")
        
        # Read document
        output.write("Reading document...\n")
        full_text = clean_text_for_llm(read_doc_for_llm(path))
        char_count = len(full_text)
        output.write(f"Document text length: {char_count:,} chars\n")
        
        # Optionally limit preview
        if preview_length:
            text_for_structure = full_text[:preview_length]
            output.write(f"LIMITING structure extraction to first {preview_length:,} chars\n")
        else:
            text_for_structure = full_text
        
        output.write(f"\n{'='*80}\n")
        output.write(f"TEXT PREVIEW (first 1000 chars):\n")
        output.write(f"{'='*80}\n")
        output.write(full_text[:1000])
        output.write("\n...\n\n")
        
        # Call structure extraction
        output.write(f"{'='*80}\n")
        output.write(f"CALLING _gemini_get_structure()...\n")
        output.write(f"{'='*80}\n")
        
        structure = _gemini_get_structure(
            filename=path.name,
            text_preview=text_for_structure,
            max_retries=0  # No retries for debugging
        )
        
        # Display results
        output.write(f"\n{'='*80}\n")
        output.write(f"STRUCTURE EXTRACTION RESULT\n")
        output.write(f"{'='*80}\n")
        
        # Basic metadata
        output.write(f"\nFilename: {structure.get('filename')}\n")
        output.write(f"Doc Type: {structure.get('doc_type')}\n")
        output.write(f"Title: {structure.get('title')}\n")
        
        if structure.get('category'):
            output.write(f"Category: {structure.get('category')}\n")
        if structure.get('form_number'):
            output.write(f"Form Number: {structure.get('form_number')}\n")
        if structure.get('normalized_topic'):
            output.write(f"Normalized Topic: {structure.get('normalized_topic')}\n")
        
        # Parse error?
        if structure.get('parse_error'):
            output.write(f"\n⚠️  PARSE ERROR: {structure['parse_error']}\n")
        
        # Section names
        section_names = structure.get('section_names', [])
        output.write(f"\n{'='*80}\n")
        output.write(f"SECTION NAMES: {len(section_names)} total\n")
        output.write(f"{'='*80}\n")
        
        if section_names:
            for i, name in enumerate(section_names, 1):
                output.write(f"{i:4d}. {name}\n")
        else:
            output.write("(No section names extracted)\n")
        
        # References
        refs = structure.get('references', {})
        output.write(f"\n{'='*80}\n")
        output.write(f"REFERENCES\n")
        output.write(f"{'='*80}\n")
        
        if refs:
            for ref_type, ref_list in refs.items():
                if ref_list:
                    output.write(f"\n{ref_type.upper()}: {len(ref_list)} items\n")
                    
                    # Check for duplicates
                    unique_refs = list(dict.fromkeys(ref_list))
                    if len(unique_refs) != len(ref_list):
                        output.write(f"⚠️  DUPLICATES DETECTED: {len(ref_list)} total, {len(unique_refs)} unique\n")
                    
                    # Show first 20 items
                    for i, ref in enumerate(ref_list[:20], 1):
                        output.write(f"  {i:3d}. {ref}\n")
                    
                    if len(ref_list) > 20:
                        output.write(f"  ... ({len(ref_list) - 20} more)\n")
                    
                    # Show last 5 if there are many
                    if len(ref_list) > 25:
                        output.write(f"\n  LAST 5:\n")
                        for i, ref in enumerate(ref_list[-5:], len(ref_list)-4):
                            output.write(f"  {i:3d}. {ref}\n")
        else:
            output.write("(No references extracted)\n")
        
        # Hierarchy (if present)
        hierarchy = structure.get('hierarchy', [])
        if hierarchy:
            output.write(f"\n{'='*80}\n")
            output.write(f"HIERARCHY: {len(hierarchy)} levels\n")
            output.write(f"{'='*80}\n")
            for i, level in enumerate(hierarchy, 1):
                output.write(f"{i}. {level}\n")
        
        # Summary
        output.write(f"\n{'='*80}\n")
        output.write(f"SUMMARY\n")
        output.write(f"{'='*80}\n")
        output.write(f"Sections extracted: {len(section_names)}\n")
        output.write(f"Total references: {sum(len(v) for v in refs.values() if isinstance(v, list))}\n")
        
        # Check for spam
        total_refs = sum(len(v) for v in refs.values() if isinstance(v, list))
        if total_refs > 200:
            output.write(f"⚠️  WARNING: {total_refs} total references - possible spam!\n")
        
        for ref_type, ref_list in refs.items():
            if isinstance(ref_list, list) and len(ref_list) > 50:
                unique_count = len(list(dict.fromkeys(ref_list)))
                output.write(f"⚠️  WARNING: {ref_type} has {len(ref_list)} items ({unique_count} unique)\n")
        
        if output_file:
            print(f"✅ Output written to: {output_file}")
        
    except Exception as exc:
        output.write(f"\n❌ ERROR: {exc}\n")
        import traceback
        traceback.print_exc(file=output)
        
    finally:
        if output_file:
            output.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_structure.py <file_path> [output.txt] [preview_length]")
        print("\nExamples:")
        print("  python debug_structure.py 'Chapter 7.2.pdf' structure_debug.txt")
        print("  python debug_structure.py 'Chapter 7.2.pdf' structure_debug.txt 100000  # Limit to 100k chars")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    preview_length = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    test_structure_extraction(file_path, output_file, preview_length)