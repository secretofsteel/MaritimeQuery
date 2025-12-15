#!/usr/bin/env python3
"""Debug script to inspect Gemini extraction cache entries and save to file."""

import json
from pathlib import Path
import sys

def inspect_cache_entry(cache_path: Path, filename: str, output_file: str = None):
    """Load and display sections from a specific cache entry."""
    
    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        return
    
    # Determine output destination
    if output_file:
        output = open(output_file, 'w', encoding='utf-8')
        print(f"Writing output to: {output_file}")
    else:
        output = sys.stdout
    
    try:
        # Load cache
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # Check if filename matches at top level
                    if record.get('filename') == filename:
                        output.write(f"\n{'='*80}\n")
                        output.write(f"FOUND: {filename}\n")
                        output.write(f"{'='*80}\n")
                        
                        # The actual extraction is nested under 'gemini' key!
                        extraction = record.get('gemini', {})
                        
                        output.write(f"Doc Type: {extraction.get('doc_type')}\n")
                        output.write(f"Title: {extraction.get('title')}\n")
                        output.write(f"Plaintext Extraction: {extraction.get('plaintext_extraction', False)}\n")
                        output.write(f"Multi-pass: {extraction.get('multi_pass_extraction', False)}\n")
                        output.write(f"Num Passes: {extraction.get('num_passes', 'N/A')}\n")
                        
                        if extraction.get('parse_error'):
                            output.write(f"\n⚠️  PARSE ERROR: {extraction['parse_error']}\n")
                        
                        if extraction.get('extraction_warning'):
                            output.write(f"\n⚠️  WARNING: {extraction['extraction_warning']}\n")
                        
                        sections = extraction.get('sections', [])
                        output.write(f"\nTotal sections: {len(sections)}\n")
                        
                        # List all section names
                        output.write(f"\n{'='*80}\n")
                        output.write("SECTION NAMES:\n")
                        output.write(f"{'='*80}\n")
                        for i, section in enumerate(sections, 1):
                            name = section.get('name', '???')
                            content_len = len(section.get('content', ''))
                            output.write(f"{i:4d}. {name} ({content_len} chars)\n")
                        
                        # Show content for each section
                        output.write(f"\n{'='*80}\n")
                        output.write("SECTION CONTENTS:\n")
                        output.write(f"{'='*80}\n")
                        for i, section in enumerate(sections, 1):
                            name = section.get('name', '???')
                            content = section.get('content', '')
                            
                            output.write(f"\n--- SECTION {i}: {name} ---\n")
                            output.write(f"Content length: {len(content)} chars\n")
                            
                            # Show first 500 chars
                            if content:
                                preview = content[:2500]
                                output.write(f"Preview:\n{preview}\n")
                                if len(content) > 2500:
                                    output.write(f"\n... ({len(content) - 2500} more chars)\n")
                            else:
                                output.write("⚠️  EMPTY CONTENT!\n")
                            output.write("\n")
                        
                        if output_file:
                            print(f"✅ Output written to: {output_file}")
                        return
                        
                except json.JSONDecodeError:
                    continue
        
        output.write(f"File not found in cache: {filename}\n")
        
    finally:
        if output_file:
            output.close()


def list_all_cache_entries(cache_path: Path, output_file: str = None):
    """List all filenames in cache."""
    
    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        return
    
    # Determine output destination
    if output_file:
        output = open(output_file, 'w', encoding='utf-8')
        print(f"Writing output to: {output_file}")
    else:
        output = sys.stdout
    
    try:
        output.write(f"\n{'='*80}\n")
        output.write("ALL CACHED FILES:\n")
        output.write(f"{'='*80}\n")
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    filename = record.get('filename', 'UNKNOWN')
                    
                    # Extract from nested 'gemini' key
                    extraction = record.get('gemini', {})
                    
                    num_sections = len(extraction.get('sections', []))
                    plaintext = extraction.get('plaintext_extraction', False)
                    multipass = extraction.get('multi_pass_extraction', False)
                    error = '⚠️ ERROR' if extraction.get('parse_error') else '✓'
                    
                    mode = 'PT' if plaintext else 'JS'
                    mp = 'MP' if multipass else 'SP'
                    
                    output.write(f"{i:3d}. [{error}] [{mode}] [{mp}] {filename} ({num_sections} sections)\n")
                except json.JSONDecodeError:
                    output.write(f"{i:3d}. [INVALID JSON]\n")
        
        if output_file:
            print(f"✅ Output written to: {output_file}")
            
    finally:
        if output_file:
            output.close()


if __name__ == "__main__":
    # Default cache path - adjust if needed
    cache_path = Path("data/cache/gemini_extract_cache.jsonl")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            # List all entries
            output_file = sys.argv[2] if len(sys.argv) > 2 else None
            list_all_cache_entries(cache_path, output_file)
        else:
            # Inspect specific file
            filename = sys.argv[1]
            output_file = sys.argv[2] if len(sys.argv) > 2 else None
            inspect_cache_entry(cache_path, filename, output_file)
    else:
        print("Usage:")
        print("  python debug_cache.py list [output.txt]              # List all cached files")
        print("  python debug_cache.py <filename> [output.txt]        # Inspect specific file")
        print("\nExamples:")
        print("  python debug_cache.py list cache_list.txt")
        print("  python debug_cache.py 'Chapter 7.2.docx' chapter72_debug.txt")