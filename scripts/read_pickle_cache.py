#!/usr/bin/env python3
"""Read a pickled text extraction cache file."""

import pickle
import sys
from pathlib import Path


def read_pickle_cache(cache_file: str, output_file: str = None):
    """
    Read a pickled text cache and optionally save to text file.
    
    Args:
        cache_file: Path to .pkl cache file
        output_file: Optional output text file path
    """
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        print(f"Cache file not found: {cache_file}")
        return
    
    try:
        with open(cache_path, 'rb') as f:
            text = pickle.load(f)
        
        print(f"Loaded cache: {len(text):,} chars")
        
        if output_file:
            # Save to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ Saved to: {output_file}")
        else:
            # Print to console (first 2000 chars)
            print("\n" + "="*80)
            print("CONTENT PREVIEW (first 2000 chars):")
            print("="*80)
            print(text[:2000])
            if len(text) > 2000:
                print(f"\n... ({len(text) - 2000:,} more chars)")
            print("\n" + "="*80)
            print(f"Total length: {len(text):,} characters")
            
    except Exception as exc:
        print(f"Error reading cache: {exc}")


def list_cache_files(cache_dir: str = "data/cache/text_extracts"):
    """List all cached text files with better formatting."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    pkl_files = list(cache_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No cache files found in {cache_dir}")
        return
    
    print(f"\n{'='*100}")
    print(f"CACHED TEXT EXTRACTIONS: {len(pkl_files)} files")
    print(f"{'='*100}\n")
    print(f"{'#':<4} {'Filename':<60} {'Size':<12} {'Chars':<12}")
    print(f"{'-'*100}")
    
    for i, pkl_file in enumerate(sorted(pkl_files), 1):
        try:
            with open(pkl_file, 'rb') as f:
                text = pickle.load(f)
            size_kb = pkl_file.stat().st_size / 1024
            
            # Extract original filename from cache filename (before the hash)
            name_parts = pkl_file.stem.rsplit('_', 1)
            display_name = name_parts[0].replace('_', ' ')  # Remove underscores for readability
            
            print(f"{i:<4} {display_name:<60} {size_kb:>8.1f} KB {len(text):>10,} chars")
        except Exception as exc:
            print(f"{i:<4} {pkl_file.name:<60} ERROR: {exc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python read_pickle_cache.py list                           # List all cached files")
        print("  python read_pickle_cache.py <cache_file.pkl>               # Show preview")
        print("  python read_pickle_cache.py <cache_file.pkl> output.txt    # Save to text file")
        print("\nExamples:")
        print("  python read_pickle_cache.py list")
        print("  python read_pickle_cache.py data/cache/text_extracts/text_extract_abc123.pkl")
        print("  python read_pickle_cache.py data/cache/text_extracts/text_extract_abc123.pkl chapter7_text.txt")
        sys.exit(1)
    
    if sys.argv[1] == "list":
        cache_dir = sys.argv[2] if len(sys.argv) > 2 else "data/cache/text_extracts"
        list_cache_files(cache_dir)
    else:
        cache_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        read_pickle_cache(cache_file, output_file)