"""
Check Gemini extraction cache for duplicate entries.

Groups by filename stem (without extension) to catch cases where the same
document was processed in different formats (e.g., .docx and .pdf).
"""

import json
from pathlib import Path
from collections import defaultdict

# Load Gemini cache
cache_path = Path("C:/MADASS/data/cache/gemini_extract_cache.jsonl")

if not cache_path.exists():
    print(f"❌ Gemini cache not found at {cache_path}")
    exit(1)

print(f"Reading Gemini cache: {cache_path}")
print("="*80)

entries = []
with open(cache_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            entry = json.loads(line)
            entries.append(entry)
        except json.JSONDecodeError as e:
            print(f"⚠️  Line {line_num}: JSON decode error: {e}")

print(f"Loaded {len(entries)} cache entries")
print()

# Group by filename stem (without extension)
entries_by_stem = defaultdict(list)

for entry in entries:
    filename = entry.get('filename', '')
    if not filename:
        continue
    
    # Get stem (filename without extension)
    stem = Path(filename).stem
    
    entries_by_stem[stem].append({
        'filename': filename,
        'extension': Path(filename).suffix,
        'doc_type': entry.get('gemini', {}).get('doc_type', 'N/A'),
        'title': entry.get('gemini', {}).get('title', 'N/A'),
        'sections': len(entry.get('gemini', {}).get('sections', [])),
        'mtime': entry.get('mtime', 0),
        'size': entry.get('size', 0),
        'has_validation_error': 'validation_error' in entry
    })

# Find duplicates
duplicates = {stem: entries for stem, entries in entries_by_stem.items() if len(entries) > 1}

print("="*80)
print("DUPLICATE ANALYSIS")
print("="*80)

if not duplicates:
    print("\n✅ No duplicates found! Each filename stem appears only once.")
else:
    print(f"\n⚠️  Found {len(duplicates)} filename stems with multiple entries:")
    print()
    
    for stem, stem_entries in sorted(duplicates.items()):
        print(f"📄 {stem}")
        print(f"   {len(stem_entries)} entries:")
        
        for i, entry in enumerate(stem_entries, 1):
            print(f"   {i}. {entry['filename']}")
            print(f"      Extension: {entry['extension']}")
            print(f"      Type: {entry['doc_type']}")
            print(f"      Title: {entry['title'][:60]}...")
            print(f"      Sections: {entry['sections']}")
            print(f"      Size: {entry['size']:,} bytes")
            print(f"      Modified: {entry['mtime']}")
            
            if entry['has_validation_error']:
                print(f"      ⚠️  Has validation error")
            
            print()

print("="*80)
print("SUMMARY")
print("="*80)
print(f"Total entries: {len(entries)}")
print(f"Unique stems: {len(entries_by_stem)}")
print(f"Duplicate stems: {len(duplicates)}")

if duplicates:
    total_duplicate_entries = sum(len(entries) for entries in duplicates.values())
    print(f"Total duplicate entries: {total_duplicate_entries}")
    
    # Count by extension combination
    ext_combos = defaultdict(int)
    for entries in duplicates.values():
        exts = tuple(sorted(e['extension'] for e in entries))
        ext_combos[exts] += 1
    
    print()
    print("Duplicate patterns:")
    for exts, count in sorted(ext_combos.items(), key=lambda x: -x[1]):
        print(f"  {' + '.join(exts)}: {count} cases")
