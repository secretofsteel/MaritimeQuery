"""
Standalone validation script for maritime RAG extraction quality.
Compares raw text against extracted content using n-gram matching.

Usage:
    python validate_extraction.py <filename> [options]
    python validate_extraction.py "Form C 002.pdf" --n-gram-size 4
"""

import pickle
import json
import re
from pathlib import Path
from typing import Dict, Set
import argparse
from app.files import read_doc_for_llm, clean_text_for_llm


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove formatting artifacts but keep content
    text = re.sub(r'[•◦▪\-–—…]', '', text)
    # Remove special chars that might vary
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()


def get_ngrams(text: str, n: int) -> Set[str]:
    """Extract n-grams from text."""
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def validate_extraction_ngram(raw_text: str, extracted_data: dict, n: int = 3) -> dict:
    """
    Validate extraction using n-gram matching to detect missing/hallucinated content.
    
    Args:
        raw_text: Original document text
        extracted_data: Gemini extraction result with 'sections'
        n: N-gram size (default 3 = trigrams)
    
    Returns:
        Validation report with coverage, hallucination rate, and issues
    """
    # Extract all section content
    extracted_content = " ".join(
        section.get("content", "") 
        for section in extracted_data.get("sections", [])
    )
    
    # Normalize both texts
    raw_norm = normalize(raw_text)
    extracted_norm = normalize(extracted_content)
    
    # Generate n-grams
    raw_ngrams = get_ngrams(raw_norm, n)
    extracted_ngrams = get_ngrams(extracted_norm, n)
    
    # Calculate metrics
    ngrams_in_both = raw_ngrams & extracted_ngrams
    missing_ngrams = raw_ngrams - extracted_ngrams
    hallucinated_ngrams = extracted_ngrams - raw_ngrams
    
    coverage = len(ngrams_in_both) / len(raw_ngrams) if raw_ngrams else 0
    hallucination_rate = len(hallucinated_ngrams) / len(extracted_ngrams) if extracted_ngrams else 0
    
    # Also track word-level stats for context
    raw_words = set(raw_norm.split())
    extracted_words = set(extracted_norm.split())
    word_coverage = len(raw_words & extracted_words) / len(raw_words) if raw_words else 0
    
    # Character count comparison
    length_ratio = len(extracted_norm) / len(raw_norm) if raw_norm else 0
    
    # Determine issues
    issues = []
    if coverage < 0.75:
        issues.append(f"Low n-gram coverage: {coverage:.1%} - significant content missing")
    if hallucination_rate > 0.20:
        issues.append(f"High hallucination rate: {hallucination_rate:.1%} - extracted content not in original")
    if length_ratio < 0.60:
        issues.append(f"Extracted text too short: {length_ratio:.1%} of original length")
    if length_ratio > 1.40:
        issues.append(f"Extracted text too long: {length_ratio:.1%} of original - possible duplication")
    
    return {
        'passed': coverage >= 0.75 and hallucination_rate <= 0.20,
        'ngram_coverage': coverage,
        'hallucination_rate': hallucination_rate,
        'word_coverage': word_coverage,
        'length_ratio': length_ratio,
        'missing_ngram_count': len(missing_ngrams),
        'hallucinated_ngram_count': len(hallucinated_ngrams),
        'sample_missing_ngrams': sorted(list(missing_ngrams))[:10],
        'sample_hallucinated_ngrams': sorted(list(hallucinated_ngrams))[:10],
        'issues': issues,
        'raw_char_count': len(raw_norm),
        'extracted_char_count': len(extracted_norm),
    }


import hashlib

def get_text_cache_path(file_path: Path, cache_dir: Path) -> Path:
    """
    Generate cache path matching files.py logic exactly.
    Uses filename + hash of (mtime + size) for cache key.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Original file not found: {file_path}")
    
    # Create hash of mtime + size for cache invalidation
    file_stat = file_path.stat()
    cache_key = f"{file_stat.st_mtime}_{file_stat.st_size}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    
    # Use actual filename + short hash
    base_name = file_path.stem
    safe_name = re.sub(r'[^\w\-]', '_', base_name)  # Replace special chars with underscore
    
    return cache_dir / "text_extracts" / f"{safe_name}_{cache_hash}.pkl"


def load_raw_text(file_path: Path, cache_dir: Path) -> str:
    """Load raw text from pickle cache."""
    return clean_text_for_llm(read_doc_for_llm(file_path))


def load_extracted_data(filename: str, cache_dir: Path) -> dict:
    """Load Gemini extraction from JSONL cache."""
    gemini_cache = cache_dir / "gemini_extract_cache.jsonl"
    
    if not gemini_cache.exists():
        raise FileNotFoundError(f"Gemini cache not found: {gemini_cache}")
    
    # Search for matching filename in JSONL
    with open(gemini_cache, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('filename') == filename:
                    return data
    
    raise ValueError(f"No extraction found for filename: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate extraction quality using n-gram matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_extraction.py "Form C 002.pdf"
  python validate_extraction.py "Ballast Water Management.docx" --n-gram-size 4
  python validate_extraction.py "Safety Manual.pdf" --threshold 0.80 --output report.json
        """
    )
    parser.add_argument('filename', type=Path,
                       help='Filename to validate (as stored in extraction cache)')
    parser.add_argument('--docs-dir', type=Path, default=Path('data/docs'),
                        help='Directory containing original documents')
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'),
                       help='Directory containing cache files (default: data/cache)')
    parser.add_argument('--n-gram-size', type=int, default=3,
                       help='N-gram size for comparison (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Minimum coverage threshold to pass (default: 0.75)')
    parser.add_argument('--output', type=Path,
                       help='Save validation report to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed samples of missing/hallucinated n-grams')
    
    args = parser.parse_args()
    
    print(f"Validating: {args.filename}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"N-gram size: {args.n_gram_size}, Threshold: {args.threshold:.0%}\n")
    
    try:
        # Load data
        raw_text = load_raw_text(args.filename, args.cache_dir)
        extracted_data = load_extracted_data(args.filename, args.cache_dir)
        
        print(f"Raw text: {len(raw_text):,} chars")
        print(f"Extracted sections: {len(extracted_data.get('sections', []))}\n")
        
        # Validate
        validation = validate_extraction_ngram(raw_text, extracted_data, n=args.n_gram_size)
        
        # Display results
        status = "✅ PASSED" if validation['passed'] else "❌ FAILED"
        print(f"{status}\n")
        print(f"N-gram coverage:     {validation['ngram_coverage']:.1%}")
        print(f"Hallucination rate:  {validation['hallucination_rate']:.1%}")
        print(f"Word coverage:       {validation['word_coverage']:.1%}")
        print(f"Length ratio:        {validation['length_ratio']:.1%}")
        print(f"Missing n-grams:     {validation['missing_ngram_count']:,}")
        print(f"Hallucinated n-grams: {validation['hallucinated_ngram_count']:,}")
        
        if validation['issues']:
            print(f"\nIssues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if args.verbose:
            if validation['sample_missing_ngrams']:
                print(f"\nSample missing n-grams:")
                for ngram in validation['sample_missing_ngrams'][:10]:
                    print(f"  - {ngram}")
            
            if validation['sample_hallucinated_ngrams']:
                print(f"\nSample hallucinated n-grams:")
                for ngram in validation['sample_hallucinated_ngrams'][:10]:
                    print(f"  - {ngram}")
        
        # Save report if requested
        if args.output:
            report = {
                'filename': args.filename,
                'validation': validation,
                'parameters': {
                    'n_gram_size': args.n_gram_size,
                    'threshold': args.threshold
                }
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed report saved to {args.output}")
        
        # Exit with appropriate code
        exit(0 if validation['passed'] else 1)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(2)
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(2)


if __name__ == '__main__':
    main()