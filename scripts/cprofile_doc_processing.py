"""Deep profiling using cProfile for function-level timing."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cProfile
import pstats
from io import StringIO

from app.config import AppConfig
from app.extraction import gemini_extract_record


def profile_with_cprofile():
    """Profile document processing with cProfile."""
    config = AppConfig.get()
    docs_path = config.paths.docs_path
    
    # Get a sample document
    doc_files = list(docs_path.rglob("*.docx")) + list(docs_path.rglob("*.pdf"))
    if not doc_files:
        print("No documents found")
        return
    
    sample_doc = doc_files[0]
    print(f"Profiling: {sample_doc.name}\n")
    
    # Profile the extraction
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the function
    result = gemini_extract_record(sample_doc)
    
    profiler.disable()
    
    # Print results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions by cumulative time
    
    print(s.getvalue())
    
    print("\n" + "="*70)
    print("TIME-CONSUMING FUNCTIONS")
    print("="*70)
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(20)  # Top 20 by total time
    print(s.getvalue())


if __name__ == "__main__":
    profile_with_cprofile()