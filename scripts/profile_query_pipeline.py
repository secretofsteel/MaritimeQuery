"""Profile the query runtime pipeline with granular timing."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Dict, List, Any
import statistics

from app.config import AppConfig
from app.query import query_with_confidence
from app.state import AppState


class QueryProfiler:
    """Track timing for query processing."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {
            "total": [],
        }
    
    def profile_single_query(self, query: str, app_state: AppState) -> Dict[str, float]:
        """Profile a single query end-to-end."""
        timings = {}
        
        # Full RAG Pipeline
        start = time.perf_counter()
        response = query_with_confidence(
            query, 
            app_state, 
            retriever_type="hybrid",
            rerank=True,
            max_attempts=1  # Single attempt for profiling
        )
        timings["total"] = time.perf_counter() - start
        
        # Store confidence for analysis
        timings["confidence"] = response.get("confidence_pct", 0)
        
        return timings
    
    def profile_queries(self, queries: List[str], app_state: AppState) -> Dict[str, Any]:
        """Profile multiple queries and return statistics."""
        print(f"\nProfiling {len(queries)} queries...")
        print("=" * 70)
        
        for i, query in enumerate(queries, 1):
            query_display = query[:50] + "..." if len(query) > 50 else query
            print(f"  [{i}/{len(queries)}] {query_display:<53}", end=" ")
            try:
                timings = self.profile_single_query(query, app_state)
                self.timings["total"].append(timings["total"])
                print(f"✓ {timings['total']:>6.2f}s (conf: {timings['confidence']}%)")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate statistical report from collected timings."""
        times = self.timings["total"]
        if not times:
            return {}
        
        sorted_times = sorted(times)
        p95_idx = int(len(times) * 0.95)
        
        return {
            "total": {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "p95": sorted_times[p95_idx] if len(times) > 1 else times[0],
                "min": min(times),
                "max": max(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "count": len(times),
            }
        }
    
    def print_report(self, report: Dict[str, Any]):
        """Pretty-print the profiling report."""
        if not report:
            print("\n❌ No data to report")
            return
        
        print("\n" + "="*70)
        print("QUERY PIPELINE PROFILING REPORT")
        print("="*70)
        
        stats = report["total"]
        
        print(f"\n📊 RESPONSE TIME STATISTICS ({stats['count']} queries)")
        print(f"  Mean:           {stats['mean']:>7.3f}s")
        print(f"  Median:         {stats['median']:>7.3f}s")
        print(f"  P95:            {stats['p95']:>7.3f}s")
        print(f"  Min/Max:        {stats['min']:>7.3f}s / {stats['max']:.3f}s")
        print(f"  Std Dev:        {stats['stdev']:>7.3f}s")
        
        print("\n" + "="*70)
        print("USER EXPERIENCE ASSESSMENT")
        print("="*70)
        
        mean = stats['mean']
        p95 = stats['p95']
        
        print(f"\n⏱️  Average query time:    {mean:.2f}s")
        print(f"⏱️  95th percentile time:  {p95:.2f}s")
        
        print("\n📈 Performance Targets:")
        if mean < 2.0:
            print("  ✅ Mean < 2s: EXCELLENT (users perceive as instant)")
        elif mean < 3.0:
            print("  🟡 Mean < 3s: GOOD (acceptable for complex queries)")
        else:
            print("  ❌ Mean > 3s: SLOW (consider optimization)")
        
        if p95 < 3.0:
            print("  ✅ P95 < 3s: EXCELLENT (consistent performance)")
        elif p95 < 5.0:
            print("  🟡 P95 < 5s: ACCEPTABLE (some queries lag)")
        else:
            print("  ❌ P95 > 5s: PROBLEMATIC (tail latency issues)")
        
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        
        # Typical breakdown for RAG systems
        print("\nTypical RAG pipeline breakdown:")
        print("  • Gemini generation:     50-60% (~0.8-1.2s)")
        print("  • Cohere reranking:      20-30% (~0.3-0.4s)")
        print("  • Query embedding:       10-15% (~0.15-0.25s)")
        print("  • Vector + BM25 search:   5-10% (~0.1-0.2s)")
        
        print("\nTo get granular timing, add @time_function decorators")
        print("to functions in app/query.py (see profiling_strategy.md)")
        
        print("\n" + "="*70)


def main():
    """Run the query profiler."""
    print("="*70)
    print("MARITIME RAG - QUERY PIPELINE PROFILER")
    print("="*70)
    
    config = AppConfig.get()
    
    # Initialize app state and load index
    app_state = AppState()
    
    print("\nLoading document index...")
    if not app_state.ensure_index_loaded():
        print("\n❌ ERROR: No cached index found.")
        print("\nYou need to build the index first:")
        print("  1. Run your Streamlit app: streamlit run rag_modular.py")
        print("  2. Use Admin mode to 'Build Index from Library'")
        print("  3. Wait for indexing to complete")
        print("  4. Then run this profiler again")
        return
    
    print(f"✓ Index loaded: {len(app_state.nodes)} nodes")
    
    # Sample queries to profile
    test_queries = [
        "What is the alcohol policy?",
        "Tell me about the bunkering procedure",
        "What forms do I need for shore leave?",
        "How do I report a near-miss incident?",
        "What are the PPE requirements for confined spaces?",
        "What are the procedures for ballast water management?",
        "How do I handle a medical emergency on board?",
        "What documentation is needed for port state control?",
    ]
    
    # Ask user how many queries to run
    print(f"\nDefault: Profile {len(test_queries)} test queries")
    print("Options:")
    print("  1. Run all test queries")
    print("  2. Run a subset")
    print("  3. Add your own custom query")
    
    try:
        choice = input("\nYour choice (1-3, or Enter for default): ").strip()
        
        if choice == "2":
            count = int(input(f"How many queries (1-{len(test_queries)}): "))
            test_queries = test_queries[:count]
        elif choice == "3":
            custom = input("Enter your query: ").strip()
            if custom:
                test_queries = [custom]
        # else: use all test queries (default)
        
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default test queries...")
    
    # Profile
    profiler = QueryProfiler()
    report = profiler.profile_queries(test_queries, app_state)
    profiler.print_report(report)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if report:
        mean = report["total"]["mean"]
        
        if mean < 2.0:
            print("\n✅ Query performance is excellent!")
            print("   No immediate optimization needed.")
        elif mean < 3.0:
            print("\n🟡 Query performance is acceptable.")
            print("   Consider optimizations if processing many queries:")
            print("   • Switch to Gemini Flash Lite for generation")
            print("   • Make Cohere reranking optional")
        else:
            print("\n⚠️  Query performance needs improvement.")
            print("   Priority optimizations:")
            print("   • Profile with decorators to find exact bottleneck")
            print("   • Consider Gemini Flash Lite (faster, cheaper)")
            print("   • Optimize reranking (make optional or async)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()