"""Profile document processing pipeline with granular timing."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Dict, List, Any
import statistics

from app.config import AppConfig
from app.extraction import gemini_extract_record
from app.files import read_doc_for_llm
from llama_index.core import Settings as LlamaSettings


class ProcessingProfiler:
    """Track timing for each stage of document processing."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {
            "file_read": [],
            "gemini_extract": [],
            "total": [],
        }
    
    def profile_single_document(self, doc_path: Path) -> Dict[str, float]:
        """Profile all stages for a single document."""
        config = AppConfig.get()
        
        timings = {}
        total_start = time.perf_counter()
        
        # 1. File Read
        start = time.perf_counter()
        text = read_doc_for_llm(doc_path)
        timings["file_read"] = time.perf_counter() - start
        
        # 2. Gemini Extraction (EXPECTED BOTTLENECK)
        start = time.perf_counter()
        extracted_data = gemini_extract_record(doc_path)
        timings["gemini_extract"] = time.perf_counter() - start
        
        # Total
        timings["total"] = time.perf_counter() - total_start
        
        return timings
    
    def profile_batch(self, doc_paths: List[Path], sample_size: int = 10) -> Dict[str, Any]:
        """Profile a batch of documents and return statistics."""
        print(f"\nProfileling {min(sample_size, len(doc_paths))} documents...")
        print("=" * 70)
        
        for i, doc_path in enumerate(doc_paths[:sample_size], 1):
            print(f"  [{i}/{min(sample_size, len(doc_paths))}] {doc_path.name[:40]:<40}", end=" ")
            try:
                timings = self.profile_single_document(doc_path)
                for key, value in timings.items():
                    self.timings[key].append(value)
                print(f"✓ {timings['total']:>6.2f}s")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate statistical report from collected timings."""
        report = {}
        
        for stage, times in self.timings.items():
            if not times:
                continue
            
            report[stage] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
                "total": sum(times),
                "count": len(times),
            }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Pretty-print the profiling report."""
        print("\n" + "="*70)
        print("DOCUMENT PROCESSING PROFILING REPORT")
        print("="*70)
        
        # Calculate percentages
        total_time = report.get("total", {}).get("total", 0)
        
        for stage in ["file_read", "gemini_extract", "total"]:
            if stage not in report:
                continue
            
            stats = report[stage]
            stage_total = stats["total"]
            percentage = (stage_total / total_time * 100) if stage != "total" and total_time > 0 else 100
            
            print(f"\n{stage.upper().replace('_', ' ')}")
            print(f"  Total Time:  {stage_total:>7.2f}s  ({percentage:>5.1f}% of pipeline)")
            print(f"  Mean:        {stats['mean']:>7.3f}s")
            print(f"  Median:      {stats['median']:>7.3f}s")
            print(f"  Min/Max:     {stats['min']:>7.3f}s / {stats['max']:.3f}s")
            print(f"  Std Dev:     {stats['stdev']:>7.3f}s")
            print(f"  Count:       {stats['count']:>7} documents")
        
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        
        # Identify bottlenecks (stages taking >20% of total time)
        bottlenecks = []
        for stage, stats in report.items():
            if stage == "total":
                continue
            percentage = (stats["total"] / total_time * 100) if total_time > 0 else 0
            if percentage > 20:
                bottlenecks.append((stage, percentage, stats["mean"]))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        if bottlenecks:
            print("\nStages consuming >20% of processing time:")
            for stage, pct, mean_time in bottlenecks:
                print(f"  • {stage.replace('_', ' ').title()}: {pct:.1f}% (avg {mean_time:.2f}s per doc)")
        else:
            print("\nNo major bottlenecks detected (processing is well-balanced)")
        
        print("\n" + "="*70)
        print("PARALLELIZATION POTENTIAL")
        print("="*70)
        
        # Calculate potential speedup with 10 workers
        gemini_time = report.get("gemini_extract", {}).get("total", 0)
        io_time = report.get("file_read", {}).get("total", 0)
        
        print(f"\nGemini API (parallelizable):     {gemini_time:>7.2f}s ({gemini_time/total_time*100:>5.1f}%)")
        print(f"File I/O (fast):                 {io_time:>7.2f}s ({io_time/total_time*100:>5.1f}%)")
        
        # Theoretical speedup with 10 workers (Amdahl's Law approximation)
        parallel_portion = gemini_time / total_time if total_time > 0 else 0
        workers = 10
        theoretical_speedup = 1 / ((1 - parallel_portion) + (parallel_portion / workers)) if parallel_portion > 0 else 1
        
        doc_count = report.get("total", {}).get("count", 0)
        
        print(f"\n📊 For {doc_count} documents:")
        print(f"  Sequential time:    {total_time:>7.2f}s ({total_time/60:.1f} minutes)")
        print(f"  Parallel time:      {total_time / theoretical_speedup:>7.2f}s ({total_time/theoretical_speedup/60:.1f} minutes)")
        print(f"  Speedup factor:     {theoretical_speedup:>7.1f}x")
        
        print(f"\n📈 Extrapolated for 100 documents:")
        mean_per_doc = report.get("total", {}).get("mean", 0)
        seq_100 = mean_per_doc * 100
        par_100 = seq_100 / theoretical_speedup
        print(f"  Sequential:         {seq_100:>7.1f}s ({seq_100/60:.1f} minutes)")
        print(f"  Parallel (10 workers): {par_100:>7.1f}s ({par_100/60:.1f} minutes)")
        print(f"  Time saved:         {seq_100 - par_100:>7.1f}s ({(seq_100-par_100)/60:.1f} minutes)")
        
        print("\n" + "="*70)


def main():
    """Run the profiler on sample documents."""
    print("="*70)
    print("MARITIME RAG - DOCUMENT PROCESSING PROFILER")
    print("="*70)
    
    config = AppConfig.get()
    docs_path = config.paths.docs_path
    
    # Get all documents
    doc_files = (
        list(docs_path.rglob("*.docx")) + 
        list(docs_path.rglob("*.pdf")) + 
        list(docs_path.rglob("*.xlsx"))
    )
    
    if not doc_files:
        print(f"\n❌ No documents found in {docs_path}")
        print("\nPlease add documents to your docs directory:")
        print(f"  {docs_path}")
        return
    
    print(f"\n✓ Found {len(doc_files)} documents in {docs_path}")
    
    # Ask user how many to profile
    print("\nHow many documents to profile? (default: 10)")
    try:
        user_input = input("Enter number (or press Enter for 10): ").strip()
        sample_size = int(user_input) if user_input else 10
        sample_size = max(1, min(sample_size, len(doc_files)))  # Clamp between 1 and total
    except (ValueError, KeyboardInterrupt):
        sample_size = 10
    
    # Profile a sample
    profiler = ProcessingProfiler()
    report = profiler.profile_batch(doc_files, sample_size=sample_size)
    profiler.print_report(report)
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    gemini_pct = report.get("gemini_extract", {}).get("total", 0) / report.get("total", {}).get("total", 1) * 100
    
    if gemini_pct > 60:
        print("\n✅ HIGH IMPACT: Gemini extraction dominates processing time.")
        print("   → Implementing parallel processing will yield ~10x speedup")
        print("   → Priority: HIGH")
    elif gemini_pct > 40:
        print("\n⚠️  MODERATE IMPACT: Gemini extraction is significant but not dominant.")
        print("   → Parallel processing will yield ~3-5x speedup")
        print("   → Priority: MEDIUM")
    else:
        print("\n❓ LOW IMPACT: Gemini extraction is not the bottleneck.")
        print("   → Investigate other stages before parallelizing")
        print("   → Priority: LOW")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()