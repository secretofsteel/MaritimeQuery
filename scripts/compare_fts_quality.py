"""
Script to compare FTS retrieval quality between SQLite baseline and PostgreSQL.

Loads baseline data from 'data/cache/fts_baseline.json' and runs the same
queries against the new PgFTSRetriever. Calculates overlap (Jaccard)
and Rank-Biased Overlap (RBO) or simple rank agreement.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Set

from llama_index.core.schema import QueryBundle

from app.config import AppConfig
from app.retrieval import PgFTSRetriever
from app.pg_database import init_pg_pool, close_pg_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASELINE_FILE = Path("data/cache/fts_baseline.json")


def calculate_jaccard(list1: List[str], list2: List[str]) -> float:
    """Calculate Jaccard similarity index between two lists of IDs."""
    set1 = set(list1)
    set2 = set(list2)
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union


def calculate_recall_at_k(baseline: List[str], current: List[str], k: int) -> float:
    """Calculate recall of baseline items in current items at K."""
    if not baseline:
        return 1.0
    
    baseline_set = set(baseline[:k])
    current_set = set(current[:k])
    
    found = len(baseline_set.intersection(current_set))
    return found / len(baseline_set)


def compare_results(baseline_results: List[Dict], pg_retriever: PgFTSRetriever):
    """Run comparisons for all queries."""
    
    total_jaccard = 0.0
    total_recall = 0.0
    count = 0
    
    print(f"{'Query':<40} | {'Jaccard':<10} | {'Recall@Baseline':<15} | {'Counts (Base/PG)':<15}")
    print("-" * 90)
    
    for entry in baseline_results:
        query_text = entry["query"]
        baseline_nodes = entry["fts_nodes"]  # List of {node_id, doc_id, score}
        
        # Extract baseline IDs
        baseline_ids = [n["node_id"] for n in baseline_nodes]
        
        if not baseline_ids:
            continue
            
        # Run query against PG
        pg_nodes = pg_retriever.retrieve(query_text)
        pg_ids = [n.node.node_id for n in pg_nodes]
        
        # Metrics
        jaccard = calculate_jaccard(baseline_ids, pg_ids)
        recall = calculate_recall_at_k(baseline_ids, pg_ids, len(baseline_ids))
        
        total_jaccard += jaccard
        total_recall += recall
        count += 1
        
        print(f"{query_text[:38]:<40} | {jaccard:.2f}       | {recall:.2f}            | {len(baseline_ids)} / {len(pg_ids)}")
        
        # Debug mismatches for low scores
        if jaccard < 0.5:
            missing = set(baseline_ids) - set(pg_ids)
            extra = set(pg_ids) - set(baseline_ids)
            LOGGER.debug(f"Query: {query_text}")
            LOGGER.debug(f"Missing from PG: {list(missing)[:5]}")
            LOGGER.debug(f"Extra in PG: {list(extra)[:5]}")

    if count > 0:
        avg_jaccard = total_jaccard / count
        avg_recall = total_recall / count
        print("-" * 90)
        print(f"AVERAGE JACCARD SIMILARITY: {avg_jaccard:.2f}")
        print(f"AVERAGE RECALL (vs Baseline): {avg_recall:.2f}")
        
        if avg_recall >= 0.8:
            print("\n[OK] SUCCESS: Retrieval quality is comparable to baseline.")
        else:
            print("\n[!] WARNING: Significant deviation from baseline detected.")
    else:
        print("\nNo valid baseline queries found.")


def main():
    if not BASELINE_FILE.exists():
        print(f"Error: Baseline file not found at {BASELINE_FILE}")
        return

    # Load baseline
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Transform to list format expected by compare_results
    baseline_data = []
    if "queries" in raw_data:
        for query, results in raw_data["queries"].items():
            baseline_data.append({
                "query": query,
                "fts_nodes": results.get("fts_only", [])
            })
    else:
        # Fallback if user somehow has old format (unlikely given we just made it)
        baseline_data = raw_data if isinstance(raw_data, list) else []

    print(f"Loaded {len(baseline_data)} queries from baseline.")

    # Init AppConfig (loads env vars) and PG connection
    config = AppConfig.get()
    init_pg_pool(config.database_url)
    
    try:
        # Initialize Retriever
        # The baseline script captured whatever top_k was configured (likely 10 or 20)
        top_k = 20
        if baseline_data:
             top_k = len(baseline_data[0]["fts_nodes"])

        retriever = PgFTSRetriever(
            tenant_id=None,  # Global search to match baseline
            similarity_top_k=max(top_k, 20) 
        )
        
        compare_results(baseline_data, retriever)

    finally:
        close_pg_pool()


if __name__ == "__main__":
    main()
