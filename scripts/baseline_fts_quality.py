#!/usr/bin/env python3
"""Capture FTS retrieval baseline BEFORE PostgreSQL migration.

Run this ONCE while the system still uses SQLite FTS5.
Results are saved to data/cache/fts_baseline.json for comparison
after migration to PostgreSQL ts_rank_cd.

Usage:
    python scripts/baseline_fts_quality.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig
from app.state import AppState
from llama_index.core.schema import QueryBundle


BASELINE_QUERIES = [
    # Regulation references (exact match critical)
    "MARPOL Annex VI regulation 14 sulphur content",
    "SOLAS Chapter II-2 fire safety",
    "ISM Code section 9 reports",
    
    # Form/procedure lookups (form numbers matter)
    "master review checklist",
    "bunkering procedure",
    "enclosed space entry permit",
    
    # Gap analysis style (complex, multi-concept)
    "ballast water management plan requirements BWM convention",
    "oil record book entries regulation",
    "safety management system audit nonconformity",
    
    # Vetting / inspection (domain-specific)
    "SIRE VIQ chapter 7 navigation",
    "PSC detention deficiency",
    "risk assessment procedures",
    
    # General maritime queries
    "fire drill frequency requirements",
    "crew training requirements",
    "emergency response procedures",
]


def capture_baseline():
    config = AppConfig.get()
    state = AppState(tenant_id="default")

    if not state.ensure_index_loaded():
        print("ERROR: Could not load index. Is Qdrant running?")
        sys.exit(1)

    fts = state.fts_retriever
    if fts is None:
        print("ERROR: FTS retriever not initialized")
        sys.exit(1)

    results = {}

    for query_str in BASELINE_QUERIES:
        print(f"  Querying: {query_str[:60]}...")

        # 1. FTS-only results
        fts_nodes = fts.retrieve_filtered(query_str=query_str, top_k=20)
        fts_data = []
        for nws in fts_nodes:
            fts_data.append({
                "node_id": nws.node.node_id,
                "doc_id": nws.node.metadata.get("source", ""),
                "score": round(nws.score, 6),
                "text_preview": nws.node.text[:150],
            })

        # 2. Hybrid pipeline results (if available)
        hybrid_data = []
        try:
            from app.query import parallel_hybrid_retrieval
            vec = state.vector_retriever
            bm25 = state.bm25_retriever
            if vec and bm25:
                hybrid_nodes = parallel_hybrid_retrieval(
                    query_str, vec, bm25, k=60, top_k=20
                )
                for nws in hybrid_nodes:
                    hybrid_data.append({
                        "node_id": nws.node.node_id,
                        "doc_id": nws.node.metadata.get("source", ""),
                        "score": round(nws.score, 6),
                        "text_preview": nws.node.text[:150],
                    })
        except Exception as exc:
            print(f"    WARNING: Hybrid pipeline failed: {exc}")

        results[query_str] = {
            "fts_only": fts_data,
            "hybrid_pipeline": hybrid_data,
        }
        print(f"    FTS: {len(fts_data)} results, Hybrid: {len(hybrid_data)} results")

    output = {
        "captured_at": datetime.now().isoformat(),
        "retriever": "SQLiteFTS5Retriever",
        "node_count_at_capture": state.fts_retriever._get_node_count()
            if hasattr(state.fts_retriever, '_get_node_count') else "unknown",
        "queries": results,
    }

    output_path = config.paths.cache_dir / "fts_baseline.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nBaseline saved to {output_path}")
    print(f"Captured {len(results)} queries")


if __name__ == "__main__":
    capture_baseline()
