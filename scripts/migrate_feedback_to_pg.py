#!/usr/bin/env python3
"""Migrate feedback from JSONL to PostgreSQL.

Usage:
    python scripts/migrate_feedback_to_pg.py
    python scripts/migrate_feedback_to_pg.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig
from app.pg_database import pg_connection


def migrate_feedback(dry_run: bool = False):
    config = AppConfig.get()
    log_path = config.paths.feedback_log
    
    if not log_path.exists():
        print(f"Feedback log not found at {log_path}")
        return

    print(f"Migrating feedback from: {log_path}")
    print(f"Dry run: {dry_run}")

    entries = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Found {len(entries)} feedback entries.")
    
    migrated_count = 0
    skipped_count = 0

    with pg_connection() as pg_conn:
        with pg_conn.cursor() as cur:
            # Check if table is empty to avoid duplicates on re-run
            # Basic check: if count > 0, warn user
            cur.execute("SELECT COUNT(*) FROM feedback")
            count = cur.fetchone()["count"]
            if count > 0:
                print(f"WARNING: Feedback table already has {count} entries.")
                # Simple dedup: check (timestamp, query) unique combo? 
                # For now, let's just insert if not exact match probably overkill.
                # Let's just append but warn.
            
            for entry in entries:
                # Map fields
                # "top_sources" is a list in JSONL, need to dump to string for PG JSONB
                top_sources = json.dumps(entry.get("top_sources", []))
                
                if not dry_run:
                    cur.execute(
                        """
                        INSERT INTO feedback (
                            tenant_id, query, answer, confidence_pct,
                            confidence_level, num_sources, top_sources,
                            retriever_type, feedback, correction,
                            attempts, was_refined, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            "shared", # JSONL is always shared
                            entry.get("query", ""),
                            entry.get("answer", ""),
                            entry.get("confidence_pct", 0),
                            entry.get("confidence_level", ""),
                            entry.get("num_sources", 0),
                            top_sources,
                            entry.get("retriever_type", "unknown"),
                            entry.get("feedback", ""),
                            entry.get("correction", ""),
                            entry.get("attempts", 1),
                            entry.get("was_refined", False),
                            entry.get("timestamp") # Maps to created_at
                        )
                    )
                migrated_count += 1

    print(f"Feedback: {migrated_count} entries migrated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate feedback from JSONL to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration")
    args = parser.parse_args()
    
    try:
        migrate_feedback(dry_run=args.dry_run)
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
