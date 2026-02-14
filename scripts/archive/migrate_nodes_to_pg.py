#!/usr/bin/env python3
"""Migrate nodes from SQLite to PostgreSQL.

Reads all nodes from SQLite data/maritime.db and inserts into PostgreSQL.
The PG trigger auto-populates the tsv (tsvector) column.

Usage:
    python scripts/migrate_nodes_to_pg.py
    python scripts/migrate_nodes_to_pg.py --dry-run
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig
from app.pg_database import pg_connection, get_pg_pool


def migrate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = AppConfig.get()
    sqlite_path = config.paths.data_dir / "maritime.db"

    print("=" * 60)
    print("NODE MIGRATION: SQLite -> PostgreSQL")
    print("=" * 60)
    print(f"Source: {sqlite_path}")
    print(f"Target: PostgreSQL ({config.database_url.split('@')[1] if '@' in config.database_url else 'localhost'})")
    print()

    if not sqlite_path.exists():
        print("ERROR: SQLite database not found")
        sys.exit(1)

    # Read from SQLite
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT COUNT(*) FROM nodes")
    total = cursor.fetchone()[0]
    print(f"SQLite nodes: {total}")

    cursor = conn.execute("""
        SELECT node_id, doc_id, text, metadata, section_id, tenant_id,
               created_at, updated_at
        FROM nodes
        ORDER BY doc_id, node_id
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No nodes to migrate")
        return

    # Check PG current count
    get_pg_pool()  # Ensure pool is initialized
    with pg_connection() as pgconn:
        with pgconn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM nodes")
            pg_count = cur.fetchone()["count"]

    print(f"PostgreSQL nodes (before): {pg_count}")

    if pg_count > 0:
        print(f"WARNING: PostgreSQL already has {pg_count} nodes.")
        print("Migration uses ON CONFLICT DO NOTHING — existing nodes will be skipped.")

    if args.dry_run:
        print(f"\nDRY RUN: Would migrate {len(rows)} nodes")
        # Show sample
        for row in rows[:3]:
            print(f"  {row['node_id']}: {row['doc_id']} ({len(row['text'])} chars)")
        return

    # Migrate in batches
    BATCH_SIZE = 500
    migrated = 0
    skipped = 0

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]

        with pg_connection() as pgconn:
            with pgconn.cursor() as cur:
                for row in batch:
                    try:
                        cur.execute("""
                            INSERT INTO nodes
                            (node_id, doc_id, text, metadata, section_id, tenant_id,
                             created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (node_id) DO NOTHING
                        """, (
                            row["node_id"],
                            row["doc_id"],
                            row["text"],
                            row["metadata"],  # Already JSON string, PG accepts for JSONB
                            row["section_id"],
                            row["tenant_id"] or "shared",
                            row["created_at"] or datetime.now().isoformat(),
                            row["updated_at"] or datetime.now().isoformat(),
                        ))
                        if cur.rowcount > 0:
                            migrated += 1
                        else:
                            skipped += 1
                    except Exception as exc:
                        print(f"  ERROR on {row['node_id']}: {exc}")

        print(f"  Batch {i // BATCH_SIZE + 1}: processed {len(batch)} rows")

    # Verify
    with pg_connection() as pgconn:
        with pgconn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM nodes")
            final_count = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) AS count FROM nodes WHERE tsv IS NOT NULL")
            tsv_count = cur.fetchone()["count"]

    print()
    print(f"Migration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  PG total nodes: {final_count}")
    print(f"  PG nodes with tsvector: {tsv_count}")

    if tsv_count < final_count:
        print(f"  WARNING: {final_count - tsv_count} nodes missing tsvector")
        print("  Running tsvector refresh...")
        with pg_connection() as pgconn:
            with pgconn.cursor() as cur:
                cur.execute("""
                    UPDATE nodes SET tsv = to_tsvector('simple', COALESCE(text, ''))
                    WHERE tsv IS NULL
                """)
        print("  Done.")

    if final_count == total:
        print(f"\n✓ Counts match: SQLite={total}, PG={final_count}")
    else:
        print(f"\n⚠ Count mismatch: SQLite={total}, PG={final_count}")
        print("  This may be expected if PG had pre-existing nodes.")


if __name__ == "__main__":
    migrate()
