#!/usr/bin/env python3
"""Migrate sessions and messages from SQLite to PostgreSQL.

Safe to re-run — skips sessions that already exist in PostgreSQL.

Usage:
    python scripts/migrate_sessions_to_pg.py
    python scripts/migrate_sessions_to_pg.py --dry-run
"""

import argparse
import sqlite3
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db_path
from app.pg_database import pg_connection


def migrate_sessions(dry_run: bool = False):
    db_path = get_db_path()
    if not db_path.exists():
        print(f"SQLite database not found at {db_path}")
        return

    print(f"Migrating from SQLite: {db_path}")
    print(f"Dry run: {dry_run}")

    sqlite_conn = sqlite3.connect(db_path)
    sqlite_conn.row_factory = sqlite3.Row
    
    try:
        # 1. Migrate Sessions
        sessions = sqlite_conn.execute("SELECT * FROM sessions").fetchall()
        print(f"Found {len(sessions)} sessions in SQLite.")

        migrated_sessions = 0
        skipped_sessions = 0
        
        with pg_connection() as pg_conn:
            with pg_conn.cursor() as cur:
                for session in sessions:
                    # Check if exists
                    cur.execute("SELECT 1 FROM sessions WHERE session_id = %s", (session["session_id"],))
                    if cur.fetchone():
                        skipped_sessions += 1
                        continue

                    if not dry_run:
                        cur.execute(
                            """
                            INSERT INTO sessions (session_id, tenant_id, title, created_at, last_active, message_count)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                session["session_id"],
                                session["tenant_id"],
                                session["title"],
                                session["created_at"],
                                session["last_active"],
                                session["message_count"]
                            )
                        )
                    migrated_sessions += 1

        print(f"Sessions: {migrated_sessions} migrated, {skipped_sessions} skipped (already exist).")

        # 2. Migrate Messages
        # We only migrate messages for sessions that we just migrated OR sessions that exist but have 0 messages in PG (incomplete migration)
        # Simpler approach: Iterate all messages, check if session exists in PG, insert if not duplicate.
        # Since messages don't have UUIDs in SQLite (just auto-inc ID), we can't easily check 'exists' by ID.
        # But we can check if the session exists in PG.
        
        # Better approach for idempotency:
        # If we skipped the session (already exists), assume messages are there too.
        # If we inserted the session, we MUST insert messages.
        
        # HOWEVER, what if a previous run crashed halfway through messages?
        # Safe strategy: For each session, delete all messages in PG and re-insert? No, dangerous.
        # Strategy: Count messages in PG. If 0 and SQLite has > 0, migrate.
        
        messages_migrated = 0
        
        with pg_connection() as pg_conn:
            with pg_conn.cursor() as cur:
                # Get all session IDs from SQLite
                for session in sessions:
                    sid = session["session_id"]
                    
                    # Check PG message count
                    cur.execute("SELECT COUNT(*) FROM messages WHERE session_id = %s", (sid,))
                    pg_count = cur.fetchone()["count"]
                    
                    if pg_count > 0:
                        # Assume already migrated
                        continue
                        
                    # Get messages from SQLite
                    msgs = sqlite_conn.execute("SELECT * FROM messages WHERE session_id = ?", (sid,)).fetchall()
                    if not msgs:
                        continue
                        
                    if dry_run:
                        messages_migrated += len(msgs)
                        continue
                        
                    # Insert all
                    for msg in msgs:
                        cur.execute(
                            """
                            INSERT INTO messages (session_id, role, content, timestamp, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                msg["session_id"],
                                msg["role"],
                                msg["content"],
                                msg["timestamp"],
                                msg["metadata"] # JSON string from SQLite, accepted by PG JSONB
                            )
                        )
                    messages_migrated += len(msgs)

        print(f"Messages: {messages_migrated} migrated.")

    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate sessions from SQLite to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without writing to PG")
    args = parser.parse_args()
    
    try:
        migrate_sessions(dry_run=args.dry_run)
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
