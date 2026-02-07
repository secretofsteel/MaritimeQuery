#!/usr/bin/env python3
"""
Migration Script: File-Based Sessions to SQLite Multi-Tenant

Migrates sessions from the old file-based storage (data/cache/sessions/)
to the new SQLite-based multi-tenant system.

Old structure:
    data/cache/sessions/
    ├── index.json              # Session metadata (titles, dates)
    ├── {session_id}.jsonl      # Messages for each session
    └── {session_id}/           # Some old sessions might be folders

Usage:
    python migrate_file_sessions_to_sqlite.py --target-tenant alpha_shipping
    python migrate_file_sessions_to_sqlite.py --target-tenant alpha_shipping --dry-run
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def get_db_connection(db_path: Path):
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_index_json(sessions_path: Path) -> Dict[str, Any]:
    """Load the index.json file."""
    index_path = sessions_path / "index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load index.json: {e}")
    return {}


def load_messages_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load messages from a .jsonl file."""
    messages = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        messages.append(msg)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Bad JSON line in {jsonl_path.name}: {e}")
    except OSError as e:
        print(f"  Warning: Could not read {jsonl_path}: {e}")
    return messages


def migrate_session(
    conn,
    session_id: str,
    metadata: Dict[str, Any],
    messages: List[Dict[str, Any]],
    target_tenant: str,
    dry_run: bool = False
) -> bool:
    """Migrate a single session to SQLite."""
    
    # Check if session already exists
    cursor = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,)
    )
    if cursor.fetchone():
        return False  # Already exists
    
    if dry_run:
        return True
    
    # Extract metadata
    title = metadata.get("title", "Imported Session")
    created_at = metadata.get("created_at", datetime.now().isoformat())
    last_active = metadata.get("last_active", created_at)
    message_count = metadata.get("message_count", len(messages))
    
    # Insert session
    conn.execute("""
        INSERT INTO sessions (session_id, tenant_id, title, created_at, last_active, message_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        target_tenant,
        title,
        created_at,
        last_active,
        message_count
    ))
    
    # Insert messages
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp") or msg.get("created_at") or created_at
        
        # Store extra fields as metadata
        extra_fields = {k: v for k, v in msg.items() 
                       if k not in ["id", "role", "content", "timestamp", "created_at"]}
        msg_metadata = json.dumps(extra_fields) if extra_fields else "{}"
        
        conn.execute("""
            INSERT INTO messages (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            role,
            content,
            timestamp,
            msg_metadata
        ))
    
    conn.commit()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate file-based sessions to SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-tenant",
        required=True,
        help="Target tenant_id to assign to migrated sessions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--db-path",
        default="data/maritime.db",
        help="Path to SQLite database (default: data/maritime.db)"
    )
    parser.add_argument(
        "--sessions-path",
        default="data/cache/sessions",
        help="Path to file-based sessions folder (default: data/cache/sessions)"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    sessions_path = Path(args.sessions_path)
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return 1
    
    if not sessions_path.exists():
        print(f"ERROR: Sessions folder not found at {sessions_path}")
        return 1
    
    print(f"{'DRY RUN - ' if args.dry_run else ''}Session Migration (File → SQLite)")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Sessions folder: {sessions_path}")
    print(f"Target tenant: {args.target_tenant}")
    print()
    
    # Load index.json
    index_data = load_index_json(sessions_path)
    print(f"Found index.json with {len(index_data)} session entries")
    
    # Find all .jsonl files (these are the sessions)
    jsonl_files = list(sessions_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} .jsonl session files")
    print()
    
    if not jsonl_files and not index_data:
        print("No sessions to migrate!")
        return 0
    
    conn = get_db_connection(db_path)
    
    # Check current SQLite state
    cursor = conn.execute("SELECT COUNT(*) FROM sessions")
    existing_count = cursor.fetchone()[0]
    print(f"Existing sessions in SQLite: {existing_count}")
    print()
    
    # Migrate each session
    migrated = 0
    skipped = 0
    failed = 0
    total_messages = 0
    
    print("Migrating sessions...")
    
    for jsonl_file in jsonl_files:
        session_id = jsonl_file.stem  # filename without .jsonl
        
        try:
            # Load messages from .jsonl file
            messages = load_messages_from_jsonl(jsonl_file)
            
            # Get metadata from index.json (or create default)
            metadata = index_data.get(session_id, {
                "title": "Imported Session",
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "message_count": len(messages)
            })
            
            # Migrate
            success = migrate_session(
                conn,
                session_id,
                metadata,
                messages,
                args.target_tenant,
                dry_run=args.dry_run
            )
            
            if success:
                migrated += 1
                total_messages += len(messages)
                title = metadata.get("title", "Untitled")[:40]
                print(f"  ✅ {title} ({len(messages)} msgs)")
            else:
                skipped += 1
                
        except Exception as e:
            failed += 1
            print(f"  ❌ {session_id[:8]}... Error: {e}")
    
    print()
    print("=" * 60)
    print(f"Results:")
    print(f"  Migrated: {migrated} sessions ({total_messages} messages)")
    print(f"  Skipped:  {skipped} (already existed)")
    print(f"  Failed:   {failed}")
    
    if not args.dry_run and migrated > 0:
        # Show final count
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE tenant_id = ?", 
            (args.target_tenant,)
        )
        final_count = cursor.fetchone()[0]
        print()
        print(f"Total sessions for '{args.target_tenant}': {final_count}")
    
    conn.close()
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
