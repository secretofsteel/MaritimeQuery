#!/usr/bin/env python3
"""
Migration Script: File-Based Sessions to SQLite Multi-Tenant

Migrates sessions from the old file-based storage (data/cache/sessions/)
to the new SQLite-based multi-tenant system.

Old structure:
    data/cache/sessions/
    ├── {session_uuid}/
    │   ├── {message_uuid}.jsonl
    │   └── ...
    └── index.json

New structure:
    SQLite tables: sessions, messages

Usage:
    python migrate_file_sessions_to_sqlite.py --target-tenant alpha_shipping
    python migrate_file_sessions_to_sqlite.py --target-tenant alpha_shipping --dry-run
    python migrate_file_sessions_to_sqlite.py --target-tenant alpha_shipping --sessions-path data/cache/sessions
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
    """Load the index.json file if it exists."""
    index_path = sessions_path / "index.json"
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load index.json: {e}")
    return {}


def load_session_messages(session_path: Path) -> List[Dict[str, Any]]:
    """Load all messages from a session folder."""
    messages = []
    
    for jsonl_file in session_path.glob("*.jsonl"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            messages.append(msg)
                        except json.JSONDecodeError:
                            continue
        except OSError as e:
            print(f"  Warning: Could not read {jsonl_file}: {e}")
    
    # Also try loading single .json files (some systems use this)
    for json_file in session_path.glob("*.json"):
        if json_file.name == "metadata.json":
            continue  # Skip metadata files
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    messages.extend(data)
                elif isinstance(data, dict):
                    messages.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    
    return messages


def get_session_metadata(session_path: Path, index_data: Dict) -> Dict[str, Any]:
    """Extract session metadata from folder or index."""
    session_id = session_path.name
    
    # Try to get from index.json first
    if session_id in index_data:
        return index_data[session_id]
    
    # Try metadata.json in session folder
    metadata_path = session_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    
    # Fall back to folder timestamps
    try:
        stat = session_path.stat()
        return {
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "last_active": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except OSError:
        return {
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
        }


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
        print(f"  Skipping {session_id[:8]}... (already exists)")
        return False
    
    if dry_run:
        return True
    
    # Extract metadata
    title = metadata.get("title", "Imported Session")
    created_at = metadata.get("created_at", datetime.now().isoformat())
    last_active = metadata.get("last_active", created_at)
    
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
        len(messages)
    ))
    
    # Insert messages
    for i, msg in enumerate(messages):
        msg_id = msg.get("id", f"{session_id}_{i}")
        role = msg.get("role", "user")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", created_at)
        msg_metadata = json.dumps({k: v for k, v in msg.items() if k not in ["id", "role", "content", "timestamp"]})
        
        try:
            conn.execute("""
                INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                msg_id,
                session_id,
                role,
                content,
                timestamp,
                msg_metadata
            ))
        except sqlite3.IntegrityError:
            # Message already exists, skip
            pass
    
    conn.commit()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate file-based sessions to SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
    
    # Load index.json if available
    index_data = load_index_json(sessions_path)
    if index_data:
        print(f"Found index.json with {len(index_data)} entries")
    
    # Find all session folders (UUID-named directories)
    session_folders = [
        p for p in sessions_path.iterdir() 
        if p.is_dir() and len(p.name) == 36 and "-" in p.name  # UUID format
    ]
    
    print(f"Found {len(session_folders)} session folders")
    print()
    
    if not session_folders:
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
    for session_folder in session_folders:
        session_id = session_folder.name
        
        try:
            # Load messages
            messages = load_session_messages(session_folder)
            
            # Get metadata
            metadata = get_session_metadata(session_folder, index_data)
            
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
                print(f"  ✅ {session_id[:8]}... ({len(messages)} messages)")
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
    
    if not args.dry_run:
        # Show final count
        cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE tenant_id = ?", (args.target_tenant,))
        final_count = cursor.fetchone()[0]
        print()
        print(f"Total sessions for '{args.target_tenant}': {final_count}")
    
    conn.close()
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())