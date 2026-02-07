#!/usr/bin/env python3
"""
Pickle to SQLite Migration Script
=================================

Migrates existing nodes_cache.pkl data to SQLite database.
Also migrates any JSONL session data if not already in SQLite.

Run this ONCE after deploying Phase 3 code.

Usage:
    python scripts/migrate_pickle_to_sqlite.py
    
    # Dry run (show what would be migrated without changing anything)
    python scripts/migrate_pickle_to_sqlite.py --dry-run
    
    # Force re-migration (useful if first attempt had issues)
    python scripts/migrate_pickle_to_sqlite.py --force
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import AppConfig
from app.database import init_db, db_connection, get_db_path, rebuild_fts_index
from app.nodes import bulk_insert_nodes
from app.logger import LOGGER


def migrate_pickle_nodes(dry_run: bool = False, force: bool = False) -> bool:
    """
    Migrate nodes from pickle to SQLite.
    
    Args:
        dry_run: If True, don't actually migrate, just show what would happen.
        force: If True, migrate even if SQLite already has nodes.
    
    Returns:
        True if migration succeeded (or wasn't needed), False on error.
    """
    config = AppConfig.get()
    pickle_path = config.paths.nodes_cache_path
    db_path = get_db_path()
    
    print("=" * 60)
    print("PICKLE TO SQLITE MIGRATION")
    print("=" * 60)
    print()
    print(f"Pickle path: {pickle_path}")
    print(f"SQLite path: {db_path}")
    print()
    
    # Check pickle exists
    if not pickle_path.exists():
        print("✓ No pickle file found - nothing to migrate")
        return True
    
    # Initialize database
    print("Initializing database schema...")
    init_db(db_path)
    
    # Check if SQLite already has nodes
    with db_connection(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        existing_count = cursor.fetchone()[0]
    
    if existing_count > 0 and not force:
        print(f"✓ SQLite already has {existing_count} nodes")
        print("  Use --force to re-migrate (will clear existing nodes)")
        return True
    
    # Load pickle
    print(f"Loading pickle file...")
    try:
        with open(pickle_path, "rb") as f:
            nodes = pickle.load(f)
        print(f"  Loaded {len(nodes)} nodes from pickle")
    except Exception as exc:
        print(f"✗ Failed to load pickle: {exc}")
        return False
    
    if not nodes:
        print("✓ Pickle is empty - nothing to migrate")
        return True
    
    # Analyze nodes
    doc_ids = set()
    for node in nodes:
        doc_id = node.metadata.get("source", "unknown")
        doc_ids.add(doc_id)
    
    print(f"  Found {len(doc_ids)} unique documents")
    print()
    
    if dry_run:
        print("DRY RUN - No changes will be made")
        print()
        print("Would migrate:")
        print(f"  - {len(nodes)} nodes")
        print(f"  - {len(doc_ids)} documents")
        print()
        print("Documents:")
        for doc_id in sorted(doc_ids)[:20]:
            print(f"    {doc_id}")
        if len(doc_ids) > 20:
            print(f"    ... and {len(doc_ids) - 20} more")
        return True
    
    # Clear existing if force
    if force and existing_count > 0:
        print(f"Clearing {existing_count} existing nodes (--force)...")
        with db_connection(db_path) as conn:
            conn.execute("DELETE FROM nodes")
    
    # Migrate
    print("Migrating nodes to SQLite...")
    try:
        inserted = bulk_insert_nodes(nodes, tenant_id="shared", db_path=db_path)
        print(f"  Inserted {inserted} nodes")
    except Exception as exc:
        print(f"✗ Migration failed: {exc}")
        return False
    
    # Rebuild FTS index
    print("Rebuilding FTS5 index...")
    try:
        rebuild_fts_index(db_path)
        print("  FTS5 index rebuilt")
    except Exception as exc:
        print(f"⚠ FTS rebuild failed (non-fatal): {exc}")
    
    # Verify
    print("Verifying migration...")
    with db_connection(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        final_count = cursor.fetchone()[0]
    
    if final_count == len(nodes):
        print(f"✓ Verified: {final_count} nodes in SQLite")
    else:
        print(f"⚠ Count mismatch: pickle={len(nodes)}, sqlite={final_count}")
    
    # Backup pickle
    backup_path = pickle_path.with_suffix(".pkl.backup")
    print(f"Backing up pickle to {backup_path}...")
    try:
        os.rename(pickle_path, backup_path)
        print("  Pickle renamed to .backup")
    except Exception as exc:
        print(f"⚠ Could not rename pickle (non-fatal): {exc}")
    
    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print()
    print(f"✓ {len(nodes)} nodes migrated to SQLite")
    print(f"✓ Original pickle backed up to {backup_path}")
    print()
    print("Next steps:")
    print("1. Test the application to verify everything works")
    print("2. If issues, restore: mv nodes_cache.pkl.backup nodes_cache.pkl")
    print("3. If working, optionally delete the backup after a week")
    
    return True


def verify_migration() -> bool:
    """
    Verify that migration was successful.
    
    Checks:
    - SQLite has nodes
    - FTS5 is working
    - Node count seems reasonable
    """
    db_path = get_db_path()
    
    print("Verifying migration...")
    
    # Check node count
    with db_connection(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        node_count = cursor.fetchone()[0]
        
        if node_count == 0:
            print("✗ No nodes in SQLite")
            return False
        
        print(f"✓ {node_count} nodes in SQLite")
        
        # Check FTS
        cursor = conn.execute("SELECT COUNT(*) FROM nodes_fts")
        fts_count = cursor.fetchone()[0]
        
        if fts_count != node_count:
            print(f"⚠ FTS count mismatch: {fts_count} vs {node_count}")
            print("  Run: python -c 'from app.database import rebuild_fts_index; rebuild_fts_index()'")
        else:
            print(f"✓ FTS5 index has {fts_count} entries")
        
        # Test FTS search
        cursor = conn.execute("""
            SELECT COUNT(*) FROM nodes_fts WHERE nodes_fts MATCH '"safety"'
        """)
        search_count = cursor.fetchone()[0]
        print(f"✓ FTS search test: {search_count} nodes match 'safety'")
        
        # Check documents
        cursor = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM nodes")
        doc_count = cursor.fetchone()[0]
        print(f"✓ {doc_count} unique documents")
    
    print()
    print("Migration verified successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate pickle node cache to SQLite"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-migration even if SQLite already has nodes"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing migration, don't migrate"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_migration()
    else:
        success = migrate_pickle_nodes(dry_run=args.dry_run, force=args.force)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
