#!/usr/bin/env python3
"""
Migration Script: Transfer Single-Tenant Sessions to Multi-Tenant

This script migrates existing sessions from a single-tenant deployment
to a specific tenant in the new multi-tenant system.

Usage:
    python migrate_sessions_to_tenant.py --target-tenant alpha_shipping
    python migrate_sessions_to_tenant.py --target-tenant alpha_shipping --dry-run
    python migrate_sessions_to_tenant.py --target-tenant alpha_shipping --source-tenant shared

Arguments:
    --target-tenant   The tenant_id to assign to migrated sessions (REQUIRED)
    --source-tenant   Only migrate sessions with this tenant_id (default: migrate all)
    --dry-run         Show what would be migrated without making changes
    --db-path         Path to maritime.db (default: data/maritime.db)
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime


def get_db_connection(db_path: Path):
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def count_sessions(conn, tenant_id: str = None) -> int:
    """Count sessions, optionally filtered by tenant."""
    if tenant_id:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE tenant_id = ?",
            (tenant_id,)
        )
    else:
        cursor = conn.execute("SELECT COUNT(*) FROM sessions")
    return cursor.fetchone()[0]


def list_sessions_by_tenant(conn) -> dict:
    """Get session counts grouped by tenant."""
    cursor = conn.execute("""
        SELECT tenant_id, COUNT(*) as count 
        FROM sessions 
        GROUP BY tenant_id 
        ORDER BY tenant_id
    """)
    return {row["tenant_id"]: row["count"] for row in cursor.fetchall()}


def migrate_sessions(
    conn, 
    target_tenant: str, 
    source_tenant: str = None,
    dry_run: bool = False
) -> int:
    """
    Migrate sessions to target tenant.
    
    Args:
        conn: Database connection
        target_tenant: Tenant ID to assign
        source_tenant: Only migrate from this tenant (None = all)
        dry_run: If True, don't actually modify
    
    Returns:
        Number of sessions migrated
    """
    if source_tenant:
        # Migrate specific tenant
        if dry_run:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE tenant_id = ?",
                (source_tenant,)
            )
        else:
            cursor = conn.execute(
                "UPDATE sessions SET tenant_id = ? WHERE tenant_id = ?",
                (target_tenant, source_tenant)
            )
    else:
        # Migrate all sessions
        if dry_run:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
        else:
            cursor = conn.execute(
                "UPDATE sessions SET tenant_id = ?",
                (target_tenant,)
            )
    
    if dry_run:
        return cursor.fetchone()[0]
    else:
        conn.commit()
        return cursor.rowcount


def migrate_nodes(
    conn,
    target_tenant: str,
    source_tenant: str = None,
    dry_run: bool = False
) -> int:
    """
    Migrate nodes to target tenant (optional, if needed).
    
    Args:
        conn: Database connection
        target_tenant: Tenant ID to assign
        source_tenant: Only migrate from this tenant (None = all)
        dry_run: If True, don't actually modify
    
    Returns:
        Number of nodes migrated
    """
    if source_tenant:
        if dry_run:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE tenant_id = ?",
                (source_tenant,)
            )
        else:
            cursor = conn.execute(
                "UPDATE nodes SET tenant_id = ? WHERE tenant_id = ?",
                (target_tenant, source_tenant)
            )
    else:
        if dry_run:
            cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        else:
            cursor = conn.execute(
                "UPDATE nodes SET tenant_id = ?",
                (target_tenant,)
            )
    
    if dry_run:
        return cursor.fetchone()[0]
    else:
        conn.commit()
        return cursor.rowcount


def main():
    parser = argparse.ArgumentParser(
        description="Migrate sessions to a specific tenant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--target-tenant",
        required=True,
        help="Target tenant_id to assign to sessions"
    )
    parser.add_argument(
        "--source-tenant",
        default=None,
        help="Only migrate sessions from this tenant (default: all)"
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
        "--include-nodes",
        action="store_true",
        help="Also migrate nodes table (use with caution)"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return 1
    
    print(f"{'DRY RUN - ' if args.dry_run else ''}Session Migration")
    print("=" * 50)
    print(f"Database: {db_path}")
    print(f"Target tenant: {args.target_tenant}")
    if args.source_tenant:
        print(f"Source tenant: {args.source_tenant}")
    print()
    
    conn = get_db_connection(db_path)
    
    # DEBUG: Show table structure
    print("DEBUG - Tables in database:")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for row in cursor:
        print(f"  {row[0]}")
    print()
    
    # DEBUG: Show sessions table structure
    print("DEBUG - Sessions table columns:")
    cursor = conn.execute("PRAGMA table_info(sessions)")
    for row in cursor:
        print(f"  {row[1]} ({row[2]})")
    print()
    
    # DEBUG: Show raw count
    cursor = conn.execute("SELECT COUNT(*) FROM sessions")
    print(f"DEBUG - Total sessions in table: {cursor.fetchone()[0]}")
    print()
    
    # DEBUG: Show
    
    # Migrate sessions
    session_count = migrate_sessions(
        conn,
        target_tenant=args.target_tenant,
        source_tenant=args.source_tenant,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        print(f"Would migrate {session_count} sessions to '{args.target_tenant}'")
    else:
        print(f"✅ Migrated {session_count} sessions to '{args.target_tenant}'")
    
    # Optionally migrate nodes
    if args.include_nodes:
        print()
        node_count = migrate_nodes(
            conn,
            target_tenant=args.target_tenant,
            source_tenant=args.source_tenant,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print(f"Would migrate {node_count} nodes to '{args.target_tenant}'")
        else:
            print(f"✅ Migrated {node_count} nodes to '{args.target_tenant}'")
    
    # Show final state
    if not args.dry_run:
        print()
        print("Final session distribution:")
        for tenant, count in list_sessions_by_tenant(conn).items():
            print(f"  {tenant}: {count} sessions")
    
    conn.close()
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
