"""SQLite database initialization and connection management.

Phase 1: Sessions and messages tables
Phase 3: Nodes table with FTS5 for keyword search (replaces pickle + BM25)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from .logger import LOGGER


# Schema version - increment when making breaking changes
SCHEMA_VERSION = 2  # Bumped for Phase 3 (nodes + FTS5)


def get_db_path() -> Path:
    """Get default database path from config."""
    from .config import AppConfig
    config = AppConfig.get()
    return config.paths.data_dir / "maritime.db"


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize database schema if tables don't exist.
    
    Safe to call multiple times - uses CREATE TABLE IF NOT EXISTS.
    
    Args:
        db_path: Path to SQLite database. Uses default if None.
    """
    db_path = db_path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # =====================================================================
        # PHASE 1: Sessions tables
        # =====================================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_tenant 
            ON sessions(tenant_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_last_active 
            ON sessions(last_active DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        # =====================================================================
        # PHASE 3: Nodes table (replaces pickle)
        # =====================================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT,
                section_id TEXT,
                tenant_id TEXT DEFAULT 'shared',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_doc 
            ON nodes(doc_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_tenant 
            ON nodes(tenant_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_section 
            ON nodes(section_id)
        """)
        
        # =====================================================================
        # PHASE 3: FTS5 virtual table for keyword search (replaces BM25)
        # =====================================================================
        
        # Check if FTS table exists (can't use IF NOT EXISTS with virtual tables)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='nodes_fts'
        """)
        
        if cursor.fetchone() is None:
            # Create FTS5 virtual table
            # content='nodes' means it references the nodes table
            # content_rowid='rowid' links to the implicit rowid
            cursor.execute("""
                CREATE VIRTUAL TABLE nodes_fts USING fts5(
                    text,
                    node_id UNINDEXED,
                    tenant_id UNINDEXED,
                    content='nodes',
                    content_rowid='rowid'
                )
            """)
            LOGGER.info("Created FTS5 virtual table: nodes_fts")
            
            # Triggers to keep FTS in sync with nodes table
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS nodes_fts_insert 
                AFTER INSERT ON nodes BEGIN
                    INSERT INTO nodes_fts(rowid, text, node_id, tenant_id) 
                    VALUES (new.rowid, new.text, new.node_id, new.tenant_id);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
                AFTER DELETE ON nodes BEGIN
                    INSERT INTO nodes_fts(nodes_fts, rowid, text, node_id, tenant_id) 
                    VALUES('delete', old.rowid, old.text, old.node_id, old.tenant_id);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS nodes_fts_update
                AFTER UPDATE ON nodes BEGIN
                    INSERT INTO nodes_fts(nodes_fts, rowid, text, node_id, tenant_id) 
                    VALUES('delete', old.rowid, old.text, old.node_id, old.tenant_id);
                    INSERT INTO nodes_fts(rowid, text, node_id, tenant_id) 
                    VALUES (new.rowid, new.text, new.node_id, new.tenant_id);
                END
            """)
            LOGGER.info("Created FTS5 sync triggers")
        
        # =====================================================================
        # Schema version tracking
        # =====================================================================
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO schema_info (key, value) 
            VALUES ('version', ?)
        """, (str(SCHEMA_VERSION),))
        
        conn.commit()
        LOGGER.debug("Database initialized: %s (schema v%d)", db_path, SCHEMA_VERSION)
        
    finally:
        conn.close()


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection with recommended settings.
    
    Settings:
    - WAL mode for concurrent reads
    - Foreign keys enabled
    - Row factory for dict-like access
    
    Args:
        db_path: Path to SQLite database. Uses default if None.
    
    Returns:
        sqlite3.Connection configured for the application.
    
    Note:
        Caller is responsible for closing the connection.
    """
    db_path = db_path or get_db_path()
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row  # Access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
    
    return conn


@contextmanager
def db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    
    Automatically commits on success, rolls back on exception.
    
    Usage:
        with db_connection() as conn:
            conn.execute("INSERT INTO ...")
    
    Args:
        db_path: Path to SQLite database. Uses default if None.
    
    Yields:
        sqlite3.Connection
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_schema_version(db_path: Optional[Path] = None) -> Optional[int]:
    """
    Get current schema version from database.
    
    Returns:
        Schema version number, or None if not initialized.
    """
    db_path = db_path or get_db_path()
    
    if not db_path.exists():
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        )
        row = cursor.fetchone()
        conn.close()
        return int(row[0]) if row else None
    except sqlite3.OperationalError:
        return None


def rebuild_fts_index(db_path: Optional[Path] = None) -> int:
    """
    Rebuild the FTS5 index from the nodes table.
    
    Use after bulk inserts or if FTS gets out of sync.
    
    Returns:
        Number of nodes indexed.
    """
    db_path = db_path or get_db_path()
    
    with db_connection(db_path) as conn:
        # Clear and rebuild
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        
        # Get count
        cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        count = cursor.fetchone()[0]
    
    LOGGER.info("Rebuilt FTS index: %d nodes", count)
    return count


def get_node_count(tenant_id: Optional[str] = None, db_path: Optional[Path] = None) -> int:
    """
    Get count of nodes, optionally filtered by tenant.
    
    Args:
        tenant_id: If provided, count only this tenant's nodes
        db_path: Path to database
    
    Returns:
        Node count
    """
    db_path = db_path or get_db_path()
    
    with db_connection(db_path) as conn:
        if tenant_id:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE tenant_id = ?",
                (tenant_id,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        return cursor.fetchone()[0]


def get_distinct_doc_count(tenant_id: Optional[str] = None, db_path: Optional[Path] = None) -> int:
    """
    Count distinct documents, optionally filtered by tenant.
    
    Args:
        tenant_id: If provided, count only this tenant's documents
        db_path: Path to database
    
    Returns:
        Count of distinct doc_ids
    """
    db_path = db_path or get_db_path()
    
    with db_connection(db_path) as conn:
        if tenant_id:
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM nodes WHERE tenant_id = ?",
                (tenant_id,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM nodes")
        return cursor.fetchone()[0]


__all__ = [
    "init_db",
    "get_connection", 
    "db_connection",
    "get_db_path",
    "get_schema_version",
    "rebuild_fts_index",
    "get_node_count",
    "get_distinct_doc_count",
    "SCHEMA_VERSION",
]
