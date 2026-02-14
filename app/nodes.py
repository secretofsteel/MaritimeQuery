"""Node repository for PostgreSQL storage.

Handles all CRUD operations for nodes using the shared connection pool.
Replaces the SQLite storage pattern.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from llama_index.core.schema import TextNode

from .pg_database import pg_connection
from .logger import LOGGER


class NodeRepository:
    """
    Repository for node storage in PostgreSQL.
    """
    
    def __init__(self, tenant_id: str = "shared"):
        """
        Initialize repository for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier. Default 'shared' for global docs.
        """
        self.tenant_id = tenant_id
    
    def add_nodes(self, nodes: List[TextNode], doc_id: Optional[str] = None) -> int:
        """
        Add new nodes to the database.
        
        Args:
            nodes: List of TextNode objects to add.
            doc_id: Optional override for doc_id (otherwise extracted from metadata).
        
        Returns:
            Number of nodes added.
        """
        if not nodes:
            return 0
        
        now = datetime.now().isoformat()
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                added = 0
                for node in nodes:
                    node_doc_id = doc_id or node.metadata.get("source", "unknown")
                    section_id = node.metadata.get("section_id")
                    
                    try:
                        cur.execute("""
                            INSERT INTO nodes (node_id, doc_id, text, metadata, section_id, tenant_id, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (node_id) DO NOTHING
                        """, (
                            node.node_id,
                            node_doc_id,
                            node.text,
                            json.dumps(node.metadata),
                            section_id,
                            self.tenant_id,
                            now,
                            now,
                        ))
                        if cur.rowcount > 0:
                            added += 1
                    except Exception as exc:
                        LOGGER.warning("Failed to add node %s: %s", node.node_id, exc)
        
        LOGGER.info("Added %d nodes for doc %s (tenant=%s)", added, doc_id or "multiple", self.tenant_id)
        return added
    
    def upsert_nodes(self, nodes: List[TextNode], doc_id: Optional[str] = None) -> int:
        """
        Add or update nodes.
        
        Args:
            nodes: List of TextNode objects.
            doc_id: Optional override for doc_id.
        
        Returns:
            Number of nodes upserted.
        """
        if not nodes:
            return 0
        
        now = datetime.now().isoformat()
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                upserted = 0
                for node in nodes:
                    node_doc_id = doc_id or node.metadata.get("source", "unknown")
                    section_id = node.metadata.get("section_id")
                    
                    try:
                        cur.execute("""
                            INSERT INTO nodes (node_id, doc_id, text, metadata, section_id, tenant_id, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (node_id) DO UPDATE SET
                                doc_id = EXCLUDED.doc_id,
                                text = EXCLUDED.text,
                                metadata = EXCLUDED.metadata,
                                section_id = EXCLUDED.section_id,
                                tenant_id = EXCLUDED.tenant_id,
                                updated_at = %s
                        """, (
                            node.node_id,
                            node_doc_id,
                            node.text,
                            json.dumps(node.metadata),
                            section_id,
                            self.tenant_id,
                            now,  # created_at (INSERT only)
                            now,  # updated_at (INSERT)
                            now,  # updated_at (UPDATE)
                        ))
                        upserted += 1
                    except Exception as exc:
                        LOGGER.warning("Failed to upsert node %s: %s", node.node_id, exc)
        
        LOGGER.info("Upserted %d nodes (tenant=%s)", upserted, self.tenant_id)
        return upserted
    
    def delete_by_doc(self, doc_id: str) -> int:
        """
        Delete all nodes for a document.
        
        Args:
            doc_id: Document identifier (source filename).
        
        Returns:
            Number of nodes deleted.
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM nodes 
                    WHERE doc_id = %s AND tenant_id = %s
                """, (doc_id, self.tenant_id))
                deleted = cur.rowcount
        
        LOGGER.info("Deleted %d nodes for doc %s (tenant=%s)", deleted, doc_id, self.tenant_id)
        return deleted
    
    def delete_by_docs(self, doc_ids: Set[str]) -> int:
        """
        Delete all nodes for multiple documents.
        
        Args:
            doc_ids: Set of document identifiers.
        
        Returns:
            Number of nodes deleted.
        """
        if not doc_ids:
            return 0
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM nodes 
                    WHERE doc_id = ANY(%s) AND tenant_id = %s
                """, (list(doc_ids), self.tenant_id))
                deleted = cur.rowcount
        
        LOGGER.info("Deleted %d nodes for %d docs (tenant=%s)", deleted, len(doc_ids), self.tenant_id)
        return deleted
    
    def delete_by_node_ids(self, node_ids: List[str]) -> int:
        """
        Delete specific nodes by ID.
        
        Args:
            node_ids: List of node IDs to delete.
        
        Returns:
            Number of nodes deleted.
        """
        if not node_ids:
            return 0
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM nodes 
                    WHERE node_id = ANY(%s)
                """, (node_ids,))
                deleted = cur.rowcount
        
        LOGGER.info("Deleted %d nodes by ID", deleted)
        return deleted
    
    def get_all_nodes(self) -> List[TextNode]:
        """
        Get all nodes for this tenant.
        
        Returns:
            List of TextNode objects.
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT node_id, doc_id, text, metadata, section_id
                    FROM nodes
                    WHERE tenant_id = %s
                    ORDER BY doc_id, node_id
                """, (self.tenant_id,))
                rows = cur.fetchall()
        
        nodes = [self._row_to_node(row) for row in rows]
        LOGGER.debug("Loaded %d nodes (tenant=%s)", len(nodes), self.tenant_id)
        return nodes
    
    def get_nodes_by_doc(self, doc_id: str) -> List[TextNode]:
        """
        Get all nodes for a specific document.
        
        Args:
            doc_id: Document identifier.
        
        Returns:
            List of TextNode objects.
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT node_id, doc_id, text, metadata, section_id
                    FROM nodes
                    WHERE doc_id = %s AND tenant_id = %s
                    ORDER BY node_id
                """, (doc_id, self.tenant_id))
                rows = cur.fetchall()
        
        return [self._row_to_node(row) for row in rows]
    
    def get_node_by_id(self, node_id: str) -> Optional[TextNode]:
        """
        Get a specific node by ID.
        
        Args:
            node_id: Node identifier.
        
        Returns:
            TextNode or None if not found.
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT node_id, doc_id, text, metadata, section_id
                    FROM nodes
                    WHERE node_id = %s
                """, (node_id,))
                row = cur.fetchone()
        
        if row:
            return self._row_to_node(row)
        return None
    
    def get_doc_ids(self) -> Set[str]:
        """
        Get all unique document IDs for this tenant.
        
        Returns:
            Set of document identifiers.
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT doc_id
                    FROM nodes
                    WHERE tenant_id = %s
                """, (self.tenant_id,))
                rows = cur.fetchall()
        
        return {row["doc_id"] for row in rows}
    
    def get_node_count(self) -> int:
        """Get total node count for this tenant."""
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) AS count FROM nodes
                    WHERE tenant_id = %s
                """, (self.tenant_id,))
                row = cur.fetchone()
                return row["count"] if row else 0
    
    def get_node_count_by_doc(self, doc_id: str) -> int:
        """Get node count for a specific document."""
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) AS count FROM nodes
                    WHERE doc_id = %s AND tenant_id = %s
                """, (doc_id, self.tenant_id))
                row = cur.fetchone()
                return row["count"] if row else 0
    
    def clear_all(self) -> int:
        """
        Delete ALL nodes for this tenant.
        
        Returns:
            Number of nodes deleted.
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM nodes WHERE tenant_id = %s
                """, (self.tenant_id,))
                deleted = cur.rowcount
        
        LOGGER.warning("Cleared all %d nodes for tenant %s", deleted, self.tenant_id)
        return deleted
    
    def update_metadata(self, node_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific node.
        
        Args:
            node_id: Node identifier.
            metadata: New metadata dict (replaces existing).
        
        Returns:
            True if node was updated, False if not found.
        """
        now = datetime.now().isoformat()
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE nodes
                    SET metadata = %s, updated_at = %s
                    WHERE node_id = %s
                """, (json.dumps(metadata), now, node_id))
                updated = cur.rowcount > 0
        
        if updated:
            LOGGER.debug("Updated metadata for node %s", node_id)
        return updated
    
    def _row_to_node(self, row) -> TextNode:
        """Convert a database row to TextNode."""
        metadata = row["metadata"] or {}
        # Defensive: handle both dict (PG JSONB) and string (edge case)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        metadata.setdefault("source", row["doc_id"])
        if row.get("section_id"):
            metadata.setdefault("section_id", row["section_id"])
        
        return TextNode(
            id_=row["node_id"],
            text=row["text"],
            metadata=metadata,
        )


def bulk_insert_nodes(
    nodes: List[TextNode],
    tenant_id: str = "shared",
) -> int:
    """Bulk insert nodes with optimized transaction."""
    if not nodes:
        return 0
    
    now = datetime.now().isoformat()
    
    rows = []
    for node in nodes:
        doc_id = node.metadata.get("source", "unknown")
        section_id = node.metadata.get("section_id")
        node_tenant_id = node.metadata.get("tenant_id", tenant_id)
        rows.append((
            node.node_id,
            doc_id,
            node.text,
            json.dumps(node.metadata),
            section_id,
            node_tenant_id,
            now,
            now,
        ))
    
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO nodes
                (node_id, doc_id, text, metadata, section_id, tenant_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    section_id = EXCLUDED.section_id,
                    updated_at = EXCLUDED.updated_at
            """, rows)
            # Rowcount for executemany with psycopg3 isn't always reliable for exact updated counts
            # but usually reflects total operations.
            
    inserted = len(rows)
    LOGGER.info("Bulk inserted %d nodes (tenant=%s)", inserted, tenant_id)
    return inserted


# --- Module-level utility functions (moved from database.py) ---

def get_node_count(tenant_id: Optional[str] = None) -> int:
    """Get count of nodes, optionally filtered by tenant."""
    with pg_connection() as conn:
        with conn.cursor() as cur:
            if tenant_id:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM nodes WHERE tenant_id = %s",
                    (tenant_id,)
                )
            else:
                cur.execute("SELECT COUNT(*) AS count FROM nodes")
            row = cur.fetchone()
            return row["count"] if row else 0


def get_distinct_doc_count(tenant_id: Optional[str] = None) -> int:
    """Count distinct documents, optionally filtered by tenant."""
    with pg_connection() as conn:
        with conn.cursor() as cur:
            if tenant_id:
                cur.execute(
                    "SELECT COUNT(DISTINCT doc_id) AS count FROM nodes WHERE tenant_id = %s",
                    (tenant_id,)
                )
            else:
                cur.execute("SELECT COUNT(DISTINCT doc_id) AS count FROM nodes")
            row = cur.fetchone()
            return row["count"] if row else 0


def rebuild_fts_index() -> int:
    """Refresh tsvector column for all nodes.

    With the PG trigger, tsv is auto-populated on INSERT/UPDATE.
    This function exists for admin use — recalculates tsv for ALL rows.
    Useful if the tsvector config changes or data was loaded bypassing the trigger.
    """
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE nodes SET tsv = to_tsvector('simple', COALESCE(text, ''))
            """)
            cur.execute("SELECT COUNT(*) AS count FROM nodes")
            row = cur.fetchone()
            count = row["count"] if row else 0
            
    LOGGER.info("Rebuilt FTS index (tsvector refresh): %d nodes", count)
    return count


__all__ = [
    "NodeRepository",
    "bulk_insert_nodes",
    "get_node_count",
    "get_distinct_doc_count",
    "rebuild_fts_index",
]
