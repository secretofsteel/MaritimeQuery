"""Node repository for SQLite storage.

Handles all CRUD operations for nodes in SQLite.
Replaces the pickle-based storage pattern.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from llama_index.core.schema import TextNode

from .database import db_connection, get_db_path, rebuild_fts_index
from .logger import LOGGER


class NodeRepository:
    """
    Repository for node storage in SQLite.
    
    Replaces the pickle-based pattern of:
    - pickle.dump(nodes, file) → add_nodes() / upsert_nodes()
    - pickle.load(file) → get_all_nodes()
    - Manual sync → delete_by_doc() + add_nodes()
    """
    
    def __init__(self, tenant_id: str = "shared", db_path: Optional[Path] = None):
        """
        Initialize repository for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier. Default 'shared' for global docs.
            db_path: Path to SQLite database.
        """
        self.tenant_id = tenant_id
        self.db_path = db_path or get_db_path()
    
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
        
        with db_connection(self.db_path) as conn:
            added = 0
            for node in nodes:
                node_doc_id = doc_id or node.metadata.get("source", "unknown")
                section_id = node.metadata.get("section_id")
                
                try:
                    conn.execute("""
                        INSERT INTO nodes (node_id, doc_id, text, metadata, section_id, tenant_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    added += 1
                except Exception as exc:
                    LOGGER.warning("Failed to add node %s: %s", node.node_id, exc)
        
        LOGGER.info("Added %d nodes for doc %s (tenant=%s)", added, doc_id or "multiple", self.tenant_id)
        return added
    
    def upsert_nodes(self, nodes: List[TextNode], doc_id: Optional[str] = None) -> int:
        """
        Add or update nodes (INSERT OR REPLACE).
        
        Args:
            nodes: List of TextNode objects.
            doc_id: Optional override for doc_id.
        
        Returns:
            Number of nodes upserted.
        """
        if not nodes:
            return 0
        
        now = datetime.now().isoformat()
        
        with db_connection(self.db_path) as conn:
            upserted = 0
            for node in nodes:
                node_doc_id = doc_id or node.metadata.get("source", "unknown")
                section_id = node.metadata.get("section_id")
                
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO nodes 
                        (node_id, doc_id, text, metadata, section_id, tenant_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 
                                COALESCE((SELECT created_at FROM nodes WHERE node_id = ?), ?),
                                ?)
                    """, (
                        node.node_id,
                        node_doc_id,
                        node.text,
                        json.dumps(node.metadata),
                        section_id,
                        self.tenant_id,
                        node.node_id,  # For COALESCE subquery
                        now,  # created_at if new
                        now,  # updated_at
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
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM nodes 
                WHERE doc_id = ? AND tenant_id = ?
            """, (doc_id, self.tenant_id))
            deleted = cursor.rowcount
        
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
        
        with db_connection(self.db_path) as conn:
            # SQLite doesn't have great IN clause performance for large sets
            # But for typical document counts this is fine
            placeholders = ",".join("?" * len(doc_ids))
            params = list(doc_ids) + [self.tenant_id]
            
            cursor = conn.execute(f"""
                DELETE FROM nodes 
                WHERE doc_id IN ({placeholders}) AND tenant_id = ?
            """, params)
            deleted = cursor.rowcount
        
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
        
        with db_connection(self.db_path) as conn:
            placeholders = ",".join("?" * len(node_ids))
            cursor = conn.execute(f"""
                DELETE FROM nodes 
                WHERE node_id IN ({placeholders})
            """, node_ids)
            deleted = cursor.rowcount
        
        LOGGER.info("Deleted %d nodes by ID", deleted)
        return deleted
    
    def get_all_nodes(self) -> List[TextNode]:
        """
        Get all nodes for this tenant.
        
        Returns:
            List of TextNode objects.
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, doc_id, text, metadata, section_id
                FROM nodes
                WHERE tenant_id = ?
                ORDER BY doc_id, node_id
            """, (self.tenant_id,))
            rows = cursor.fetchall()
        
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
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, doc_id, text, metadata, section_id
                FROM nodes
                WHERE doc_id = ? AND tenant_id = ?
                ORDER BY node_id
            """, (doc_id, self.tenant_id))
            rows = cursor.fetchall()
        
        return [self._row_to_node(row) for row in rows]
    
    def get_node_by_id(self, node_id: str) -> Optional[TextNode]:
        """
        Get a specific node by ID.
        
        Args:
            node_id: Node identifier.
        
        Returns:
            TextNode or None if not found.
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT node_id, doc_id, text, metadata, section_id
                FROM nodes
                WHERE node_id = ?
            """, (node_id,))
            row = cursor.fetchone()
        
        if row:
            return self._row_to_node(row)
        return None
    
    def get_doc_ids(self) -> Set[str]:
        """
        Get all unique document IDs for this tenant.
        
        Returns:
            Set of document identifiers.
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT doc_id
                FROM nodes
                WHERE tenant_id = ?
            """, (self.tenant_id,))
            rows = cursor.fetchall()
        
        return {row["doc_id"] for row in rows}
    
    def get_node_count(self) -> int:
        """Get total node count for this tenant."""
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM nodes
                WHERE tenant_id = ?
            """, (self.tenant_id,))
            return cursor.fetchone()[0]
    
    def get_node_count_by_doc(self, doc_id: str) -> int:
        """Get node count for a specific document."""
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM nodes
                WHERE doc_id = ? AND tenant_id = ?
            """, (doc_id, self.tenant_id))
            return cursor.fetchone()[0]
    
    def clear_all(self) -> int:
        """
        Delete ALL nodes for this tenant.
        
        Returns:
            Number of nodes deleted.
        """
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM nodes WHERE tenant_id = ?
            """, (self.tenant_id,))
            deleted = cursor.rowcount
        
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
        
        with db_connection(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE nodes
                SET metadata = ?, updated_at = ?
                WHERE node_id = ?
            """, (json.dumps(metadata), now, node_id))
            updated = cursor.rowcount > 0
        
        if updated:
            LOGGER.debug("Updated metadata for node %s", node_id)
        return updated
    
    def _row_to_node(self, row) -> TextNode:
        """Convert a database row to TextNode."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
        
        # Ensure core fields are in metadata
        metadata.setdefault("source", row["doc_id"])
        if row["section_id"]:
            metadata.setdefault("section_id", row["section_id"])
        
        return TextNode(
            id_=row["node_id"],
            text=row["text"],
            metadata=metadata,
        )


def bulk_insert_nodes(
    nodes: List[TextNode],
    tenant_id: str = "shared",
    db_path: Optional[Path] = None
) -> int:
    """
    Bulk insert nodes with optimized transaction.
    
    For large imports (like migration), this is faster than NodeRepository.add_nodes().
    
    Args:
        nodes: List of TextNode objects.
        tenant_id: Tenant identifier.
        db_path: Path to database.
    
    Returns:
        Number of nodes inserted.
    """
    if not nodes:
        return 0
    
    db_path = db_path or get_db_path()
    now = datetime.now().isoformat()
    
    # Prepare all rows
    rows = []
    for node in nodes:
        doc_id = node.metadata.get("source", "unknown")
        section_id = node.metadata.get("section_id")
        rows.append((
            node.node_id,
            doc_id,
            node.text,
            json.dumps(node.metadata),
            section_id,
            tenant_id,
            now,
            now,
        ))
    
    with db_connection(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO nodes 
            (node_id, doc_id, text, metadata, section_id, tenant_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        inserted = len(rows)
    
    LOGGER.info("Bulk inserted %d nodes (tenant=%s)", inserted, tenant_id)
    return inserted


__all__ = ["NodeRepository", "bulk_insert_nodes"]
