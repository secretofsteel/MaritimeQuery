"""Session management for chat history using PostgreSQL storage.

This module replaces the SQLite-based session storage with PostgreSQL,
enabling proper multi-tenancy and better data integrity.

The public interface remains identical to the original implementation,
ensuring backward compatibility with the rest of the application.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

from .config import AppConfig
from .pg_database import pg_connection
from .logger import LOGGER


@dataclass
class Session:
    """Represents a chat session."""
    session_id: str
    title: str
    created_at: datetime
    last_active: datetime
    message_count: int
    tenant_id: Optional[str] = None  # Added for multi-tenancy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "message_count": self.message_count,
            "tenant_id": self.tenant_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create Session from dictionary."""
        return cls(
            session_id=data["session_id"],
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            message_count=data["message_count"],
            tenant_id=data.get("tenant_id"),
        )
    
    @classmethod
    def from_row(cls, row) -> "Session":
        """Create Session from database row (SQLite or PostgreSQL)."""
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        last_active = row["last_active"]
        if isinstance(last_active, str):
            last_active = datetime.fromisoformat(last_active)
        return cls(
            session_id=row["session_id"],
            title=row["title"],
            created_at=created_at,
            last_active=last_active,
            message_count=row["message_count"],
            tenant_id=row.get("tenant_id") if isinstance(row, dict) else row["tenant_id"],
        )


@dataclass
class Message:
    """Represents a single message in a session."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_row(cls, row) -> "Message":
        """Create Message from database row (SQLite or PostgreSQL)."""
        # Handle metadata: PG JSONB returns dict, SQLite returns JSON string
        metadata = row["metadata"]
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse message metadata")
                metadata = {}
        # else: already a dict (PG JSONB)

        # Handle timestamp: PG returns datetime, SQLite returns string
        timestamp = row["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            role=row["role"],
            content=row["content"],
            timestamp=timestamp,
            metadata=metadata,
        )


class SessionManager:
    """
    Manages chat sessions using PostgreSQL storage.

    All operations are scoped to the tenant_id provided at initialization,
    ensuring complete data isolation between tenants.
    """

    def __init__(self, tenant_id: str):
        """
        Initialize session manager for a specific tenant.

        Args:
            tenant_id: Tenant identifier. All queries are scoped to this tenant.
        """
        self.tenant_id = tenant_id
        LOGGER.debug("SessionManager initialized for tenant '%s' (PostgreSQL)", tenant_id)
    
    def create_session(self, title: str = "New Chat") -> str:
        """
        Create a new chat session.
        
        Args:
            title: Session title (will be auto-generated after first message if default)
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (session_id, tenant_id, title, created_at, last_active, message_count)
                    VALUES (%s, %s, %s, %s, %s, 0)
                    """,
                    (session_id, self.tenant_id, title, now, now)
                )
        
        LOGGER.info("Created session %s for tenant %s", session_id, self.tenant_id)
        return session_id
    
    def list_sessions(self, limit: Optional[int] = None) -> List[Session]:
        """
        List all sessions for this tenant, sorted by last_active (most recent first).
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of Session objects
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT session_id, tenant_id, title, created_at, last_active, message_count
                    FROM sessions
                    WHERE tenant_id = %s
                    ORDER BY last_active DESC
                """
                params = [self.tenant_id]
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        
        return [Session.from_row(row) for row in rows]
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session object or None if not found (or wrong tenant)
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, tenant_id, title, created_at, last_active, message_count
                    FROM sessions
                    WHERE session_id = %s AND tenant_id = %s
                    """,
                    (session_id, self.tenant_id)
                )
                row = cur.fetchone()
        
        if row:
            return Session.from_row(row)
        return None
    
    def load_messages(self, session_id: str) -> List[Message]:
        """
        Load all messages from a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of Message objects (empty if session doesn't exist or wrong tenant)
        """
        # Verify session belongs to this tenant
        session = self.get_session(session_id)
        if session is None:
            LOGGER.warning("Session not found or access denied: %s", session_id)
            return []
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY id ASC
                    """,
                    (session_id,)
                )
                rows = cur.fetchall()
        
        return [Message.from_row(row) for row in rows]
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to a session.
        
        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (confidence, sources, etc.)
        """
        # Verify session belongs to this tenant
        session = self.get_session(session_id)
        if session is None:
            LOGGER.error("Cannot add message: session %s not found or access denied", session_id)
            return
        
        now = datetime.now().isoformat()
        # Ensure metadata is JSON serializable
        metadata_json = json.dumps(metadata) if metadata else None
        
        with pg_connection() as conn:
            with conn.cursor() as cur:
                # Insert message
                cur.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, role, content, now, metadata_json)
                )
                
                # Update session metadata
                cur.execute(
                    """
                    UPDATE sessions 
                    SET last_active = %s, message_count = message_count + 1
                    WHERE session_id = %s AND tenant_id = %s
                    """,
                    (now, session_id, self.tenant_id)
                )
        
        LOGGER.debug("Added %s message to session %s", role, session_id)
    
    def update_title(self, session_id: str, title: str) -> None:
        """
        Update session title.
        
        Args:
            session_id: Session identifier
            title: New title
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions 
                    SET title = %s
                    WHERE session_id = %s AND tenant_id = %s
                    """,
                    (title, session_id, self.tenant_id)
                )
                
                if cur.rowcount == 0:
                    LOGGER.warning("Failed to update title: session %s not found", session_id)
                else:
                    LOGGER.debug("Updated title for session %s: %s", session_id, title)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if session was deleted, False if not found
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                # Messages are deleted automatically via ON DELETE CASCADE
                cur.execute(
                    """
                    DELETE FROM sessions
                    WHERE session_id = %s AND tenant_id = %s
                    """,
                    (session_id, self.tenant_id)
                )
                
                deleted = cur.rowcount > 0
        
        if deleted:
            LOGGER.info("Deleted session %s", session_id)
        else:
            LOGGER.warning("Session not found for deletion: %s", session_id)
        
        return deleted
    
    def auto_generate_title(
        self, 
        session_id: str, 
        first_query: str, 
        first_answer: str
    ) -> str:
        """
        Generate a descriptive title based on first exchange.
        
        Uses Gemini Flash Lite for title generation with fallback
        to query preview if generation fails.
        
        Args:
            session_id: Session identifier
            first_query: First user query
            first_answer: First assistant answer
        
        Returns:
            Generated title (fallback to query preview if generation fails)
        """
        try:
            config = AppConfig.get()
            
            prompt = f"""Generate a short, descriptive title (3-6 words) for this chat session.

User's first question: "{first_query}"
Assistant's answer preview: "{first_answer[:200]}..."

Requirements:
- Be specific to the topic discussed
- Use title case
- No quotes or punctuation at end
- Examples: "Ballast Water Management Procedures", "PPE Requirements for Deck Work"

Title:"""
            
            response = config.client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=20,
                ),
            )
            
            title = response.text.strip().strip('"').strip("'")
            
            # Fallback if generation produced garbage
            if not title or len(title) < 5 or len(title) > 60:
                raise ValueError("Generated title invalid")
            
            self.update_title(session_id, title)
            LOGGER.info("Auto-generated title for %s: %s", session_id, title)
            return title
            
        except Exception as exc:
            LOGGER.warning("Failed to auto-generate title: %s", exc)
            # Fallback: use first 40 chars of query
            fallback = first_query[:40] + "..." if len(first_query) > 40 else first_query
            self.update_title(session_id, fallback)
            return fallback
    
    def clear_all_sessions(self) -> int:
        """
        Delete all sessions for this tenant (for testing/reset).
        
        Returns:
            Number of sessions deleted
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                # Get count first
                cur.execute(
                    "SELECT COUNT(*) FROM sessions WHERE tenant_id = %s",
                    (self.tenant_id,)
                )
                count_row = cur.fetchone()
                count = count_row["count"] if count_row else 0
                
                # Delete all (messages cascade)
                cur.execute(
                    "DELETE FROM sessions WHERE tenant_id = %s",
                    (self.tenant_id,)
                )
        
        LOGGER.info("Cleared all sessions for tenant %s: %d deleted", 
                    self.tenant_id, count)
        return count
    
    def get_session_count(self) -> int:
        """
        Get total number of sessions for this tenant.
        
        Returns:
            Number of sessions
        """
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM sessions WHERE tenant_id = %s",
                    (self.tenant_id,)
                )
                count_row = cur.fetchone()
                return count_row["count"] if count_row else 0


__all__ = ["Session", "Message", "SessionManager"]
