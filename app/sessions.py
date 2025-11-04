"""Session management for chat history using JSONL storage."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

from .config import AppConfig
from .logger import LOGGER


@dataclass
class Session:
    """Represents a chat session."""
    session_id: str
    title: str
    created_at: datetime
    last_active: datetime
    message_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "message_count": self.message_count,
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
        )


@dataclass
class Message:
    """Represents a single message in a session."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)  # For confidence, sources, etc.
    
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


class SessionManager:
    """Manages chat sessions using JSONL file storage."""
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        """
        Initialize session manager.
        
        Args:
            sessions_dir: Directory to store session files. If None, uses config default.
        """
        if sessions_dir is None:
            config = AppConfig.get()
            sessions_dir = config.paths.cache_dir / "sessions"
        
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.sessions_dir / "index.json"
        
        # Create index if doesn't exist
        if not self.index_file.exists():
            self.index_file.write_text("{}")
        
        LOGGER.debug("SessionManager initialized: %s", self.sessions_dir)
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load session index from disk."""
        try:
            return json.loads(self.index_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            LOGGER.warning("Failed to load session index, creating new one")
            return {}
    
    def _save_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        """Save session index to disk."""
        self.index_file.write_text(json.dumps(index, indent=2))
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session JSONL file."""
        return self.sessions_dir / f"{session_id}.jsonl"
    
    def create_session(self, title: str = "New Chat") -> str:
        """
        Create a new chat session.
        
        Args:
            title: Session title (will be auto-generated after first message if default)
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = Session(
            session_id=session_id,
            title=title,
            created_at=now,
            last_active=now,
            message_count=0,
        )
        
        # Update index
        index = self._load_index()
        index[session_id] = session.to_dict()
        self._save_index(index)
        
        # Create empty session file
        self._get_session_file(session_id).touch()
        
        LOGGER.info("Created session: %s", session_id)
        return session_id
    
    def list_sessions(self, limit: Optional[int] = None) -> List[Session]:
        """
        List all sessions, sorted by last_active (most recent first).
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of Session objects
        """
        index = self._load_index()
        sessions = [Session.from_dict(data) for data in index.values()]
        
        # Sort by last_active descending
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        
        if limit:
            sessions = sessions[:limit]
        
        return sessions
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session object or None if not found
        """
        index = self._load_index()
        session_data = index.get(session_id)
        
        if session_data:
            return Session.from_dict(session_data)
        return None
    
    def load_messages(self, session_id: str) -> List[Message]:
        """
        Load all messages from a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of Message objects
        """
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            LOGGER.warning("Session file not found: %s", session_id)
            return []
        
        messages = []
        for line in session_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    messages.append(Message.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    LOGGER.warning("Failed to parse message in %s: %s", session_id, exc)
                    continue
        
        return messages
    
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
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        # Append to JSONL file
        session_file = self._get_session_file(session_id)
        with session_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
        
        # Update session metadata
        index = self._load_index()
        if session_id in index:
            session_data = index[session_id]
            session_data["last_active"] = datetime.now().isoformat()
            session_data["message_count"] = session_data.get("message_count", 0) + 1
            self._save_index(index)
        
        LOGGER.debug("Added %s message to session %s", role, session_id)
    
    def update_title(self, session_id: str, title: str) -> None:
        """
        Update session title.
        
        Args:
            session_id: Session identifier
            title: New title
        """
        index = self._load_index()
        if session_id in index:
            index[session_id]["title"] = title
            self._save_index(index)
            LOGGER.info("Updated session %s title to: %s", session_id, title)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its messages.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        # Remove from index
        index = self._load_index()
        if session_id not in index:
            LOGGER.warning("Session not found for deletion: %s", session_id)
            return False
        
        del index[session_id]
        self._save_index(index)
        
        # Delete JSONL file
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()
        
        LOGGER.info("Deleted session: %s", session_id)
        return True
    
    def auto_generate_title(self, session_id: str, first_query: str, first_answer: str) -> str:
        """
        Auto-generate a session title based on first Q&A using Gemini Flash Lite.
        
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
            
            # Fallback if generation failed or produced garbage
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
        Delete all sessions (for testing/reset).
        
        Returns:
            Number of sessions deleted
        """
        index = self._load_index()
        count = len(index)
        
        # Delete all JSONL files
        for session_id in index.keys():
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()
        
        # Clear index
        self._save_index({})
        
        LOGGER.info("Cleared all sessions: %d deleted", count)
        return count


__all__ = ["Session", "Message", "SessionManager"]
