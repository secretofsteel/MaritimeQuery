"""Session-scoped upload handling, chunking, and retrieval."""

from __future__ import annotations

import json
import os
import pickle
from hashlib import md5
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from llama_index.core import Document, Settings as LlamaSettings
from llama_index.core.schema import NodeWithScore

from .config import AppConfig
from .files import clean_text_for_llm, read_doc_for_llm
from .indexing import chunk_documents
from .logger import LOGGER


MAX_UPLOADS_PER_SESSION = 50


@dataclass
class SessionUploadChunk:
    """Single chunk derived from a user-uploaded document."""

    file_id: str
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class SessionUploadManager:
    """Handle parsing, storage, and retrieval of session-specific uploads."""

    def __init__(self, uploads_dir: Optional[Path] = None) -> None:
        if uploads_dir is None:
            config = AppConfig.get()
            uploads_dir = config.paths.cache_dir / "session_uploads"

        self.base_dir = uploads_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._chunks_cache: Dict[str, List[SessionUploadChunk]] = {}

    # ------------------------------------------------------------------
    # Filesystem helpers
    def _metadata_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}_uploads.jsonl"

    def _chunks_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}_uploads.pkl"

    # ------------------------------------------------------------------
    # Metadata management
    def _load_metadata_dict(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        if session_id in self._metadata_cache:
            return {record["file_id"]: record for record in self._metadata_cache[session_id]}

        metadata_path = self._metadata_path(session_id)
        records: Dict[str, Dict[str, Any]] = {}
        if metadata_path.exists():
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed upload record for session %s", session_id)
                    continue
                file_id = record.get("file_id")
                if file_id:
                    records[file_id] = record

        # Cache ordered list for reuse
        ordered = sorted(records.values(), key=lambda r: r.get("uploaded_at", ""))
        self._metadata_cache[session_id] = ordered
        return records

    def _write_metadata(self, session_id: str, records: Iterable[Dict[str, Any]]) -> None:
        metadata_path = self._metadata_path(session_id)
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_metadata(self, session_id: str) -> List[Dict[str, Any]]:
        self._metadata_cache.pop(session_id, None)
        self._load_metadata_dict(session_id)
        return self._metadata_cache.get(session_id, [])

    # ------------------------------------------------------------------
    # Chunk storage helpers
    def _load_chunks(self, session_id: str) -> List[SessionUploadChunk]:
        if session_id in self._chunks_cache:
            return self._chunks_cache[session_id]

        chunks_path = self._chunks_path(session_id)
        if not chunks_path.exists():
            self._chunks_cache[session_id] = []
            return []

        try:
            with chunks_path.open("rb") as handle:
                raw_chunks: List[Dict[str, Any]] = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Failed to load session chunks for %s: %s", session_id, exc)
            raw_chunks = []

        chunks = [
            SessionUploadChunk(
                file_id=chunk.get("file_id", ""),
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                embedding=chunk.get("embedding", []),
                metadata=chunk.get("metadata", {}),
            )
            for chunk in raw_chunks
        ]
        self._chunks_cache[session_id] = chunks
        return chunks

    def _write_chunks(self, session_id: str, chunks: List[SessionUploadChunk]) -> None:
        chunks_path = self._chunks_path(session_id)
        serialisable = [
            {
                "file_id": chunk.file_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": list(chunk.embedding),
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        with chunks_path.open("wb") as handle:
            pickle.dump(serialisable, handle)

    def load_chunks(self, session_id: str) -> List[SessionUploadChunk]:
        return self._load_chunks(session_id)

    # ------------------------------------------------------------------
    # Public operations
    def _derive_display_name(self, filename: str, existing: Iterable[str]) -> str:
        base, ext = os.path.splitext(filename)
        candidate = filename
        counter = 1
        existing_set = set(existing)
        while candidate in existing_set:
            candidate = f"{base} ({counter}){ext}"
            counter += 1
        return candidate

    def add_upload(
        self,
        session_id: str,
        filename: str,
        file_bytes: bytes,
        mime_type: str,
    ) -> Dict[str, Any]:
        """Parse, chunk, embed, and persist a new upload for this session."""

        records = self._load_metadata_dict(session_id)
        if len(records) >= MAX_UPLOADS_PER_SESSION:
            return {"status": "limit", "reason": "Upload limit reached for this session."}

        digest = md5(file_bytes).hexdigest()
        fingerprint = {
            "original_name": filename,
            "size": len(file_bytes),
            "hash": digest,
        }

        for record in records.values():
            if (
                record.get("original_name") == fingerprint["original_name"]
                and record.get("size") == fingerprint["size"]
                and record.get("hash") == fingerprint["hash"]
            ):
                return {"status": "duplicate", "record": record}

        display_name = self._derive_display_name(
            filename, (record.get("display_name") for record in records.values())
        )

        temp_suffix = Path(filename).suffix or ".txt"
        with NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)

        try:
            raw_text = read_doc_for_llm(temp_path)
        finally:
            try:
                temp_path.unlink()
            except OSError:
                LOGGER.debug("Temporary upload file already removed: %s", temp_path)

        cleaned = clean_text_for_llm(raw_text)
        if not cleaned:
            return {"status": "error", "reason": "Uploaded file contained no readable text."}

        document = Document(
            text=cleaned,
            metadata={
                "source": display_name,
                "title": display_name,
                "doc_type": "UPLOAD",
                "session_upload": True,
                "upload_original_name": filename,
            },
        )

        chunks = chunk_documents([document])
        if not chunks:
            return {"status": "error", "reason": "Unable to create chunks from uploaded file."}

        embed_model = LlamaSettings.embed_model
        if embed_model is None:  # pragma: no cover - configuration guard
            return {"status": "error", "reason": "Embedding model is not configured."}

        uploaded_at = datetime.utcnow().isoformat()
        chunk_objects: List[SessionUploadChunk] = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{digest}:{index}"
            metadata = {
                **chunk.metadata,
                "source": display_name,
                "upload_display_name": display_name,
                "upload_original_name": filename,
                "session_upload": True,
                "uploaded_at": uploaded_at,
                "doc_type": "UPLOAD",
                "section": chunk.metadata.get("section") or "Uploaded document content",
                "upload_chunk_id": chunk_id,
                "chunk_id": chunk_id,
            }
            embedding = embed_model.get_text_embedding(chunk.text)
            chunk_objects.append(
                SessionUploadChunk(
                    file_id=digest,
                    chunk_id=chunk_id,
                    text=chunk.text,
                    embedding=list(embedding),
                    metadata=metadata,
                )
            )

        new_record = {
            "file_id": digest,
            "display_name": display_name,
            "original_name": filename,
            "size": len(file_bytes),
            "mime_type": mime_type,
            "uploaded_at": uploaded_at,
            "hash": digest,
            "num_chunks": len(chunk_objects),
            "session_upload": True,
        }

        records[digest] = new_record
        ordered_records = sorted(records.values(), key=lambda r: r.get("uploaded_at", ""))
        self._write_metadata(session_id, ordered_records)
        self._metadata_cache[session_id] = ordered_records

        all_chunks = self._load_chunks(session_id)
        all_chunks.extend(chunk_objects)
        self._write_chunks(session_id, all_chunks)
        self._chunks_cache[session_id] = all_chunks

        return {"status": "added", "record": new_record}

    def search_session_uploads(
        self, session_id: str, query: str, top_k: int = 10, boost: float = 0.15
    ) -> List[NodeWithScore]:
        """Return session upload chunks most relevant to the query."""

        chunks = self._load_chunks(session_id)
        if not chunks:
            return []

        embed_model = LlamaSettings.embed_model
        if embed_model is None:  # pragma: no cover - configuration guard
            LOGGER.warning("Embedding model unavailable; skipping session upload retrieval")
            return []

        query_embedding = np.array(embed_model.get_query_embedding(query))
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        scored: List[NodeWithScore] = []
        for chunk in chunks:
            chunk_vector = np.array(chunk.embedding)
            denom = np.linalg.norm(chunk_vector) * query_norm
            if denom == 0:
                continue
            similarity = float(np.dot(query_embedding, chunk_vector) / denom)
            boosted = similarity * (1 + boost) + boost
            document = Document(text=chunk.text, metadata=chunk.metadata)
            scored.append(NodeWithScore(node=document, score=boosted))

        scored.sort(key=lambda node: node.score, reverse=True)
        return scored[:top_k]

    def delete_session_uploads(self, session_id: str) -> None:
        self._metadata_cache.pop(session_id, None)
        self._chunks_cache.pop(session_id, None)

        metadata_path = self._metadata_path(session_id)
        if metadata_path.exists():
            metadata_path.unlink()

        chunks_path = self._chunks_path(session_id)
        if chunks_path.exists():
            chunks_path.unlink()

    def clear_all(self) -> None:
        self._metadata_cache.clear()
        self._chunks_cache.clear()
        for path in self.base_dir.glob("*_uploads.jsonl"):
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        for path in self.base_dir.glob("*_uploads.pkl"):
            path.unlink(missing_ok=True)  # type: ignore[arg-type]


__all__ = [
    "MAX_UPLOADS_PER_SESSION",
    "SessionUploadChunk",
    "SessionUploadManager",
]

