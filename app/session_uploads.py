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
# REMOVE these imports from top:
# from .extraction import gemini_extract_record, to_documents_from_gemini
# from .files import clean_text_for_llm, read_doc_for_llm
# from .indexing import chunk_documents
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
    
    # Save extraction JSONL
    def _get_session_extraction_dir(self, session_id: str) -> Path:
        """Get directory for storing extraction JSONLs for a session."""
        config = AppConfig.get()
        extraction_dir = config.paths.cache_dir / "sessions" / session_id / "uploads"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        return extraction_dir

    def save_extraction_jsonl(
        self,
        session_id: str,
        file_hash: str,
        filename: str,
        gemini_result: Dict[str, Any],
        documents: List[Document]
    ) -> None:
        """
        Save Gemini extraction result as JSONL for later restoration.
        
        This allows re-indexing uploaded files when loading old sessions
        without storing the binary files.
        
        Args:
            session_id: Session ID
            file_hash: MD5 hash of file content
            filename: Original filename
            gemini_result: Gemini extraction metadata
            documents: Extracted Document objects
        """
        extraction_dir = self._get_session_extraction_dir(session_id)
        jsonl_path = extraction_dir / f"{file_hash}.jsonl"
        
        # Serialize documents
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "text": doc.text,
                "metadata": doc.metadata,
            })
        
        extraction_data = {
            "filename": filename,
            "file_hash": file_hash,
            "saved_at": datetime.utcnow().isoformat(),
            "gemini_extraction": gemini_result,
            "documents": doc_dicts,
            "doc_count": len(doc_dicts),
        }
        
        with jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(extraction_data, ensure_ascii=False) + "\n")
        
        LOGGER.info("💾 Saved extraction JSONL for %s (%s)", filename, file_hash[:8])

    # Load extraction JSONLs
    def load_extraction_jsonls(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load all extraction JSONLs for a session.
        
        Returns list of extraction data dicts, each containing:
        - filename: Original filename
        - file_hash: MD5 hash
        - documents: List of document dicts
        - gemini_extraction: Original Gemini metadata
        """
        extraction_dir = self._get_session_extraction_dir(session_id)
        
        if not extraction_dir.exists():
            return []
        
        extractions = []
        for jsonl_file in extraction_dir.glob("*.jsonl"):
            try:
                with jsonl_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            extraction = json.loads(line)
                            extractions.append(extraction)
            except Exception as exc:
                LOGGER.warning("Failed to load extraction JSONL %s: %s", jsonl_file.name, exc)
                continue
        
        LOGGER.info("📂 Loaded %d extraction JSONLs for session %s", len(extractions), session_id[:8])
        return extractions

    #Re-index from extraction JSONLs
    def restore_uploads_from_jsonl(self, session_id: str) -> Dict[str, Any]:
        """
        Re-index uploaded files from saved extraction JSONLs.
        
        Called when loading a session with uploaded files.
        
        Process:
        1. Load extraction JSONLs
        2. Reconstruct Document objects
        3. Chunk documents
        4. Generate embeddings
        5. Store as session uploads
        
        Returns:
            Dict with 'status', 'restored_count', 'failed_count'
        """
        from .indexing import chunk_documents
        
        extractions = self.load_extraction_jsonls(session_id)
        
        if not extractions:
            return {"status": "no_uploads", "restored_count": 0, "failed_count": 0}
        
        LOGGER.info("🔄 Re-indexing %d uploaded files from extraction JSONLs...", len(extractions))
        
        embed_model = LlamaSettings.embed_model
        if embed_model is None:
            LOGGER.error("Embedding model not configured, cannot restore uploads")
            return {"status": "error", "reason": "Embedding model not configured"}
        
        restored_count = 0
        failed_count = 0
        all_restored_chunks = []
        restored_metadata = []
        
        for extraction in extractions:
            try:
                filename = extraction.get("filename", "unknown")
                file_hash = extraction.get("file_hash")
                doc_dicts = extraction.get("documents", [])
                
                if not doc_dicts:
                    LOGGER.warning("No documents in extraction for %s, skipping", filename)
                    failed_count += 1
                    continue
                
                # Reconstruct Document objects
                documents = []
                for doc_dict in doc_dicts:
                    doc = Document(
                        text=doc_dict.get("text", ""),
                        metadata=doc_dict.get("metadata", {})
                    )
                    # Ensure upload markers
                    doc.metadata["session_upload"] = True
                    doc.metadata["upload_display_name"] = filename
                    doc.metadata["upload_original_name"] = filename
                    documents.append(doc)
                
                # Chunk the documents
                chunks = chunk_documents(documents)
                if not chunks:
                    LOGGER.warning("No chunks created for %s, skipping", filename)
                    failed_count += 1
                    continue
                
                # Generate embeddings for chunks
                uploaded_at = extraction.get("saved_at", datetime.utcnow().isoformat())
                chunk_objects = []
                
                for index, chunk in enumerate(chunks):
                    chunk_id = f"{file_hash}:{index}"
                    metadata = {
                        **chunk.metadata,
                        "uploaded_at": uploaded_at,
                        "upload_chunk_id": chunk_id,
                        "chunk_id": chunk_id,
                    }
                    
                    # Generate embedding
                    try:
                        embedding = embed_model.get_text_embedding(chunk.text)
                    except Exception as embed_exc:
                        LOGGER.warning("Failed to embed chunk %d of %s: %s", index, filename, embed_exc)
                        continue
                    
                    chunk_objects.append(
                        SessionUploadChunk(
                            file_id=file_hash,
                            chunk_id=chunk_id,
                            text=chunk.text,
                            embedding=list(embedding),
                            metadata=metadata,
                        )
                    )
                
                if chunk_objects:
                    all_restored_chunks.extend(chunk_objects)
                    
                    # Reconstruct metadata record
                    restored_metadata.append({
                        "file_id": file_hash,
                        "display_name": filename,
                        "original_name": filename,
                        "size": 0,  # Don't have original size
                        "mime_type": "restored",
                        "uploaded_at": uploaded_at,
                        "hash": file_hash,
                        "num_chunks": len(chunk_objects),
                        "session_upload": True,
                        "doc_type": documents[0].metadata.get("doc_type", "UPLOAD"),
                    })
                    
                    restored_count += 1
                    LOGGER.info("✅ Restored %s: %d chunks", filename, len(chunk_objects))
                else:
                    failed_count += 1
                    LOGGER.warning("No valid chunks for %s", filename)
                    
            except Exception as exc:
                LOGGER.error("Failed to restore upload %s: %s", extraction.get("filename", "unknown"), exc)
                failed_count += 1
                continue
        
        # Save restored chunks and metadata
        if all_restored_chunks:
            self._write_chunks(session_id, all_restored_chunks)
            self._chunks_cache[session_id] = all_restored_chunks
            
            self._write_metadata(session_id, restored_metadata)
            self._metadata_cache[session_id] = restored_metadata
            
            LOGGER.info("✅ Upload restoration complete: %d files restored, %d failed", 
                    restored_count, failed_count)
        
        return {
            "status": "restored",
            "restored_count": restored_count,
            "failed_count": failed_count,
            "total_chunks": len(all_restored_chunks),
        }


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
        
        # Import here to avoid circular dependency
        from .extraction import gemini_extract_record, to_documents_from_gemini
        from .indexing import chunk_documents

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
            # CRITICAL FIX: Use Gemini extraction for structured processing
            LOGGER.info("Processing %s with Gemini extraction (flash-lite)...", display_name)
            try:
                gemini_meta = gemini_extract_record(temp_path)
                if "parse_error" in gemini_meta:
                    LOGGER.warning("Gemini extraction failed for %s: %s, falling back to simple parsing", 
                                 display_name, gemini_meta.get("parse_error"))
                    # Fallback to simple extraction
                    documents = self._simple_parse_file(temp_path, display_name)
                else:
                    # Success - use Gemini-extracted documents
                    documents = to_documents_from_gemini(temp_path, gemini_meta)
                    LOGGER.info("✅ Gemini extraction succeeded: %d sections found", len(documents))
                # NEW: Save extraction JSONL for future restoration
                if "parse_error" not in gemini_meta:
                    try:
                        self.save_extraction_jsonl(
                            session_id=session_id,
                            file_hash=digest,
                            filename=display_name,
                            gemini_result=gemini_meta,
                            documents=documents
                        )
                    except Exception as save_exc:
                        LOGGER.warning("Failed to save extraction JSONL for %s: %s", display_name, save_exc)
                        # Don't fail the upload if JSONL save fails
            except Exception as exc:
                LOGGER.warning("Gemini extraction error for %s: %s, falling back to simple parsing", 
                             display_name, exc)
                documents = self._simple_parse_file(temp_path, display_name)
            
            if not documents:
                return {"status": "error", "reason": "No readable content found in uploaded file."}
            
            # Mark all documents as session uploads
            for doc in documents:
                doc.metadata["session_upload"] = True
                doc.metadata["upload_display_name"] = display_name
                doc.metadata["upload_original_name"] = filename
            
            # Chunk the documents
            chunks = chunk_documents(documents)
            if not chunks:
                return {"status": "error", "reason": "Unable to create chunks from uploaded file."}
        
        finally:
            try:
                temp_path.unlink()
            except OSError:
                LOGGER.debug("Temporary upload file already removed: %s", temp_path)
        
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
            # Add doc_type from Gemini extraction if available
            "doc_type": documents[0].metadata.get("doc_type", "UPLOAD") if documents else "UPLOAD",
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

    def _simple_parse_file(self, temp_path: Path, display_name: str) -> List[Document]:
        """Fallback simple parsing when Gemini fails."""
        # Import here to avoid circular dependency
        from .files import clean_text_for_llm, read_doc_for_llm
        LOGGER.info("Using simple text extraction for %s", display_name)
        raw_text = read_doc_for_llm(temp_path)
        cleaned = clean_text_for_llm(raw_text)
        if not cleaned:
            return []
        
        metadata = {
            "source": display_name,
            "title": display_name.rsplit('.', 1)[0].replace('_', ' ').title(),
            "doc_type": "UPLOAD",
            "section": "Full Document Content",
        }
        return [Document(text=cleaned, metadata=metadata)]

    def search_session_uploads(
        self, session_id: str, query: str, top_k: int = 10, boost: float = 0.5
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
            # CRITICAL FIX: Apply 50% boost (multiply by 1.5)
            boosted_score = similarity * 1.5
            document = Document(text=chunk.text, metadata=chunk.metadata)
            scored.append(NodeWithScore(node=document, score=boosted_score))
        
        scored.sort(key=lambda node: node.score, reverse=True)
        LOGGER.info("Session upload search: found %d chunks, returning top %d (boost=%.1f%%)", 
                   len(scored), min(top_k, len(scored)), boost * 100)
        return scored[:top_k]

    def delete_session_extraction_jsonls(self, session_id: str) -> None:
        """Delete all extraction JSONLs for a session."""
        extraction_dir = self._get_session_extraction_dir(session_id)
        
        if extraction_dir.exists():
            import shutil
            try:
                shutil.rmtree(extraction_dir)
                LOGGER.info("🗑️  Deleted extraction JSONLs for session %s", session_id[:8])
            except Exception as exc:
                LOGGER.warning("Failed to delete extraction JSONLs for %s: %s", session_id[:8], exc)

    def delete_session_uploads(self, session_id: str) -> None:
        self._metadata_cache.pop(session_id, None)
        self._chunks_cache.pop(session_id, None)
        metadata_path = self._metadata_path(session_id)
        if metadata_path.exists():
            metadata_path.unlink()
        chunks_path = self._chunks_path(session_id)
        if chunks_path.exists():
            chunks_path.unlink()
        self.delete_session_extraction_jsonls(session_id)

    def clear_all(self) -> None:
        self._metadata_cache.clear()
        self._chunks_cache.clear()
        for path in self.base_dir.glob("*_uploads.jsonl"):
            path.unlink(missing_ok=True)
        for path in self.base_dir.glob("*_uploads.pkl"):
            path.unlink(missing_ok=True)


__all__ = [
    "MAX_UPLOADS_PER_SESSION",
    "SessionUploadChunk",
    "SessionUploadManager",
]