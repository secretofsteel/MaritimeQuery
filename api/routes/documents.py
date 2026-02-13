"""Document management API endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import PurePosixPath
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from api.dependencies import (
    get_admin_app_state,
    get_current_user,
    get_target_tenant,
)
from app.config import AppConfig
from app.files import current_files_index, load_jsonl
from app.nodes import NodeRepository
from app.services import batch_delete_documents, delete_document_by_source
from app.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# --- Pydantic Models ---

class DocumentInfo(BaseModel):
    """Single document in the listing."""
    filename: str
    file_size_bytes: int
    last_modified: str                       # ISO timestamp
    chunk_count: int                         # from SQLite NodeRepository
    extraction_cached: bool                  # has entry in Gemini cache
    extraction_status: str                   # "success" | "error" | "not_extracted"
    validation_coverage: Optional[float]     # n-gram coverage score, if available


class DocumentListResponse(BaseModel):
    tenant_id: str
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentDetailResponse(BaseModel):
    """Rich detail for a single document."""
    filename: str
    file_size_bytes: int
    last_modified: str
    chunk_count: int
    extraction_cached: bool
    extraction_status: str
    validation_coverage: Optional[float]
    # From Gemini cache — present only if extraction was successful
    title: Optional[str]
    doc_type: Optional[str]
    topics: Optional[List[str]]
    section_count: Optional[int]
    sections: Optional[List[str]]           # section names/titles


class DeleteDocumentResponse(BaseModel):
    filename: str
    success: bool
    deleted_from_disk: bool
    error: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    filenames: List[str] = Field(..., min_length=1)


class BatchDeleteResponse(BaseModel):
    deleted_count: int
    db_entries_removed: int
    error: Optional[str] = None


class FileUploadResult(BaseModel):
    """Result for a single file in the upload batch."""
    filename: str
    saved: bool
    size_bytes: int
    overwritten: bool                # True if file existed and was replaced


class UploadResponse(BaseModel):
    """Response for batch upload."""
    tenant_id: str
    files_saved: int
    files: List[FileUploadResult]


# --- Endpoints ---

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    tenant_id: str = Depends(get_target_tenant),
):
    """List all documents for the target tenant with rich status info."""
    config = AppConfig.get()
    docs_path = config.docs_path_for(tenant_id)
    repo = NodeRepository(tenant_id=tenant_id)

    # 1. Files on disk
    disk_files = current_files_index(docs_path)  # {filename: {name, size, mtime}}

    # 2. Chunk counts from SQLite - get all counts for this tenant first to avoid N+1 queries
    # Optimized: get_node_count_by_doc is per-doc, but we can query all at once if we add a method
    # or just iterate over doc_ids. For now, following plan (N+1 but strictly local SQLite, fast enough)
    doc_ids = repo.get_doc_ids()
    chunk_counts = {doc_id: repo.get_node_count_by_doc(doc_id) for doc_id in doc_ids}

    # 3. Gemini cache status
    gemini_cache_path = config.gemini_cache_for(tenant_id)
    # Check if cache file exists before loading
    if gemini_cache_path.exists():
        gemini_cache = load_jsonl(gemini_cache_path)
    else:
        gemini_cache = {}

    # 4. Merge
    documents = []
    
    # Sort by filename
    sorted_filenames = sorted(disk_files.keys())
    
    for filename in sorted_filenames:
        finfo = disk_files[filename]
        
        # Get cache details
        cache_entry = gemini_cache.get(filename, {})
        gemini_meta = cache_entry.get("gemini", {})

        has_error = "parse_error" in gemini_meta or "extraction_error" in gemini_meta
        has_sections = bool(gemini_meta.get("sections"))

        if has_error:
            ext_status = "error"
        elif has_sections:
            ext_status = "success"
        else:
            ext_status = "not_extracted"

        validation = gemini_meta.get("validation", {})

        documents.append(DocumentInfo(
            filename=filename,
            file_size_bytes=finfo["size"],
            last_modified=datetime.fromtimestamp(finfo["mtime"]).isoformat(),
            chunk_count=chunk_counts.get(filename, 0),
            extraction_cached=filename in gemini_cache,
            extraction_status=ext_status,
            validation_coverage=validation.get("ngram_coverage"),
        ))

    total_chunks = repo.get_node_count()

    return DocumentListResponse(
        tenant_id=tenant_id,
        documents=documents,
        total_documents=len(documents),
        total_chunks=total_chunks,
    )


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt"}


def _sanitize_filename(raw_name: str | None) -> str | None:
    """Strip path components, return None if invalid."""
    if not raw_name:
        return None
    name = PurePosixPath(raw_name).name
    if not name or name.startswith("."):
        return None
    return name


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Documents to upload"),
    overwrite: bool = Query(False, description="Overwrite existing files"),
    tenant_id: str = Depends(get_target_tenant),
):
    """Upload documents to the tenant's library.

    Files are saved to disk only. Call POST /documents/process to trigger
    extraction, chunking, and embedding.
    """
    config = AppConfig.get()
    docs_path = config.docs_path_for(tenant_id)
    
    # Ensure directory exists (it should, but safety first)
    docs_path.mkdir(parents=True, exist_ok=True)

    # --- Validation (all checks before any writes) ---

    # 1. Empty check
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # 2. Sanitize filenames
    sanitized: List[Tuple[UploadFile, str]] = []   # (file, safe_name)
    invalid_names = []

    for f in files:
        safe = _sanitize_filename(f.filename)
        if safe is None:
            invalid_names.append(f.filename or "<empty>")
        else:
            sanitized.append((f, safe))

    if invalid_names:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid filename(s)",
                "rejected": invalid_names,
            },
        )

    # 3. Extension validation
    rejected_ext = []
    for f, safe in sanitized:
        suffix = PurePosixPath(safe).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            rejected_ext.append(safe)

    if rejected_ext:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported file type(s)",
                "rejected": rejected_ext,
                "allowed_extensions": list(ALLOWED_EXTENSIONS),
            },
        )

    # 4. Duplicate check (if not overwriting)
    if not overwrite:
        existing = {p.name for p in docs_path.iterdir() if p.is_file()}
        dupes = [safe for _, safe in sanitized if safe in existing]
        if dupes:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Duplicate filename(s) detected. Set overwrite=true to replace.",
                    "duplicates": dupes,
                },
            )

    # --- Save files ---
    existing_before = {p.name for p in docs_path.iterdir() if p.is_file()} \
                      if overwrite else set()

    results = []
    for upload_file, safe_name in sanitized:
        was_existing = safe_name in existing_before
        dest = docs_path / safe_name
        try:
            content = await upload_file.read()
            dest.write_bytes(content)
            results.append(FileUploadResult(
                filename=safe_name,
                saved=True,
                size_bytes=len(content),
                overwritten=was_existing,
            ))
        except Exception as exc:
            logger.error("Failed to save %s: %s", safe_name, exc)
            results.append(FileUploadResult(
                filename=safe_name,
                saved=False,
                size_bytes=0,
                overwritten=False,
            ))

    saved_count = sum(1 for r in results if r.saved)

    return UploadResponse(
        tenant_id=tenant_id,
        files_saved=saved_count,
        files=results,
    )


@router.post("/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete(
    request: BatchDeleteRequest,
    app_state: AppState = Depends(get_admin_app_state),
):
    """Delete multiple documents in one operation.
    
    Note: Blocks until completion (syncs library).
    """
    deleted_count, db_entries_removed, error = batch_delete_documents(
        filenames=request.filenames,
        app_state=app_state,
    )

    return BatchDeleteResponse(
        deleted_count=deleted_count,
        db_entries_removed=db_entries_removed,
        error=error,
    )


@router.get("/{filename}", response_model=DocumentDetailResponse)
async def get_document_detail(
    filename: str,
    tenant_id: str = Depends(get_target_tenant),
):
    """Get detailed info for a single document."""
    config = AppConfig.get()
    docs_path = config.docs_path_for(tenant_id)
    repo = NodeRepository(tenant_id=tenant_id)

    # Check disk existence
    disk_files = current_files_index(docs_path)
    if filename not in disk_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found for tenant '{tenant_id}'",
        )

    finfo = disk_files[filename]
    
    # Chunk count
    chunk_count = repo.get_node_count_by_doc(filename)

    # Gemini cache
    gemini_cache_path = config.gemini_cache_for(tenant_id)
    gemini_cache = {}
    if gemini_cache_path.exists():
        gemini_cache = load_jsonl(gemini_cache_path)

    cache_entry = gemini_cache.get(filename, {})
    gemini_meta = cache_entry.get("gemini", {})
    
    has_error = "parse_error" in gemini_meta or "extraction_error" in gemini_meta
    has_sections = bool(gemini_meta.get("sections"))

    if has_error:
        ext_status = "error"
    elif has_sections:
        ext_status = "success"
    else:
        ext_status = "not_extracted"

    validation = gemini_meta.get("validation", {})
    
    # Extract cache details if available
    sections=gemini_meta.get("sections", [])
    section_names = [s.get("name", "Untitled") for s in sections] if sections else None

    return DocumentDetailResponse(
        filename=filename,
        file_size_bytes=finfo["size"],
        last_modified=datetime.fromtimestamp(finfo["mtime"]).isoformat(),
        chunk_count=chunk_count,
        extraction_cached=filename in gemini_cache,
        extraction_status=ext_status,
        validation_coverage=validation.get("ngram_coverage"),
        title=gemini_meta.get("title"),
        doc_type=gemini_meta.get("doc_type"),
        topics=gemini_meta.get("topics"),
        section_count=len(sections) if sections else None,
        sections=section_names,
    )


@router.delete("/{filename}", response_model=DeleteDocumentResponse)
async def delete_document(
    filename: str,
    app_state: AppState = Depends(get_admin_app_state),
):
    """Delete a single document.
    
    Note: Blocks until completion (syncs library).
    """
    # tenant_id is inside app_state
    
    result = delete_document_by_source(
        source_filename=filename,
        app_state=app_state,
        tenant_id=app_state.tenant_id,  # Explicitly pass tenant_id
    )

    if not result.success:
        # If file not found, we still return 500? Or 404?
        # Service returns success=False, error="File not found..."
        # We can map specific errors if needed, but 500 is safe per spec
        raise HTTPException(status_code=500, detail=result.error)

    return DeleteDocumentResponse(
        filename=filename,
        success=result.success,
        deleted_from_disk=result.deleted_from_disk,
        error=result.error,
    )
