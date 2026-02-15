"""Document management API endpoints."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import PurePosixPath
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

from api.dependencies import (
    get_admin_app_state,
    get_current_user,
    get_target_tenant,
)
from app.constants import ALLOWED_DOC_TYPES
from app.metadata_updates import (
    apply_corrections_to_gemini_record,
    update_metadata_everywhere,
)
from api.processing_jobs import ProcessingJob, complete_job, get_job, start_job
from app.config import AppConfig
from app.files import current_files_index, load_jsonl
from app.indexing import build_index_from_library_parallel
from app.nodes import NodeRepository
from app.processing_status import load_processing_report
from app.services import batch_delete_documents, delete_document_by_source, rebuild_document_trees
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
    # NEW — from Gemini cache
    title: Optional[str] = None
    doc_type: Optional[str] = None
    topics: Optional[List[str]] = None
    owner_tenant: Optional[str] = None  # NEW


class DocumentListResponse(BaseModel):
    tenant_id: str
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int
    doc_type_filter: Optional[str] = None  # NEW — echo back active filter


class MetadataUpdateRequest(BaseModel):
    """Partial update — only include fields you want to change."""
    title: Optional[str] = None
    doc_type: Optional[str] = None
    form_number: Optional[str] = None
    form_category_name: Optional[str] = None


class BatchMetadataUpdateRequest(BaseModel):
    filenames: List[str]
    # Fields to update (if provided)
    doc_type: Optional[str] = None
    owner_tenant: Optional[str] = None


class BatchMetadataUpdateResponse(BaseModel):
    success_count: int
    failure_count: int
    errors: dict[str, str]  # filename -> error message


class MetadataUpdateResponse(BaseModel):
    filename: str
    success: bool
    updated_fields: List[str]
    error: Optional[str] = None



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


class ProcessRequest(BaseModel):
    """Options for document processing."""
    force_rebuild: bool = False
    doc_type_override: Optional[str] = None


class ProcessingJobResponse(BaseModel):
    """Current state of a processing job."""
    job_id: str
    tenant_id: str
    status: str                     # "running" | "completed" | "failed"
    started_at: str
    completed_at: Optional[str]
    mode: str                       # "sync" | "rebuild"
    phase: str
    progress: dict                  # {current, total, current_item}
    sync_result: Optional[dict]
    report_summary: Optional[dict]
    error: Optional[str]


class ProcessingReportResponse(BaseModel):
    """Full processing report with per-file details."""
    timestamp: str
    total_files: int
    successful: int
    warnings: int
    failed: int
    total_duration_sec: Optional[float]
    file_statuses: List[dict]


# --- Endpoints ---

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    tenant_id: str = Depends(get_target_tenant),
    doc_type: Optional[str] = Query(
        None,
        description="Filter by document type (e.g., FORM, PROCEDURE, CHECKLIST)",
    ),
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
        corrections = cache_entry.get("corrections", {})

        # Apply corrections to get effective values
        effective_meta = apply_corrections_to_gemini_record(gemini_meta, corrections)

        # NEW: filter by doc_type if requested
        file_doc_type = effective_meta.get("doc_type", "")
        if doc_type and file_doc_type and file_doc_type.upper() != doc_type.upper():
            continue
        # If doc_type filter is set but file has no doc_type, skip it (strict filtering)
        if doc_type and not file_doc_type:
            continue

        has_error = "parse_error" in gemini_meta or "extraction_error" in gemini_meta
        has_sections = bool(gemini_meta.get("sections"))  # Sections usually don't change via corrections yet

        if has_error:
            ext_status = "error"
        elif has_sections:
            ext_status = "success"
        else:
            ext_status = "not_extracted"

        validation = gemini_meta.get("validation", {})
        
        # Add these extractions:
        doc_title = effective_meta.get("title")
        doc_topics = effective_meta.get("topics")  # list of strings or None
        doc_owner = cache_entry.get("tenant_id", tenant_id)

        documents.append(DocumentInfo(
            filename=filename,
            file_size_bytes=finfo["size"],
            last_modified=datetime.fromtimestamp(finfo["mtime"]).isoformat(),
            chunk_count=chunk_counts.get(filename, 0),
            extraction_cached=filename in gemini_cache,
            extraction_status=ext_status,
            validation_coverage=validation.get("ngram_coverage"),
            # NEW fields:
            title=doc_title,
            doc_type=file_doc_type if file_doc_type else None,
            topics=doc_topics,
            owner_tenant=doc_owner,
        ))

    total_chunks = repo.get_node_count()

    return DocumentListResponse(
        tenant_id=tenant_id,
        documents=documents,
        total_documents=len(documents),
        total_chunks=total_chunks,
        doc_type_filter=doc_type,
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


async def _run_processing(app, app_state: AppState, job: ProcessingJob) -> None:
    """Run processing in a background thread, updating job state."""
    try:
        if job.mode == "sync":
            await asyncio.to_thread(_run_sync, app, app_state, job)
        else:
            await asyncio.to_thread(_run_rebuild, app, app_state, job)
    except Exception as exc:
        logger.exception("Processing failed for tenant %s", job.tenant_id)
        complete_job(job.tenant_id, error=str(exc))


def _make_progress_callback(job: ProcessingJob):
    """Create a progress callback that updates the job state."""
    def callback(phase: str, current: int, total: int, item: str) -> None:
        job.phase = phase
        job.current = current
        job.total = total
        job.current_item = item
    return callback


def _run_sync(app, app_state: AppState, job: ProcessingJob) -> None:
    """Synchronous sync_library execution."""
    tenant_id = job.tenant_id
    callback = _make_progress_callback(job)

    manager = app_state.ensure_manager(target_tenant_id=tenant_id)
    manager.nodes = app_state.nodes

    try:
        sync_result, report = manager.sync_library(
            app_state.index,
            force_retry_errors=True,
            progress_callback=callback,
            doc_type_override=job.doc_type_override,
        )

        # Update app_state nodes for tree rebuild
        app_state.nodes = manager.nodes

        # Post-processing: rebuild trees
        job.phase = "tree_building"
        try:
            rebuild_document_trees(app_state)
        except Exception as exc:
            logger.warning("Tree rebuild failed (non-fatal): %s", exc)

        # Complete
        complete_job(
            tenant_id,
            report=report.to_dict() if report else None,
            sync_result={
                "added": len(sync_result.added),
                "modified": len(sync_result.modified),
                "deleted": len(sync_result.deleted),
            },
        )
    except Exception as exc:
        raise exc


def _run_rebuild(app, app_state: AppState, job: ProcessingJob) -> None:
    """Synchronous build_index_from_library_parallel execution."""
    tenant_id = job.tenant_id
    callback = _make_progress_callback(job)

    try:
        nodes, index, report = build_index_from_library_parallel(
            progress_callback=callback,
            clear_gemini_cache=False,
            tenant_id=tenant_id,
            doc_type_override=job.doc_type_override,
        )

        # Update shared app state with new index
        app.state.index = index

        # Refresh Qdrant client reference
        from app.vector_store import get_qdrant_client, ensure_collection
        try:
            qdrant_client = get_qdrant_client()
            collection_name = ensure_collection(qdrant_client)
            app.state.qdrant_client = qdrant_client
            app.state.qdrant_collection_name = collection_name
        except Exception as exc:
            logger.error("Failed to refresh Qdrant connection: %s", exc)

        # Update app_state for tree rebuild
        app_state.nodes = nodes

        # Post-processing: rebuild trees
        job.phase = "tree_building"
        try:
            rebuild_document_trees(app_state)
        except Exception as exc:
            logger.warning("Tree rebuild failed (non-fatal): %s", exc)

        # Complete — rebuild has no SyncResult, just report
        complete_job(
            tenant_id,
            report=report.to_dict() if report else None,
            sync_result={"added": report.successful if report else 0, "modified": 0, "deleted": 0},
        )
    except Exception as exc:
        raise exc


@router.post("/process", response_model=ProcessingJobResponse, status_code=202)
async def start_processing(
    request: Request,
    body: ProcessRequest = ProcessRequest(),
    app_state: AppState = Depends(get_admin_app_state),
):
    """Start document processing (extraction, chunking, embedding).

    Auto-detects whether to run incremental sync or full rebuild based on
    whether vectors exist for this tenant. Set force_rebuild=true to force
    a full rebuild.

    Returns 202 immediately. Poll GET /process/status for progress.
    """
    tenant_id = app_state.tenant_id

    # --- Determine mode ---
    has_vectors = False
    if not body.force_rebuild:
        qdrant_client = request.app.state.qdrant_client
        collection_name = request.app.state.qdrant_collection_name
        if qdrant_client and collection_name:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                count_result = qdrant_client.count(
                    collection_name=collection_name,
                    count_filter=Filter(must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
                    ]),
                    exact=False,
                )
                has_vectors = count_result.count > 0
            except Exception:
                pass

    mode = "sync" if (has_vectors and not body.force_rebuild) else "rebuild"

    mode = "sync" if (has_vectors and not body.force_rebuild) else "rebuild"

    # --- Register job (rejects if already running) ---
    try:
        job = start_job(tenant_id, mode, body.doc_type_override)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # --- Launch background processing ---
    asyncio.get_event_loop().create_task(
        _run_processing(request.app, app_state, job)
    )

    return ProcessingJobResponse(**job.to_dict())


@router.get("/process/status", response_model=ProcessingJobResponse)
async def get_processing_status(
    tenant_id: str = Depends(get_target_tenant),
):
    """Get the current/latest processing job status for this tenant.

    Returns live progress while running, or final results if completed.
    """
    job = get_job(tenant_id)
    if not job:
        # Return idle state instead of 404 to avoid log spam
        return ProcessingJobResponse(
            job_id="",
            tenant_id=tenant_id,
            status="idle",
            started_at="",
            completed_at=None,
            mode="none",
            phase="idle",
            progress={"current": 0, "total": 0, "current_item": ""},
            sync_result=None,
            report_summary=None,
            error=None
        )
    return ProcessingJobResponse(**job.to_dict())


@router.get("/process/report", response_model=ProcessingReportResponse)
async def get_processing_report(
    tenant_id: str = Depends(get_target_tenant),
):
    """Get the last completed processing report.

    Returns detailed per-file processing results from the most recent
    sync or rebuild. This data persists across app restarts.
    """
    config = AppConfig.get()
    report_path = config.paths.cache_dir / "last_sync_report.json"

    report = load_processing_report(report_path)
    if not report:
        raise HTTPException(
            status_code=404,
            detail="No processing report found. Run processing first.",
        )

    return ProcessingReportResponse(**report.to_dict())


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


@router.patch("/{filename}/metadata", response_model=MetadataUpdateResponse)
async def update_document_metadata(
    filename: str,
    request: MetadataUpdateRequest,
    app_state: AppState = Depends(get_admin_app_state),
):
    """Update metadata for a processed document.

    Applies corrections to the Gemini cache, in-memory nodes, and Qdrant
    vector store. Only include fields you want to change.

    Supported fields: title, doc_type, form_number, form_category_name.
    """
    # Build corrections dict from non-None fields
    corrections = {}
    if request.title is not None:
        corrections["title"] = request.title
    if request.doc_type is not None:
        # Validate against allowed types
        if request.doc_type.upper() not in [dt.upper() for dt in ALLOWED_DOC_TYPES]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Invalid doc_type: {request.doc_type}",
                    "allowed": ALLOWED_DOC_TYPES,
                },
            )
        corrections["doc_type"] = request.doc_type.upper()
    if request.form_number is not None:
        corrections["form_number"] = request.form_number
    if request.form_category_name is not None:
        corrections["form_category_name"] = request.form_category_name

    if not corrections:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Get nodes for in-memory update
    nodes = app_state.nodes or []

    try:
        success = update_metadata_everywhere(filename, corrections, nodes)
    except Exception as exc:
        logger.error("Metadata update failed for %s: %s", filename, exc)
        return MetadataUpdateResponse(
            filename=filename,
            success=False,
            updated_fields=[],
            error=str(exc),
        )

    if success:
        logger.info(
            "Metadata updated: file=%s fields=%s",
            filename,
            list(corrections.keys()),
        )

    return MetadataUpdateResponse(
        filename=filename,
        success=success,
        updated_fields=list(corrections.keys()) if success else [],
        error=None if success else "Update failed — check server logs",
    )


@router.post("/batch-metadata", response_model=BatchMetadataUpdateResponse)
async def batch_update_metadata(
    request: BatchMetadataUpdateRequest,
    app_state: AppState = Depends(get_admin_app_state),
):
    """Batch update metadata and/or ownership for multiple documents.

    Supports updating `doc_type` and `owner_tenant` for a list of files.
    """
    from app.metadata_updates import transfer_document_ownership

    success_count = 0
    failure_count = 0
    errors = {}
    
    current_tenant = app_state.tenant_id
    nodes = app_state.nodes or []

    for filename in request.filenames:
        file_success = True
        file_errors = []

        # 1. Update doc_type if provided
        if request.doc_type:
            try:
                # Reuse existing update logic
                update_metadata_everywhere(
                    filename=filename,
                    corrections={"doc_type": request.doc_type},
                    nodes=nodes,
                    base_path=None # derived inside
                )
            except Exception as e:
                file_success = False
                file_errors.append(f"Metadata update failed: {str(e)}")
                logger.error("Batch update failed for %s: %s", filename, e)

        # 2. Transfer ownership if provided and different
        if request.owner_tenant and request.owner_tenant != current_tenant:
            try:
                # If we already failed metadata, we might still try transfer? 
                # Better to try to do both independently or stop?
                # Let's try transfer even if metadata failed, as they are separate systems.
                transfer_success = transfer_document_ownership(filename, request.owner_tenant)
                if not transfer_success:
                     file_success = False
                     file_errors.append("Transfer failed")
            except Exception as e:
                file_success = False
                file_errors.append(f"Transfer error: {str(e)}")
                logger.error("Batch transfer failed for %s: %s", filename, e)

        if file_success:
            success_count += 1
        else:
            failure_count += 1
            errors[filename] = "; ".join(file_errors)

    return BatchMetadataUpdateResponse(
        success_count=success_count,
        failure_count=failure_count,
        errors=errors,
    )


@router.get("/{filename}/metadata")
async def get_document_metadata(
    filename: str,
    tenant_id: str = Depends(get_target_tenant),
):
    """Get document metadata including any corrections applied.

    Returns the Gemini-extracted metadata with corrections overlaid,
    plus the raw corrections dict so the UI can show what's been changed.
    """
    config = AppConfig.get()
    gemini_cache_path = config.gemini_cache_for(tenant_id)

    if gemini_cache_path.exists():
        cache = load_jsonl(gemini_cache_path)
    else:
        cache = {}

    entry = cache.get(filename)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No cache entry for {filename}")

    gemini_meta = entry.get("gemini", {})
    corrections = entry.get("corrections", {})

    # Apply corrections to get effective values
    effective = apply_corrections_to_gemini_record(gemini_meta, corrections)

    return {
        "filename": filename,
        "title": effective.get("title"),
        "doc_type": effective.get("doc_type"),
        "form_number": effective.get("form_number"),
        "form_category_name": effective.get("form_category_name") or effective.get("category"),
        "topics": effective.get("topics"),
        "section_count": len(effective.get("sections", [])),
        "corrections_applied": corrections,  # what's been manually overridden
        "has_corrections": bool(corrections),
        "owner_tenant": entry.get("tenant_id", tenant_id),
    }


class TransferOwnershipRequest(BaseModel):
    new_tenant_id: str = Field(..., description="Target tenant to transfer this document to")


class TransferOwnershipResponse(BaseModel):
    filename: str
    success: bool
    old_tenant: str
    new_tenant: str
    error: Optional[str] = None


@router.post("/{filename}/transfer", response_model=TransferOwnershipResponse)
async def transfer_document(
    filename: str,
    request: TransferOwnershipRequest,
    app_state: AppState = Depends(get_admin_app_state),
):
    """Transfer a document to a different tenant.

    Moves the physical file, updates the Gemini cache record, and updates
    sync caches. The caller should re-sync or rebuild after transfer to
    update vector store tenant assignments.

    Superuser only (enforced by get_admin_app_state using get_target_tenant).
    """
    from app.metadata_updates import transfer_document_ownership

    old_tenant = app_state.tenant_id

    if request.new_tenant_id == old_tenant:
        return TransferOwnershipResponse(
            filename=filename,
            success=True,
            old_tenant=old_tenant,
            new_tenant=old_tenant,
            error=None,
        )

    try:
        success = transfer_document_ownership(filename, request.new_tenant_id)
    except Exception as exc:
        logger.error("Transfer failed for %s: %s", filename, exc)
        return TransferOwnershipResponse(
            filename=filename,
            success=False,
            old_tenant=old_tenant,
            new_tenant=request.new_tenant_id,
            error=str(exc),
        )

    if success:
        logger.info("Transferred %s: %s → %s", filename, old_tenant, request.new_tenant_id)

    return TransferOwnershipResponse(
        filename=filename,
        success=success,
        old_tenant=old_tenant,
        new_tenant=request.new_tenant_id,
        error=None if success else "Transfer failed — check server logs",
    )

