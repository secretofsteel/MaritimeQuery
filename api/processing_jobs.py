"""In-memory processing job tracking."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ProcessingJob:
    """State of a document processing job."""

    job_id: str
    tenant_id: str
    status: str                     # "running" | "completed" | "failed"
    started_at: str                 # ISO timestamp
    completed_at: Optional[str] = None

    # Mode
    mode: str = "sync"              # "sync" | "rebuild"

    # Live progress (written by background thread via callback)
    phase: str = "starting"         # "starting" | "extracting" | "embedding" |
                                    #   "tree_building" | "completed"
    current: int = 0
    total: int = 0
    current_item: str = ""

    # Results (set on completion)
    sync_result: Optional[Dict[str, Any]] = None   # {added, modified, deleted}
    report: Optional[Dict[str, Any]] = None         # ProcessingReport.to_dict()
    error: Optional[str] = None

    # Options passed to processing
    doc_type_override: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "mode": self.mode,
            "phase": self.phase,
            "progress": {
                "current": self.current,
                "total": self.total,
                "current_item": self.current_item,
            },
            "sync_result": self.sync_result,
            "report_summary": {
                "total_files": self.report.get("total_files", 0),
                "successful": self.report.get("successful", 0),
                "warnings": self.report.get("warnings", 0),
                "failed": self.report.get("failed", 0),
                "total_duration_sec": self.report.get("total_duration_sec"),
            } if self.report else None,
            "error": self.error,
        }


# ── Module-level job store ──────────────────────────────────────────────

_JOBS: Dict[str, ProcessingJob] = {}   # keyed by tenant_id
_LOCK = threading.Lock()


def get_job(tenant_id: str) -> Optional[ProcessingJob]:
    """Get current/latest job for a tenant."""
    with _LOCK:
        return _JOBS.get(tenant_id)


def start_job(tenant_id: str, mode: str, doc_type_override: Optional[str] = None) -> ProcessingJob:
    """Create and register a new processing job.

    Raises ValueError if a job is already running for this tenant.
    """
    with _LOCK:
        existing = _JOBS.get(tenant_id)
        if existing and existing.status == "running":
            raise ValueError(f"Processing already running for tenant '{tenant_id}'")

        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            status="running",
            started_at=datetime.now().isoformat(),
            mode=mode,
            doc_type_override=doc_type_override,
        )
        _JOBS[tenant_id] = job
        return job


def complete_job(
    tenant_id: str,
    report: Optional[Dict[str, Any]] = None,
    sync_result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Mark a job as completed or failed."""
    with _LOCK:
        job = _JOBS.get(tenant_id)
        if not job:
            return
        job.completed_at = datetime.now().isoformat()
        job.status = "failed" if error else "completed"
        job.report = report
        job.sync_result = sync_result
        job.error = error
        if not error:
            job.phase = "completed"
