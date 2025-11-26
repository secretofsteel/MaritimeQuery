"""Document processing status tracking for index building and library sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProcessingStage(Enum):
    """Stages of document processing."""
    PARSING = "parsing"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    EMBEDDING = "embedding"


class StageStatus(Enum):
    """Status of a processing stage."""
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"
    PENDING = "pending"


@dataclass
class StageResult:
    """Result of a single processing stage."""
    status: StageStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DocumentProcessingStatus:
    """Complete processing status for a single document."""
    filename: str
    
    # Stage results
    parsing: StageResult = field(default_factory=lambda: StageResult(StageStatus.PENDING))
    extraction: StageResult = field(default_factory=lambda: StageResult(StageStatus.PENDING))
    validation: StageResult = field(default_factory=lambda: StageResult(StageStatus.PENDING))
    embedding: StageResult = field(default_factory=lambda: StageResult(StageStatus.PENDING))
    
    # Overall metadata
    file_size_bytes: Optional[int] = None
    chunks_created: Optional[int] = None
    processing_time_sec: Optional[float] = None
    used_cache: bool = False
    
    @property
    def overall_status(self) -> StageStatus:
        """Determine overall status from all stages."""
        # If any stage failed, overall is failed
        if any(stage.status == StageStatus.FAILED for stage in [
            self.parsing, self.extraction, self.validation, self.embedding
        ]):
            return StageStatus.FAILED
        
        # If any stage has warning, overall is warning
        if any(stage.status == StageStatus.WARNING for stage in [
            self.parsing, self.extraction, self.validation, self.embedding
        ]):
            return StageStatus.WARNING
        
        # All must be success for overall success
        if all(stage.status == StageStatus.SUCCESS for stage in [
            self.parsing, self.extraction, self.validation, self.embedding
        ]):
            return StageStatus.SUCCESS
        
        # Some stages still pending
        return StageStatus.PENDING
    
    @property
    def status_emoji(self) -> str:
        """Get emoji for overall status."""
        status_map = {
            StageStatus.SUCCESS: "✅",
            StageStatus.WARNING: "⚠️",
            StageStatus.FAILED: "❌",
            StageStatus.SKIPPED: "⏭️",
            StageStatus.PENDING: "⏳",
        }
        return status_map[self.overall_status]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "parsing": {
                "status": self.parsing.status.value,
                "message": self.parsing.message,
                "details": self.parsing.details,
            },
            "extraction": {
                "status": self.extraction.status.value,
                "message": self.extraction.message,
                "details": self.extraction.details,
            },
            "validation": {
                "status": self.validation.status.value,
                "message": self.validation.message,
                "details": self.validation.details,
            },
            "embedding": {
                "status": self.embedding.status.value,
                "message": self.embedding.message,
                "details": self.embedding.details,
            },
            "overall_status": self.overall_status.value,
            "file_size_bytes": self.file_size_bytes,
            "chunks_created": self.chunks_created,
            "processing_time_sec": self.processing_time_sec,
            "used_cache": self.used_cache,
        }


@dataclass
class ProcessingReport:
    """Complete report of index building or library sync operation."""
    timestamp: str
    total_files: int
    successful: int = 0
    warnings: int = 0
    failed: int = 0
    
    # Detailed status for each file
    file_statuses: List[DocumentProcessingStatus] = field(default_factory=list)
    
    # Overall timing
    total_duration_sec: Optional[float] = None
    
    def add_status(self, status: DocumentProcessingStatus) -> None:
        """Add a file status and update counts."""
        self.file_statuses.append(status)
        
        overall = status.overall_status
        if overall == StageStatus.SUCCESS:
            self.successful += 1
        elif overall == StageStatus.WARNING:
            self.warnings += 1
        elif overall == StageStatus.FAILED:
            self.failed += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_files": self.total_files,
            "successful": self.successful,
            "warnings": self.warnings,
            "failed": self.failed,
            "total_duration_sec": self.total_duration_sec,
            "file_statuses": [fs.to_dict() for fs in self.file_statuses],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessingReport:
        """Create from dictionary."""
        report = cls(
            timestamp=data["timestamp"],
            total_files=data["total_files"],
            successful=data.get("successful", 0),
            warnings=data.get("warnings", 0),
            failed=data.get("failed", 0),
            total_duration_sec=data.get("total_duration_sec"),
        )
        
        # Reconstruct file statuses (simplified - just for display)
        for fs_dict in data.get("file_statuses", []):
            status = DocumentProcessingStatus(filename=fs_dict["filename"])
            
            # Reconstruct stage results
            for stage_name in ["parsing", "extraction", "validation", "embedding"]:
                stage_data = fs_dict.get(stage_name, {})
                stage_result = StageResult(
                    status=StageStatus(stage_data.get("status", "pending")),
                    message=stage_data.get("message"),
                    details=stage_data.get("details"),
                )
                setattr(status, stage_name, stage_result)
            
            status.file_size_bytes = fs_dict.get("file_size_bytes")
            status.chunks_created = fs_dict.get("chunks_created")
            status.processing_time_sec = fs_dict.get("processing_time_sec")
            status.used_cache = fs_dict.get("used_cache", False)
            
            report.file_statuses.append(status)
        
        return report


def save_processing_report(report: ProcessingReport, output_path: Path) -> None:
    """Save processing report to JSON file."""
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


def load_processing_report(input_path: Path) -> Optional[ProcessingReport]:
    """Load processing report from JSON file."""
    import json
    
    if not input_path.exists():
        return None
    
    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ProcessingReport.from_dict(data)
    except Exception:
        return None
