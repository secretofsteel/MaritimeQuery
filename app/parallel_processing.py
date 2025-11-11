"""Parallel document processing for Maritime RAG system."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from llama_index.core import Document, Settings as LlamaSettings

from .extraction import gemini_extract_record
from .logger import LOGGER


@dataclass
class ExtractionResult:
    """Result of a single document extraction."""
    filename: str
    success: bool
    record: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class ProcessingResult:
    """Complete processing results for a batch of documents."""
    successful: List[ExtractionResult]
    failed: List[ExtractionResult]
    total_duration_sec: float
    
    @property
    def success_count(self) -> int:
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed)
    
    @property
    def total_count(self) -> int:
        return self.success_count + self.failure_count


@dataclass  
class EmbeddingBatch:
    """Batch of texts and their generated embeddings."""
    embeddings: List[List[float]]
    duration_sec: float


class ProgressTracker:
    """Thread-safe progress tracking for parallel operations."""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.lock = Lock()
    
    def increment(self, callback: Optional[Callable[[str, int, int, str], None]] = None,
                  phase: str = "processing", item_name: str = ""):
        """Increment counter and call progress callback if provided."""
        with self.lock:
            self.current += 1
            if callback:
                callback(phase, self.current, self.total, item_name)


class ParallelDocumentProcessor:
    """Handles parallel extraction of documents using Gemini API."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize processor with specified worker count."""
        self.max_workers = max_workers
    
    def extract_document(self, doc_path: Path) -> ExtractionResult:
        """Extract a single document using Gemini with error handling."""
        start_time = time.time()
        filename = doc_path.name
        
        try:
            record = gemini_extract_record(doc_path)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Check if extraction had errors
            if "parse_error" in record:
                LOGGER.warning("Extraction failed for %s: %s", filename, record["parse_error"])
                return ExtractionResult(
                    filename=filename,
                    success=False,
                    record=record,
                    error=record["parse_error"],
                    duration_ms=duration_ms
                )
            
            return ExtractionResult(
                filename=filename,
                success=True,
                record=record,
                duration_ms=duration_ms
            )
            
        except Exception as exc:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(exc)
            LOGGER.exception("Exception during extraction of %s", filename)
            
            return ExtractionResult(
                filename=filename,
                success=False,
                error=error_msg,
                duration_ms=duration_ms
            )
    
    def extract_batch(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    ) -> ProcessingResult:
        """Extract multiple documents in parallel."""
        if not file_paths:
            return ProcessingResult(successful=[], failed=[], total_duration_sec=0.0)
        
        start_time = time.time()
        tracker = ProgressTracker(len(file_paths))
        
        successful: List[ExtractionResult] = []
        failed: List[ExtractionResult] = []
        
        LOGGER.info("Starting parallel extraction of %d documents with %d workers", 
                   len(file_paths), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.extract_document, path): path
                for path in file_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                result = future.result()
                
                if result.success:
                    successful.append(result)
                else:
                    failed.append(result)
                
                tracker.increment(progress_callback, "extracting", path.name)
                
                # Log with filename visible in console
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                LOGGER.info("Extracted %s in %dms [%d/%d] %s",
                           result.filename, result.duration_ms,
                           tracker.current, tracker.total, status)
                
                if not result.success:
                    LOGGER.warning("  └─ Error: %s", result.error)
        
        total_duration = time.time() - start_time
        
        LOGGER.info("Parallel extraction complete: %d successful, %d failed in %.2fs",
                   len(successful), len(failed), total_duration)
        
        return ProcessingResult(
            successful=successful,
            failed=failed,
            total_duration_sec=total_duration
        )


class ParallelEmbeddingGenerator:
    """Handles parallel embedding generation using Google Embeddings API."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize with specified worker count."""
        self.max_workers = max_workers
    
    def generate_embedding(self, text: str, index: int) -> Tuple[int, List[float]]:
        """Generate embedding for a single text."""
        try:
            embedding = LlamaSettings.embed_model.get_text_embedding(text)
            return (index, embedding)
        except Exception as exc:
            LOGGER.exception("Failed to generate embedding for text at index %d", index)
            raise
    
    def generate_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    ) -> EmbeddingBatch:
        """Generate embeddings for multiple texts in parallel."""
        if not texts:
            return EmbeddingBatch(embeddings=[], duration_sec=0.0)
        
        start_time = time.time()
        tracker = ProgressTracker(len(texts))
        
        LOGGER.info("Starting parallel embedding generation for %d texts with %d workers",
                   len(texts), self.max_workers)
        
        indexed_embeddings: Dict[int, List[float]] = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.generate_embedding, text, i): i
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result_index, embedding = future.result()
                    indexed_embeddings[result_index] = embedding
                    
                    tracker.increment(progress_callback, "embedding", f"chunk {index+1}")
                    
                    if tracker.current % 100 == 0 or tracker.current == tracker.total:
                        LOGGER.info("Generated embeddings [%d/%d]", tracker.current, tracker.total)
                        
                except Exception as exc:
                    LOGGER.exception("Failed to generate embedding for text %d", index)
                    raise
        
        embeddings = [indexed_embeddings[i] for i in range(len(texts))]
        
        duration = time.time() - start_time
        
        LOGGER.info("Parallel embedding generation complete: %d embeddings in %.2fs (%.2f/sec)",
                   len(embeddings), duration, len(embeddings) / duration if duration > 0 else 0)
        
        return EmbeddingBatch(
            embeddings=embeddings,
            duration_sec=duration
        )


__all__ = [
    "ParallelDocumentProcessor",
    "ParallelEmbeddingGenerator",
    "ProcessingResult",
    "ExtractionResult",
    "EmbeddingBatch",
]
