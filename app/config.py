"""Environment configuration and directory management."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from google import genai
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI as LlamaGemini
from google.genai.types import EmbedContentConfig, GenerateContentConfig, ThinkingConfig

from .logger import LOGGER


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(env_key: str, default: Path) -> Path:
    """Resolve a path from environment variables or revert to a default."""
    value = os.getenv(env_key)
    return ensure_directory(Path(value).expanduser().resolve()) if value else ensure_directory(default.resolve())


def load_api_key() -> str:
    """Retrieve the Gemini API key or raise a helpful error."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is required.")
    return key


@dataclass(frozen=True)
class PathConfig:
    base_dir: Path
    data_dir: Path
    docs_path: Path
    chroma_path: Path
    cache_dir: Path
    gemini_json_cache: Path
    nodes_cache_path: Path
    cache_info_path: Path
    feedback_log: Path


def build_paths(base_dir: Optional[Path] = None) -> PathConfig:
    """Produce all filesystem paths used by the application."""
    base = base_dir or Path(__file__).resolve().parent.parent
    data_dir = resolve_path("MARITIME_RAG_DATA_DIR", base / "data")
    docs_path = resolve_path("MARITIME_RAG_DOCS", data_dir / "docs")
    chroma_path = resolve_path("MARITIME_RAG_CHROMA", data_dir / "chroma_db")
    cache_dir = resolve_path("MARITIME_RAG_CACHE", data_dir / "cache")
    return PathConfig(
        base_dir=base,
        data_dir=data_dir,
        docs_path=docs_path,
        chroma_path=chroma_path,
        cache_dir=cache_dir,
        gemini_json_cache=cache_dir / "gemini_extract_cache.jsonl",
        nodes_cache_path=cache_dir / "nodes_cache.pkl",
        cache_info_path=cache_dir / "nodes_cache_info.json",
        feedback_log=cache_dir / "feedback_log.jsonl",
    )


def configure_llama_settings(api_key: str) -> None:

    generation_config = GenerateContentConfig(temperature=0.3, thinking_config=ThinkingConfig(thinking_budget=1024))

    """Set global LlamaIndex configuration."""
    LlamaSettings.llm = LlamaGemini(
        model="gemini-2.5-flash",
        api_key=api_key,
        generation_config=generation_config
    )

    # Define the embedding configuration with the desired output dimension
    embedding_config = EmbedContentConfig(output_dimensionality=768)

    # Set up the embedding model in LlamaIndex settings
    LlamaSettings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=api_key,
        embedding_config=embedding_config
    )


class AppConfig:
    """Singleton-like accessor around shared configuration."""

    _instance: Optional["AppConfig"] = None

    def __init__(self) -> None:
        api_key = load_api_key()
        os.environ["GOOGLE_API_KEY"] = api_key

        self.paths = build_paths()
        configure_llama_settings(api_key)
        self.client = genai.Client(api_key=api_key)
        LOGGER.debug("Configuration initialised with base directory %s", self.paths.base_dir)
        self.ocr_enabled = os.getenv("ENABLE_PDF_OCR", "true").lower() == "true"
        self.visual_extraction_enabled = os.getenv("ENABLE_VISUAL_EXTRACTION", "true").lower() == "true"

        LOGGER.debug("Configuration initialised with base directory %s", self.paths.base_dir)
        if self.ocr_enabled:
            LOGGER.info("PDF OCR enabled")
        else:
            LOGGER.info("PDF OCR disabled")

        if self.visual_extraction_enabled:
            LOGGER.info("PDF visual extraction (images/drawings) enabled")
        else:
            LOGGER.info("PDF visual extraction (images/drawings) disabled")


    def update_paths(self, docs_path: Path, chroma_path: Path, cache_dir: Path) -> None:
        """Allow runtime overrides of key directories."""
        docs_path = ensure_directory(docs_path.expanduser().resolve())
        chroma_path = ensure_directory(chroma_path.expanduser().resolve())
        cache_dir = ensure_directory(cache_dir.expanduser().resolve())

        self.paths = PathConfig(
            base_dir=self.paths.base_dir,
            data_dir=self.paths.data_dir,
            docs_path=docs_path,
            chroma_path=chroma_path,
            cache_dir=cache_dir,
            gemini_json_cache=cache_dir / "gemini_extract_cache.jsonl",
            nodes_cache_path=cache_dir / "nodes_cache.pkl",
            cache_info_path=cache_dir / "nodes_cache_info.json",
            feedback_log=cache_dir / "feedback_log.jsonl",
        )
        LOGGER.info("Updated runtime paths: docs=%s chroma=%s cache=%s", docs_path, chroma_path, cache_dir)

    def update_llama_settings(self, api_key: str) -> None:
        """Allow runtime overrides of LlamaIndex settings."""
        configure_llama_settings(api_key)    
    
    def docs_path_for(self, tenant_id: str) -> Path:
        """Get the docs directory for a specific tenant.
        
        Creates the directory if it doesn't exist.
        Example: data/docs/shared/, data/docs/union/
        """
        return ensure_directory(self.paths.docs_path / tenant_id)

    def gemini_cache_for(self, tenant_id: str) -> Path:
        """Get the Gemini extraction cache path for a specific tenant.
        
        Example: data/cache/gemini_extract_cache_shared.jsonl
        """
        return self.paths.cache_dir / f"gemini_extract_cache_{tenant_id}.jsonl"

    def find_doc_file(self, source_filename: str) -> Optional[Path]:
        """Search all tenant folders for a document file.
        
        Used by document viewer and inspector where the tenant context
        may not be known. Returns the first match found.
        
        Args:
            source_filename: Just the filename (e.g. 'ISM_Code.pdf')
            
        Returns:
            Full path to the file, or None if not found.
        """
        for tenant_dir in self.paths.docs_path.iterdir():
            if tenant_dir.is_dir():
                candidate = tenant_dir / source_filename
                if candidate.exists():
                    return candidate
        # Legacy fallback: file directly in docs/ (pre-migration)
        legacy = self.paths.docs_path / source_filename
        return legacy if legacy.exists() else None

    @classmethod
    def get(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


__all__ = ["AppConfig", "PathConfig", "build_paths", "configure_llama_settings", "load_api_key"]
