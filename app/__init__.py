"""
Modular components for the Maritime RAG Streamlit application.

The original single-file implementation lives in `streamlit_app.py`.
This package provides a structured alternative without changing the
core prompts, schemas, or retrieval logic.
"""

from __future__ import annotations

__all__ = [
    "config",
    "constants",
    "files",
    "extraction",
    "indexing",
    "query",
    "feedback",
    "state",
    "ui",
]  # pragma: no cover
