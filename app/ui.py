"""Streamlit UI helpers and rendering functions."""

from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional
import html
from markdown import markdown as md
import streamlit as st
import shutil
import pickle

from .constants import (
    load_form_categories, 
    save_form_categories, 
    get_form_categories_path,
    ALLOWED_DOC_TYPES,
    MAX_CONTEXT_TURNS
)
from .config import AppConfig
from .metadata_updates import update_metadata_everywhere, get_correction_status
from .indexing import build_index_from_library, build_index_from_library_parallel, load_cached_nodes_and_index
from .query import query_with_confidence, cohere_client  # legacy fallback
from .orchestrator import orchestrated_query
from .state import AppState
from .logger import LOGGER
from .session_uploads import MAX_UPLOADS_PER_SESSION
from .processing_status import ProcessingReport, DocumentProcessingStatus, StageStatus
from app.document_inspector_helpers import (
    load_cached_text,
    load_document_tree,
    load_extraction_data,
    format_tree_visually,
    identify_problem_documents,
    get_document_metrics,
)
from .services import (
    sync_memory_to_db,
    rebuild_document_trees,
    delete_document_by_source,
    batch_delete_documents,
    delete_entire_library,
    copy_uploaded_files,
    delete_duplicate_files,
    sanitize_markdown_tables
)

_DEF_EXTS = ["extra", "tables", "fenced_code", "sane_lists"]


@st.dialog("Document Viewer", width="large")
def _view_document_dialog(source_filename: str, display_title: str) -> None:
    """Render a document preview in a modal dialog."""
    import base64
    import pandas as pd

    st.caption(display_title)

    file_path = AppConfig.get().find_doc_file(source_filename)

    if not file_path:
        st.error(f"File not found: {source_filename}")
        return

    ext = file_path.suffix.lower()
    size_mb = file_path.stat().st_size / (1024 * 1024)

    if ext == ".pdf":
        if size_mb > 30:
            st.warning(f"PDF too large to preview ({size_mb:.1f} MB). Max ~30 MB for in-browser viewing.")
            return
        b64 = base64.b64encode(file_path.read_bytes()).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="690" style="border:none; border-radius:4px;"></iframe>',
            unsafe_allow_html=True,
        )

    elif ext in (".xlsx", ".xls"):
        try:
            xls = pd.ExcelFile(file_path)
            sheet = (
                st.selectbox("Sheet", xls.sheet_names, key="docview_sheet_select")
                if len(xls.sheet_names) > 1
                else xls.sheet_names[0]
            )
            df = pd.read_excel(file_path, sheet_name=sheet)
            # Fix mixed-type columns that break Arrow serialization
            df = df.astype({col: str for col in df.select_dtypes(include=["object"]).columns})
            st.dataframe(df, width="stretch", height=700)
        except Exception as exc:
            st.error(f"Failed to read spreadsheet: {exc}")

    elif ext == ".docx":
        try:
            import mammoth
            with open(file_path, "rb") as f:
                result = mammoth.convert_to_html(f)
            if result.value.strip():
                with st.container(height=650):
                    st.html(result.value)
                if result.messages:
                    with st.expander("⚠️ Conversion warnings"):
                        for msg in result.messages:
                            st.caption(str(msg))
            else:
                st.info("Document appears empty or could not be converted.")
        except ImportError:
            # Fallback to cached text if mammoth not installed
            text = load_cached_text(source_filename)
            if text:
                with st.container(height=700):
                    st.markdown(text)
            else:
                st.info("No preview available for this document.")
        except Exception as exc:
            st.error(f"Failed to render document: {exc}")

    elif ext == ".doc":
        text = load_cached_text(source_filename)
        if text:
            with st.container(height=650):
                st.markdown(text)
        else:
            st.info("No extracted text cached for this document.")

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
        st.image(str(file_path))

    else:
        # Fallback: try reading as plain text
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            with st.container(height=600):
                st.code(text[:50_000])
        except Exception:
            st.warning(f"Cannot preview {ext} files.")

    # Download button
    if file_path.exists():
        st.download_button(
            label="Download File",
            data=file_path.read_bytes(),
            file_name=source_filename,
            key="docview_download",
            type="primary",
        )


def _get_tenant_display_names() -> Dict[str, str]:
    """Get mapping of tenant_id -> display name from users.yaml."""
    from pathlib import Path
    import yaml
    
    display_names = {"shared": "Shared (All Tenants)"}
    
    config_path = Path("config/users.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                users_config = yaml.safe_load(f)
            
            credentials = users_config.get("credentials", {}).get("usernames", {})
            for username, user_data in credentials.items():
                tenant_id = user_data.get("tenant_id")
                name = user_data.get("name", tenant_id)
                if tenant_id and tenant_id not in display_names:
                    display_names[tenant_id] = name
        except Exception as exc:
            LOGGER.warning("Failed to load tenant names from users.yaml: %s", exc)
    
    return display_names

def _get_tenant_list() -> List[str]:
    """
    Get list of available tenants from users.yaml config.
    
    Returns list of tenant IDs including 'shared'.
    """
    from pathlib import Path
    import yaml
    
    tenants = ["shared"]  # Always include shared
    
    # Load from users.yaml
    config_path = Path("config/users.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                users_config = yaml.safe_load(f)
            
            credentials = users_config.get("credentials", {}).get("usernames", {})
            for username, user_data in credentials.items():
                tenant_id = user_data.get("tenant_id")
                if tenant_id and tenant_id not in tenants:
                    tenants.append(tenant_id)
        except Exception as exc:
            LOGGER.warning("Failed to load tenants from users.yaml: %s", exc)
    
    return sorted(tenants)

def _sync_memory_to_db(app_state: AppState) -> None:
    """UI wrapper for sync_memory_to_db service."""
    
    with st.spinner("Rebuilding memory from ChromaDB..."):
        result = sync_memory_to_db(app_state)
    
    if result.success:
        st.success(
            f"✅ Synced memory to DB: {result.old_count} → {result.new_count} nodes "
            f"(removed {result.removed} bloated)"
        )
        st.rerun()
    else:
        st.error(f"❌ Sync failed: {result.error}")

def render_document_inspector(app_state):
    """
    Render the Document Inspector section in admin panel.
    """
    # Get tenant to manage
    manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
    
    # Load nodes for managed tenant ONLY
    from .nodes import NodeRepository
    
    repo = NodeRepository(tenant_id=manage_tenant)
    managed_nodes = repo.get_all_nodes()
    
    if not managed_nodes:
        st.info(f"No documents indexed for **{_get_tenant_display_names().get(manage_tenant, manage_tenant)}**.")
        return
    
    st.caption(f"Inspecting: **{_get_tenant_display_names().get(manage_tenant, manage_tenant)}**")
    
    st.markdown("""
    Inspect document processing results, view extracted text, document trees, 
    and identify potential issues.
    """)
    
    # Get all unique sources from managed_nodes
    sources = sorted(set(node.metadata.get("source", "") for node in managed_nodes if node.metadata.get("source")))
    
    if not sources:
        st.warning("No source documents found in index.")
        return
    
    st.markdown(f"**{len(sources)} documents** indexed")
    
    # ===== BULK ANALYSIS =====
    st.markdown("---")
    st.markdown("### 🔍 Bulk Analysis")
    
    if st.button("Identify Problem Documents", type="primary"):
        with st.spinner("Scanning documents..."):
            from app.document_inspector_helpers import identify_problem_documents
            problems = identify_problem_documents(managed_nodes)
        
        if problems:
            st.error(f"Found {len(problems)} document(s) with issues:")
            
            for source, issues in problems:
                with st.expander(f"⚠️ {source}", expanded=False):
                    for issue in issues:
                        st.markdown(f"- {issue}")
        else:
            st.success("✅ All documents look good!")
    
    # ===== DOCUMENT LIST =====
    st.markdown("---")
    st.markdown("### 📋 Document List")
    
    # Document selector
    selected_doc = st.selectbox(
        "Select a document to inspect:",
        options=sources,
        index=0,
        key="doc_inspector_selector"
    )
    
    if not selected_doc:
        return
    
    # Get metrics for selected document
    from app.document_inspector_helpers import get_document_metrics
    metrics = get_document_metrics(selected_doc, managed_nodes)
    
    if not metrics:
        st.error(f"Could not load metrics for: {selected_doc}")
        return
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", metrics.extraction_status)
    
    with col2:
        st.metric("Chunks", metrics.num_chunks)
    
    with col3:
        st.metric("Sections", metrics.num_sections)
    
    with col4:
        coverage_display = f"{metrics.validation_coverage * 100:.1f}%"
        st.metric("Coverage", coverage_display)
    
    # File info
    size_kb = metrics.file_size_bytes / 1024
    size_display = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
    
    st.caption(f"📄 File size: {size_display} • Text: {metrics.text_length:,} chars")
    
    # ===== ACTION BUTTONS =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_text = st.button("👁️ View Text Extract", use_container_width=True)
    
    with col2:
        view_tree = st.button("🌲 View Document Tree", use_container_width=True)
    
    with col3:
        view_extraction = st.button("📊 View Extraction Data", use_container_width=True)
    
    # ===== VIEW TEXT EXTRACT =====
    if view_text:
        from app.document_inspector_helpers import load_cached_text
        
        with st.spinner("Loading cached text..."):
            text = load_cached_text(selected_doc)
        
        if text:
            st.markdown("#### 📄 Cached Text Extract")
            st.caption(f"Length: {len(text):,} characters")
            
            # Show first 5000 chars
            preview_length = 5000
            preview_text = text[:preview_length]
            
            st.text_area(
                "Content Preview (first 5,000 chars)",
                value=preview_text,
                height=400,
                disabled=True,
                key=f"text_preview_{selected_doc}"
            )
            
            if len(text) > preview_length:
                st.caption(f"... +{len(text) - preview_length:,} more characters")
            
            # Option to download full text
            st.download_button(
                label="💾 Download Full Text",
                data=text,
                file_name=f"{Path(selected_doc).stem}_extract.txt",
                mime="text/plain"
            )
        else:
            st.error("❌ No cached text found for this document.")
    
    # ===== VIEW DOCUMENT TREE =====
    if view_tree:
        from app.document_inspector_helpers import load_document_tree, format_tree_visually, calculate_section_token_sizes
        
        with st.spinner("Loading document tree..."):
            doc_id = Path(selected_doc).stem
            tree = load_document_tree(doc_id)
        
        if tree:
            st.markdown("#### 🌲 Document Tree Structure")
            
            # Calculate section token sizes
            section_tokens = calculate_section_token_sizes(tree, app_state.nodes)
            
            # Identify mega-sections
            mega_sections = {sid: tokens for sid, tokens in section_tokens.items() if tokens > 15000}
            
            if mega_sections:
                st.warning(f"⚠️ Found {len(mega_sections)} mega-section(s) (>15K tokens):")
                for sid, tokens in mega_sections.items():
                    st.markdown(f"- **{sid}**: {tokens:,} tokens")
            
            # Display formatted tree
            formatted_tree = format_tree_visually(tree)
            
            # Use code block for better formatting (no markdown interpretation)
            st.code(formatted_tree, language=None)
            
            # Option to download JSON
            import json
            tree_json = json.dumps(tree, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="💾 Download Tree JSON",
                data=tree_json,
                file_name=f"{doc_id}_tree.json",
                mime="application/json"
            )
        else:
            st.error("❌ No document tree found for this document.")
    
    # ===== VIEW EXTRACTION DATA =====
    if view_extraction:
        from app.document_inspector_helpers import load_extraction_data
        
        with st.spinner("Loading Gemini extraction data..."):
            extraction = load_extraction_data(selected_doc)
        
        if extraction:
            st.markdown("#### 📊 Gemini Extraction Data")
            
            # Show key metadata
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Metadata:**")
                
                # Check if corrections exist and prefer those
                corrections = extraction.get('corrections', {})
                
                title = extraction.get('title', 'N/A')
                doc_type = corrections.get('doc_type') or extraction.get('doc_type', 'N/A')
                form_num = corrections.get('form_number') or extraction.get('form_number', 'N/A')
                category = corrections.get('form_category_name') or extraction.get('category', 'N/A')
                ownership = corrections.get('tenant_id') or extraction.get('tenant_id', 'N/A')
                
                st.markdown(f"- **Title:** {title}")
                st.markdown(f"- **Type:** {doc_type}")
                st.markdown(f"- **Form #:** {form_num}")
                st.markdown(f"- **Category:** {category}")
                st.markdown(f"- **Ownership:** {ownership}")
                
                if corrections:
                    st.info("ℹ️ Some fields have been corrected")
            
            with col2:
                st.markdown("**Validation:**")
                
                validation = extraction.get('validation', {})
                validation_error = extraction.get('validation_error')
                
                # Check if this is a form/checklist
                corrections = extraction.get('corrections', {})
                doc_type = corrections.get('doc_type') or extraction.get('doc_type', 'N/A')
                is_form_or_checklist = doc_type.upper() in ("FORM", "CHECKLIST")
                
                if validation and isinstance(validation, dict):
                    # Use ngram_coverage as primary metric (0-1 scale, convert to %)
                    ngram_cov = validation.get('ngram_coverage', 0) * 100
                    word_cov = validation.get('word_coverage', 0) * 100
                    halluc_rate = validation.get('hallucination_rate', 0) * 100
                    length_ratio = validation.get('length_ratio', 0)
                    
                    st.markdown(f"- **N-gram Coverage:** {ngram_cov:.1f}%")
                    st.markdown(f"- **Word Coverage:** {word_cov:.1f}%")
                    st.markdown(f"- **Hallucination Rate:** {halluc_rate:.2f}%")
                    st.markdown(f"- **Length Ratio:** {length_ratio:.2f}")
                    
                    if is_form_or_checklist and ngram_cov < 85:
                        st.info("ℹ️ Low coverage is normal for forms/checklists (empty fields discarded)")
                    elif validation_error:
                        st.warning(f"⚠️ {validation_error}")
                else:
                    st.markdown("- **Coverage:** N/A")
                    st.markdown("- **Validation:** No data")
            
            # Show sections
            sections = extraction.get('sections', [])
            st.markdown(f"**Sections:** {len(sections)}")
            
            if sections:
                with st.expander("View All Sections", expanded=False):
                    for i, section in enumerate(sections, 1):
                        section_name = section.get('name', 'Unnamed')
                        content_len = len(section.get('content', ''))
                        st.markdown(f"{i}. **{section_name}** ({content_len:,} chars)")
            
            # Debug: Show raw JSON structure
            with st.expander("🔍 Debug: View Raw JSON", expanded=False):
                st.json(extraction)
            
            # Option to download full extraction
            import json
            extraction_json = json.dumps(extraction, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="💾 Download Extraction JSON",
                data=extraction_json,
                file_name=f"{Path(selected_doc).stem}_extraction.json",
                mime="application/json"
            )
        else:
            st.error("❌ No Gemini extraction found for this document.")

def rebuild_trees_only(app_state: AppState) -> None:
    """UI wrapper for rebuild_document_trees service."""
    
    with st.spinner("🌲 Rebuilding document trees from cache..."):
        result = rebuild_document_trees(app_state)
    
    if result.success:
        success_msg = f"✅ Rebuilt {result.trees_built} document trees!"
        if result.files_skipped > 0:
            success_msg += f" ({result.files_skipped} files skipped)"
        st.success(success_msg)
        
        if result.errors:
            with st.expander("⚠️ Errors during tree building"):
                for error in result.errors:
                    st.text(error)
    else:
        st.error(f"❌ {result.error}")


def _stage_icon(status: StageStatus) -> str:
    """Get icon for stage status."""
    icons = {
        StageStatus.SUCCESS: "✅",
        StageStatus.WARNING: "⚠️",
        StageStatus.FAILED: "❌",
        StageStatus.SKIPPED: "⏭️",
        StageStatus.PENDING: "⏳",
    }
    return icons.get(status, "❓")

def detect_table_in_stream(text_buffer: str) -> bool:
    """
    Detect if response contains a Markdown table.
    
    Looks for patterns like:
    - Multiple lines with pipe characters (|)
    - Header separator line (| --- | --- |)
    - At least 2 rows (header + separator + data)
    
    Args:
        text_buffer: Accumulated text from stream
    
    Returns:
        True if table detected, False otherwise
    """
    lines = text_buffer.split('\n')
    
    # Count lines with pipe characters
    pipe_lines = [line for line in lines if '|' in line and line.strip().startswith('|')]
    
    if len(pipe_lines) < 2:
        return False  # Need at least header + separator
    
    # Check for separator line (contains hyphens between pipes)
    has_separator = any(
        '|' in line and '-' in line and line.count('-') >= 3
        for line in lines
    )
    
    # Table detected if we have multiple pipe lines AND a separator
    return len(pipe_lines) >= 2 and has_separator

def render_processing_status_table(report: ProcessingReport) -> None:
    """Display processing status table after index build/sync.
    
    Shows:
    - Summary metrics
    - Detailed table of all files with stage-by-stage status
    - Expandable details for files with issues
    
    Args:
        report: ProcessingReport from build_index_from_library_parallel
    """
    
    with st.expander("📊 Processing Status Report", expanded=True):
        # Summary stats
        st.markdown("### Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Files", report.total_files)
        with col2:
            st.metric("✅ Successful", report.successful)
        with col3:
            st.metric("⚠️ Warnings", report.warnings)
        with col4:
            st.metric("❌ Failed", report.failed)
        with col5:
            if report.total_duration_sec:
                st.metric("⏱️ Duration", f"{report.total_duration_sec:.1f}s")
        
        st.markdown("---")
        
        # Detailed table
        if report.file_statuses:
            st.markdown("### Detailed Status")
            
            # Create dataframe for display
            table_data = []
            for status in report.file_statuses:
                # Determine row color based on overall status
                row_data = {
                    "📄 File": status.filename,
                    "Status": status.status_emoji,
                    "Parse": _stage_icon(status.parsing.status),
                    "Extract": _stage_icon(status.extraction.status),
                    "Validate": _stage_icon(status.validation.status),
                    "Embed": _stage_icon(status.embedding.status),
                    "Chunks": str(status.chunks_created) if status.chunks_created else "-",
                    "Size": _format_file_size(status.file_size_bytes) if status.file_size_bytes else "-",
                }
                
                # Add cache indicator
                if status.used_cache:
                    row_data["📄 File"] = f"{status.filename} 💾"
                
                table_data.append(row_data)
            
            # Display table
            st.dataframe(
                table_data,
                width="stretch",
                height=min(400, len(table_data) * 35 + 38),  # Dynamic height
                hide_index=True,
            )
            
            # Legend
            with st.expander("ℹ️ Legend"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Status Icons:**
                    - ✅ Success
                    - ⚠️ Warning
                    - ❌ Failed
                    - ⏭️ Skipped
                    - ⏳ Pending
                    """)
                with col2:
                    st.markdown("""
                    **Indicators:**
                    - 💾 Used cached extraction
                    - Chunks: Number of text chunks created
                    - Size: Original file size
                    """)
            
            st.markdown("---")
            
            # Show details for failed/warning files
            problem_files = [
                s for s in report.file_statuses 
                if s.overall_status in [StageStatus.FAILED, StageStatus.WARNING]
            ]
            
            if problem_files:
                st.markdown("### ⚠️ Files Requiring Attention")
                st.caption(f"{len(problem_files)} file(s) with issues")
                
                for status in problem_files:
                    with st.expander(f"{status.status_emoji} {status.filename}", expanded=False):
                        # Show issue in each stage
                        for stage_name in ["parsing", "extraction", "validation", "embedding"]:
                            stage_result = getattr(status, stage_name)
                            
                            if stage_result.status == StageStatus.FAILED:
                                st.error(f"**{stage_name.title()}:** {stage_result.message}")
                                if stage_result.details:
                                    with st.expander("📋 Technical Details"):
                                        st.json(stage_result.details)
                            
                            elif stage_result.status == StageStatus.WARNING:
                                st.warning(f"**{stage_name.title()}:** {stage_result.message}")
                                if stage_result.details:
                                    with st.expander("📋 Technical Details"):
                                        st.json(stage_result.details)
                        
                        # Show file info
                        if status.file_size_bytes:
                            st.info(f"**File Size:** {_format_file_size(status.file_size_bytes)}")
                        if status.chunks_created:
                            st.info(f"**Chunks Created:** {status.chunks_created}")
            
            else:
                st.success("✅ All files processed successfully!")
        
        else:
            st.info("No file status information available.")


def render_admin_index_management(app_state) -> None:
    """Render the index management section of admin panel.
    
    This replaces the sidebar index controls and puts them in main area.
    Includes the processing status table display.
    
    Args:
        app_state: AppState instance
    """
    st.header("📚 Document Library Management")
    
    st.markdown("""
    Build or sync the document index to process maritime documents.
    The index powers the search and retrieval system.
    """)
    
    # Build controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        clear_cache = st.checkbox(
            "🔄 Clear Gemini cache and re-extract all files",
            value=False,
            help="Force re-extraction of all documents (slower but ensures fresh data)"
        )
        
        if clear_cache:
            st.warning("⚠️ All files will be re-extracted via Gemini API. This may take several minutes.")
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        
        if st.button(
            "🔨 Rebuild Index", 
            type="primary", 
            use_container_width=True,
            help="Process all documents and rebuild the search index"
        ):
            logged_in_tenant = st.session_state.get("tenant_id", "shared")
            rebuild_index_parallel_execute(app_state, clear_cache, tenant_id=logged_in_tenant)

    st.divider()
    
    # Sync controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Sync Library** checks for new, modified, or deleted files and updates the index incrementally.
        This is faster than a full rebuild.
        """)
    
    with col2:
        if st.button(
            "🔄 Sync Library",
            use_container_width=True,
            help="Incrementally update index with file changes"
        ):
            sync_library(app_state)
    
    st.divider()
    
    # Show last processing report if available
    if "last_processing_report" in st.session_state:
        report = st.session_state["last_processing_report"]
        render_processing_status_table(report)
    else:
        with st.expander("📊 Processing Status Report"):
            st.info("No recent processing report available. Build or sync the index to see processing status.")
            
            # Check if saved report exists on disk
            from .config import AppConfig
            config = AppConfig.get()
            report_path = config.paths.cache_dir / "last_processing_report.json"
            
            if report_path.exists():
                if st.button("📂 Load Last Report from Disk"):
                    from .processing_status import load_processing_report
                    report = load_processing_report(report_path)
                    if report:
                        st.session_state["last_processing_report"] = report
                        st.rerun()
                    else:
                        st.error("Failed to load report")

def _rerun_app() -> None:
    """Compat helper for rerunning the Streamlit script across versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - legacy Streamlit
        st.experimental_rerun()


def _reset_chat_state(app_state: AppState) -> None:
    """Clear chat history and feedback toggles."""
    app_state.reset_session()
    for key in list(st.session_state.keys()):
        if key.startswith("correction_toggle_") or key.startswith("correction_text_") or key.startswith("confirm_"):
            st.session_state.pop(key)


def _format_file_size(size_bytes: int) -> str:
    """Return a human-friendly file size string."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}TB"


_GITHUBISH_CSS = """
:root { color-scheme: dark; }
body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif; margin: 0; padding: 0; background: #040b14; color: #f2f7ff; }
.container { max-width: 900px; margin: 2rem auto; padding: 2.5rem; background: linear-gradient(155deg, rgba(5, 20, 35, 0.96), rgba(2, 12, 22, 0.9)); border: 1px solid rgba(120, 210, 255, 0.12); border-radius: 24px; box-shadow: 0 32px 70px rgba(0, 0, 0, 0.45); }
.markdown-body { line-height: 1.75; font-size: 18px; color: #f2f7ff; }
.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 { margin-top: 1.5em; color: #c6e6ff; }
.markdown-body h3 { border-bottom: 1px solid rgba(110, 195, 255, 0.3); padding-bottom: .3em; }
.markdown-body pre { background: rgba(11, 31, 52, .78); padding: 1rem; overflow: auto; border-radius: 10px; border: 1px solid rgba(116, 197, 255, 0.22); color: #f2f7ff; }
.markdown-body code { padding: .1em .3em; background: rgba(21, 55, 88, .55); border-radius: 4px; color: #f2f7ff; }
.markdown-body blockquote { margin: 1em 0; padding: .5em 1em; border-left: 4px solid rgba(102, 180, 255, .55); background: rgba(8, 26, 44, .72); border-radius: 6px; color: #ddeeff; }
.markdown-body table { border-collapse: collapse; width: 100%; margin: 1em 0; color: #f2f7ff; }
.markdown-body th, .markdown-body td { border: 1px solid rgba(102, 180, 255, .28); padding: .5em .75em; }
.result-markdown { display: flex; flex-direction: column; gap: 1.35rem; color: #f2f7ff; }
.result-markdown h3 { font-size: 1.2rem; margin: 0; color: #c3e7ff; }
.result-markdown h4 { font-size: 1.05rem; margin: 0; color: #9ed4ff; }
.result-markdown p, .result-markdown li { font-size: 1rem; line-height: 1.65rem; color: #f2f7ff; }
.result-markdown .summary-grid { margin-top: 0.65rem; display: grid; grid-template-columns: auto 1fr; row-gap: 0.4rem; column-gap: 0.85rem; }
.result-markdown .summary-row dt { font-weight: 600; color: #7acbff; }
.result-markdown .summary-row dd { margin: 0; color: #f2f7ff; }
.result-markdown .refinement-list { padding-left: 1.2rem; margin: 0.5rem 0 0; color: #f2f7ff; }
.result-markdown .ref-label { font-weight: 600; color: #8dd2ff; }
.result-markdown .sources-list { margin: 0.6rem 0 0; padding-left: 1.2rem; color: #ddeeff; }
.result-markdown .sources-list li { margin-bottom: 0.5rem; }
.result-markdown .answer-body { background: rgba(4, 18, 32, 0.88); border: 1px solid rgba(116, 197, 255, 0.25); border-radius: 18px; padding: 1.2rem 1.3rem; color: #f6fbff; }
.result-markdown .answer-body * { color: inherit; }
.result-markdown .source-location { font-size: 0.85rem; color: #c5daff; }
.result-markdown .confidence-note { font-size: 0.9rem; font-style: italic; color: #e4f3ff; }
.result-markdown .empty-sources { margin: 0; color: #d5e9ff; opacity: 0.75; }
hr { border: 0; border-top: 1px solid rgba(102,180,255,.25); margin: 2rem 0; }
.smallmeta { color: rgba(190, 221, 255, .55); font-size: 12px; margin-top: 2rem; text-align: right; }
.upload-chip-row { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }
.upload-chip { background: rgba(11, 44, 74, 0.8); border: 1px solid rgba(110, 195, 255, 0.3); padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; color: #cce8ff; }
.upload-chip.more { background: rgba(110, 195, 255, 0.2); color: #7acbff; }
.upload-card { background: rgba(8, 26, 44, 0.75); border: 1px solid rgba(110, 195, 255, 0.25); padding: 0.55rem 0.7rem; border-radius: 12px; margin-bottom: 0.4rem; }
.upload-card strong { color: #e4f3ff; }
.upload-card small { display: block; color: rgba(190, 221, 255, 0.6); }
.chat-input-wrapper { position: relative; padding-bottom: 0.75rem; }
.chat-input-wrapper div[data-testid="stChatInput"] { padding-left: 3.2rem; }
.chat-input-wrapper div[data-testid="stFileUploader"] { position: absolute; left: 1.1rem; bottom: calc(env(safe-area-inset-bottom) + 1.1rem); width: 44px; height: 44px; z-index: 10; }
.chat-input-wrapper div[data-testid="stFileUploader"] > label { display: none; }
.chat-input-wrapper div[data-testid="stFileUploaderDropzone"] { border: none; background: rgba(15, 70, 120, 0.85); border-radius: 999px; padding: 0; min-height: 44px; height: 44px; width: 44px; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 16px rgba(10, 30, 55, 0.55); transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease; }
.chat-input-wrapper div[data-testid="stFileUploaderDropzone"] section { display: none; }
.chat-input-wrapper div[data-testid="stFileUploaderDropzone"]::before { content: "\1F4CE"; font-size: 18px; filter: drop-shadow(0 1px 2px rgba(2, 12, 22, 0.7)); }
.chat-input-wrapper div[data-testid="stFileUploaderDropzone"]:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(10, 30, 55, 0.65); background: rgba(20, 90, 150, 0.92); }
.chat-input-wrapper div[data-testid="stFileUploaderDropzone"]:active { transform: translateY(0); box-shadow: 0 2px 10px rgba(6, 20, 36, 0.6); }
.chat-upload-chips { margin-left: 3.6rem; margin-bottom: 0.35rem; display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }
.chat-upload-chips.empty { color: rgba(190, 221, 255, 0.55); font-size: 0.75rem; }
.upload-limit-note { font-size: 0.75rem; color: rgba(190, 221, 255, 0.55); margin-left: 3.6rem; margin-bottom: 0.4rem; }
"""

_HTML_SHELL = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class=\"container\">
    <div class=\"markdown-body\">{body}</div>
    <div class=\"smallmeta\">Exported: {timestamp}</div>
  </div>
</body>
</html>
"""


def compose_result_markdown(result: Dict) -> str:
    """Compose a styled HTML fragment for a chat response."""
    original_query = (result.get("query") or "No original query").strip() or "Untitled query"
    final_query = (result.get("final_query") or "").strip()
    conf_pct = result.get("confidence_pct")
    conf_level = (result.get("confidence_level") or "N/A").strip() or "N/A"
    num_sources = result.get("num_sources")
    attempts = result.get("attempts")
    best_attempt = result.get("best_attempt")
    refinement_history = result.get("refinement_history") or []
    answer = (result.get("answer") or "No answer available.").strip()
    sources = result.get("sources") or []
    confidence_note = (result.get("confidence_note") or "").strip()

    parts: List[str] = ["<div class='result-markdown'>"]

    parts.append("<section class='result-summary'>")
    parts.append("<h3>Query overview</h3>")
    parts.append("<dl class='summary-grid'>")

    def render_summary_row(label: str, value_html: str) -> None:
        parts.append(
            "<div class='summary-row'>"
            f"<dt>{html.escape(label)}:</dt>"
            f"<dd>{value_html}</dd>"
            "</div>"
        )

    conf_pct_str = "N/A" if conf_pct is None else str(conf_pct)
    if isinstance(conf_pct, (int, float)) and not conf_pct_str.endswith('%'):
        conf_pct_str = f"{int(conf_pct)}"
    if conf_pct_str != "N/A" and not conf_pct_str.endswith('%'):
        conf_pct_str = f"{conf_pct_str}%"

    render_summary_row("Query", html.escape(original_query))
    render_summary_row("Confidence", f"{html.escape(conf_pct_str)} ({html.escape(conf_level)})")
    sources_str = "N/A" if num_sources is None else str(num_sources)
    render_summary_row("Sources analysed", html.escape(sources_str))

    if final_query and final_query.lower() != original_query.lower():
        render_summary_row("Final query used", html.escape(final_query))

    if refinement_history:
        total_attempts = attempts or len(refinement_history)
        suffix = f"; best attempt {best_attempt}" if best_attempt else ""
        render_summary_row("Attempts", f"{html.escape(str(total_attempts))}{html.escape(suffix)}")

    parts.append("</dl>")
    parts.append("</section>")

    if refinement_history:
        parts.append("<section class='result-refinement'>")
        parts.append("<h4>Refinement history</h4>")
        parts.append("<ol class='refinement-list'>")
        for entry in refinement_history:
            attempt_no = entry.get("attempt")
            entry_query = (entry.get("query") or "").strip() or "n/a"
            confidence = entry.get("confidence")
            label = "Initial query" if attempt_no in (None, 1) else f"Attempt {attempt_no}"
            confidence_bits = ""
            if confidence is not None:
                try:
                    confidence_bits = f" &middot; {int(confidence)}%"
                except (TypeError, ValueError):
                    confidence_bits = f" &middot; {html.escape(str(confidence))}%"
            parts.append(
                "<li>"
                f"<span class='ref-label'>{label}</span>: "
                f"{html.escape(entry_query)}{confidence_bits}"
                "</li>"
            )
        parts.append("</ol>")
        parts.append("</section>")

    parts.append("<section class='result-answer'>")
    parts.append("<h4>Answer</h4>")
    answer_html = md(answer, extensions=_DEF_EXTS) if answer else "<p>No answer available.</p>"
    parts.append(f"<div class='answer-body'>{answer_html}</div>")
    parts.append("</section>")

    parts.append("<section class='result-sources'>")
    parts.append("<h4>Sources</h4>")
    if sources:
        type_labels = {
            "PROC": "PROCEDURE",
            "FORM": "FORM",
            "REG": "REGULATION",
            "POLICY": "POLICY",
            "REPORT": "REPORT",
            "CHECKLIST": "CHECKLIST",
        }
        parts.append("<ol class='sources-list'>")
        for idx, src in enumerate(sources[:5], 1):
            is_upload = src.get("session_upload")
            source_file = src.get("upload_display_name") if is_upload else src.get("source", "Unknown")
            if not source_file:
                source_file = "Unknown"
            title = src.get("title") or source_file.rsplit('.', 1)[0].replace('_', ' ')
            doc_type = (src.get("doc_type") or "").upper()
            type_label = type_labels.get(doc_type, doc_type)
            if is_upload:
                display_title = src.get("upload_display_name") or title
            else:
                display_title = f"{title} ({type_label})" if type_label else title
            if is_upload:
                display_title = f"{display_title} 📎"

            location_segments: List[str] = []
            hierarchy = src.get("hierarchy")
            if hierarchy:
                location_segments.append(str(hierarchy))
            tab_name = src.get("tab_name")
            if tab_name:
                location_segments.append(f"Tab: {tab_name}")
            section = src.get("section")
            if section and section not in {"Document Content", "Main document"}:
                location_segments.append(str(section))

            form_number = src.get("form_number")
            form_category = src.get("form_category_name")
            if doc_type == "FORM" and form_number:
                form_desc = f"Form {form_number}"
                if form_category:
                    form_desc += f" - {form_category}"
                location_segments = [form_desc]

            location_text = " • ".join(filter(None, location_segments)) or "Main document"
            location_text = location_text.replace("\n", " ").strip()
            truncated = False
            if len(location_text) > 120:
                location_text = location_text[:117].rstrip()
                truncated = True
            location_html = html.escape(location_text)
            if truncated:
                location_html += '&hellip;'

            parts.append(
                "<li>"
                f"<strong>{idx}. {html.escape(display_title)}</strong><br>"
                f"<span class='source-location'>Location: {location_html}</span>"
                "</li>"
            )
        parts.append("</ol>")
    else:
        parts.append("<p class='empty-sources'><em>No sources available.</em></p>")
    parts.append("</section>")

    if confidence_note:
        parts.append(f"<p class='confidence-note'>Note: {html.escape(confidence_note)}</p>")

    parts.append("</div>")
    return "\n".join(parts)


def build_result_export_html(result: Dict, title: str = "RAG Query Result") -> str:
    """Generate an HTML export representation for a query result."""
    html_body = compose_result_markdown(result)
    return _HTML_SHELL.format(
        title=title,
        css=_GITHUBISH_CSS,
        body=html_body,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def build_session_export_html(messages: List[Dict], session_title: str = "Chat Session") -> str:
    """Generate an HTML export for an entire chat session."""
    conversation_parts: List[str] = []
    
    for idx, msg in enumerate(messages):
        if msg["role"] == "user":
            query_text = msg["content"]
            conversation_parts.append(f"<div class='user-message'><h3>Question {idx // 2 + 1}</h3><p>{html.escape(query_text)}</p></div>")
        
        elif msg["role"] == "assistant":
            # Reconstruct result dict for rendering
            result = {
                "query": messages[idx - 1]["content"] if idx > 0 else "Query",
                "answer": msg["content"],
                "confidence_pct": msg.get("confidence_pct", 0),
                "confidence_level": msg.get("confidence_level", "N/A"),
                "confidence_note": msg.get("confidence_note", ""),
                "sources": msg.get("sources", []),
                "num_sources": msg.get("num_sources", 0),
                "retriever_type": msg.get("retriever_type", "unknown"),
            }
            conversation_parts.append(compose_result_markdown(result))
            conversation_parts.append("<hr>")
    
    # Remove last HR
    if conversation_parts and conversation_parts[-1] == "<hr>":
        conversation_parts.pop()
    
    html_body = "\n".join(conversation_parts)
    
    # Add custom CSS for session export
    session_css = _GITHUBISH_CSS + """
.user-message { 
    background: rgba(10, 132, 255, 0.1); 
    border-left: 4px solid rgba(10, 132, 255, 0.5); 
    padding: 1rem; 
    margin: 1rem 0; 
    border-radius: 8px; 
}
.user-message h3 { 
    margin-top: 0; 
    color: #66b4ff; 
    font-size: 1.1rem; 
}
.user-message p { 
    margin: 0.5rem 0 0; 
    color: #f2f7ff; 
}
"""
    
    return _HTML_SHELL.format(
        title=session_title,
        css=session_css,
        body=html_body,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def save_result_as_html(result: Dict, output_file: str = "rag_result.html", title: str = "RAG Query Result") -> str:
    full_html = build_result_export_html(result, title=title)
    out_dir = AppConfig.get().paths.cache_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name, ext = os.path.splitext(output_file)
    counter = 1
    new_output_file = output_file
    while (out_dir / new_output_file).exists():
        new_output_file = f"{base_name}({counter}){ext}"
        counter += 1

    export_path = out_dir / new_output_file
    export_path.write_text(full_html, encoding="utf-8")
    return str(export_path)


def load_or_warn(app_state: AppState) -> None:
    """Load cached index if available."""
    with st.spinner("Loading cached index..."):
        if app_state.ensure_index_loaded():
            st.success(f"✅ Loaded cached index with {len(app_state.nodes)} chunks")
            LOGGER.info("Loaded cached index: %d nodes", len(app_state.nodes))
        else:
            st.warning("⚠️ No cached index found. Please build the index first.")
            LOGGER.warning("No cached index found")


def rebuild_index(app_state: AppState) -> None:
    with st.spinner("Building index from library (this may take several minutes)..."):
        try:
            nodes, index = build_index_from_library()
            app_state.nodes = nodes
            app_state.index = index
            app_state.invalidate_node_map_cache()  # Clear stale cache
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
            app_state.ensure_manager().nodes = nodes
            st.success(f"✅ Rebuilt index with {len(nodes)} chunks.")
            LOGGER.info("Index rebuilt successfully: %d nodes", len(nodes))
        except Exception as exc:
            LOGGER.exception("Failed to rebuild index")
            st.error(f"❌ Failed to rebuild index: {exc}")

def rebuild_index_parallel_execute(app_state: AppState, clear_gemini_cache: bool = False, tenant_id: str = "shared", doc_type_override: str | None = None) -> None:
    """Execute parallel index rebuild with progress tracking.
    
    Args:
        app_state: Application state
        clear_gemini_cache: If True, re-extract all files via Gemini
    """
    # Create progress containers
    st.write("### 🚀 Parallel Processing Progress")
    
    phase1_container = st.container()
    phase2_container = st.container()

    with phase1_container:
        st.write("**Phase 1:** Extracting documents (Gemini)")
        phase1_progress = st.progress(0.0)
        phase1_status = st.empty()
    
    with phase2_container:
        st.write("**Phase 2:** Generating embeddings")
        phase2_progress = st.progress(0.0)
        phase2_status = st.empty()
    
    overall_status = st.empty()
    
    # Progress callback
    def progress_callback(phase: str, current: int, total: int, item: str) -> None:
        """Update progress bars based on current phase."""
        progress_pct = current / total if total > 0 else 0.0
        
        if phase == "extracting":
            phase1_progress.progress(progress_pct)
            phase1_status.text(f"Extracting: {current}/{total} - {item}")
        
        elif phase == "embedding":
            phase2_progress.progress(progress_pct)
            phase2_status.text(f"Embedding: {current}/{total}")
    
    try:
        if clear_gemini_cache:
            overall_status.warning("⚠️ Will re-extract all files (Gemini cache cleared)")
        else:
            overall_status.info("⏳ Starting parallel rebuild (using cached extractions)...")
        
        # Run parallel processing
        nodes, index, report = build_index_from_library_parallel(
            progress_callback=progress_callback,
            clear_gemini_cache=clear_gemini_cache,
            tenant_id=tenant_id,
            doc_type_override=doc_type_override
        )

        # Store report in session state
        st.session_state["last_processing_report"] = report

        # Update app state
        app_state.nodes = nodes
        app_state.index = index
        app_state.invalidate_node_map_cache()  # Clear stale cache
        app_state.vector_retriever = None
        app_state.bm25_retriever = None
        app_state.manager = None
        app_state.ensure_retrievers()
        app_state.ensure_manager().nodes = nodes
        
        # Success
        cache_msg = " (all files re-extracted)" if clear_gemini_cache else ""
        overall_status.success(
            f"✅ Rebuilt index with {len(nodes)} chunks using parallel processing!{cache_msg}"
        )
        LOGGER.info("Parallel index rebuild complete: %d nodes", len(nodes))

        # Display processing status table
        if report:
            render_processing_status_table(report)
        
        # Mark progress complete
        phase1_progress.progress(1.0)
        phase2_progress.progress(1.0)
        
    except Exception as exc:
        LOGGER.exception("Failed to rebuild index with parallel processing")
        overall_status.error(f"❌ Failed to rebuild index: {exc}")


def rebuild_index_parallel(app_state: AppState) -> None:
    """Compatibility wrapper - calls rebuild_index_parallel_execute with default settings."""
    rebuild_index_parallel_execute(app_state, clear_gemini_cache=False)

def sync_library_with_ui(app_state: AppState, tenant_id: str = "shared", doc_type_override: str | None = None) -> None:
    """
    Execute sync_library with progress tracking UI.
    
    Matches the UI pattern from rebuild_index_parallel_execute.
    """
    
    # Create progress containers
    st.write("#### 🔄 Syncing...")
    
    phase1_container = st.container()
    phase2_container = st.container()
    
    with phase1_container:
        st.write("**Phase 1:** Extracting modified documents (Gemini)")
        phase1_progress = st.progress(0.0)
        phase1_status = st.empty()
    
    with phase2_container:
        st.write("**Phase 2:** Generating embeddings")
        phase2_progress = st.progress(0.0)
        phase2_status = st.empty()
    
    overall_status = st.empty()
    
    # Progress callback
    def progress_callback(phase: str, current: int, total: int, item: str) -> None:
        """Update progress bars based on current phase."""
        progress_pct = current / total if total > 0 else 0.0
        
        if phase == "extracting":
            phase1_progress.progress(progress_pct)
            phase1_status.text(f"Extracting: {current}/{total} - {item}")
        
        elif phase == "embedding":
            phase2_progress.progress(progress_pct)
            phase2_status.text(f"Embedding: {current}/{total}")
    
    try:
        overall_status.info("⏳ Starting incremental sync...")
        
        # Run sync with progress callback
        manager = app_state.ensure_manager()
        manager.tenant_id = tenant_id
        manager.nodes = app_state.nodes
        
        sync_result, report = manager.sync_library(
            app_state.index,
            force_retry_errors=True,
            progress_callback=progress_callback,
            doc_type_override=doc_type_override
        )
        
        # Store report in session state
        st.session_state["last_processing_report"] = report
        
        # Update app state
        app_state.nodes = manager.nodes
        app_state.invalidate_node_map_cache()
        app_state.vector_retriever = None
        app_state.bm25_retriever = None
        app_state.ensure_retrievers()
        
        # Success message
        overall_status.success(
            f"✅ Sync complete! Added {len(sync_result.added)}, "
            f"modified {len(sync_result.modified)}, deleted {len(sync_result.deleted)}."
        )
        LOGGER.info("Library synced: +%d, ~%d, -%d", 
                   len(sync_result.added), len(sync_result.modified), len(sync_result.deleted))
        
        # Display processing status table
        #if report:
        #    render_processing_status_table(report)
        
        # Mark progress complete
        phase1_progress.progress(1.0)
        phase2_progress.progress(1.0)
        
    except Exception as exc:
        LOGGER.exception("Failed to sync library")
        overall_status.error(f"❌ Failed to sync library: {exc}")

def sync_library(app_state: AppState) -> None:
    """
    Simple sync without progress UI (for backward compatibility).
    
    Used in contexts where progress bars aren't appropriate.
    """
    manager = app_state.ensure_manager()
    manager.nodes = app_state.nodes
    
    with st.spinner("Syncing library..."):
        try:
            # Call enhanced sync without progress callback
            sync_result, report = manager.sync_library(
                app_state.index,
                force_retry_errors=True,
                progress_callback=None
            )
            
            st.success(
                f"✅ Sync complete. Added {len(sync_result.added)}, "
                f"modified {len(sync_result.modified)}, deleted {len(sync_result.deleted)}."
            )
            
            # Update app state
            app_state.nodes = manager.nodes
            app_state.invalidate_node_map_cache()
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
            
            # Store report if available
            if report:
                st.session_state["last_processing_report"] = report
            
            LOGGER.info("Library synced: +%d, ~%d, -%d", 
                       len(sync_result.added), len(sync_result.modified), len(sync_result.deleted))
            
        except Exception as exc:
            LOGGER.exception("Failed to sync library")
            st.error(f"❌ Failed to sync library: {exc}")


def render_feedback_stats_panel(app_state: AppState) -> None:
    analysis = app_state.feedback_system.analyze_feedback()
    if "error" in analysis:
        st.info("No feedback data yet. Start using the system!")
        return

    main_stats = f"""
    <div class='feedback-stats'>
        <h4>Feedback Analytics</h4>
        <ul>
            <li>Total feedback: {analysis['total_feedback']}</li>
            <li>Satisfaction rate: {analysis['satisfaction_rate']:.1f}%</li>
            <li>Incorrect rate: {analysis['incorrect_rate']:.1f}%</li>
        </ul>
    </div>
    """
    #st.markdown(main_stats, unsafe_allow_html=True)

    # Confidence calibration
    cal = analysis["confidence_calibration"]
    cal_html = f"""
    <div class='confidence-calibration'>
        <h4>Confidence calibration</h4>
        <ul>
            <li>High confidence correct: {cal['high_conf_accurate']}</li>
            <li>High confidence wrong: {cal['high_conf_wrong']}</li>
            <li>Overconfidence rate: {cal['overconfidence_rate']:.1f}%</li>
            <li>Low confidence correct: {cal['low_conf_accurate']}</li>
            <li>Low confidence wrong: {cal['low_conf_wrong']}</li>
            <li>Underconfidence rate: {cal['underconfidence_rate']:.1f}%</li>
        </ul>
    </div>
    """
    #st.markdown(cal_html, unsafe_allow_html=True)

    # Query refinement
    ref = analysis["query_refinement"]
    ref_html = f"""
    <div class='query-refinement'>
        <h4>Query refinement</h4>
        <ul>
            <li>Queries refined: {ref['total_refined']}</li>
            <li>Refinement success: {ref['refinement_success_rate']:.1f}%</li>
        </ul>
    </div>
    """
    #st.markdown(ref_html, unsafe_allow_html=True)

    # Recommendations
    if analysis["recommendations"]:
        rec_items = "".join(f"<li>{rec}</li>" for rec in analysis["recommendations"])
        rec_html = f"""
        <div class='recommendations'>
            <h4>Recommendations</h4>
            <ul>{rec_items}</ul>
        </div>
        """
    else:
        rec_html = ""
        #st.markdown(rec_html, unsafe_allow_html=True)

    # Recent problem queries
    problems = app_state.feedback_system.get_problem_queries(limit=3)
    if problems:
        problem_items = []
        for idx, item in enumerate(problems, 1):
            item_html = f"<li>{idx}. \"{item['query']}\" - {item['confidence_pct']}% ({item['confidence_level']})"
            if item.get("correction"):
                item_html += f"<br><span style='margin-left: 20px;'>User feedback: {item['correction'][:100]}</span>"
            item_html += "</li>"
            problem_items.append(item_html)
        problems_html = f"""
        <div class='problem-queries'>
            <h4>Recent problem queries</h4>
            <ul>{"".join(problem_items)}</ul>
        </div>
        """
    else:
        problems_html = ""

    full_html = f"""
    <div class='sidebar-panel docs'>
        {main_stats}
        {cal_html}
        {ref_html}
        {rec_html}
        {problems_html}
    </div>
    """

    # Render everything as one HTML block
    st.markdown(full_html, unsafe_allow_html=True)


def render_chat_message_with_feedback(app_state: AppState, result: Dict, message_index: int) -> None:
    """Render a single assistant message with inline feedback controls."""
    
    # Extract key info
    answer = result.get("answer", "No answer available.")
    conf_pct = result.get("confidence_pct", 0)
    conf_level = result.get("confidence_level", "N/A")
    num_sources = result.get("num_sources", 0)
    sources = result.get("sources", [])
    confidence_note = result.get("confidence_note", "")
    
    # Render the answer
    with st.chat_message("assistant"):
        # Process answer to make inline citations smaller and italic
        import re
        
        # Pattern to match citations like [3, P 020 - PPE Inventory > PPE Inventory Table]
        citation_pattern = r'\[([^\]]+)\]'
        
        def format_citation(match):
            citation_text = match.group(1)
            return f"<span style='font-size: 0.75em; font-style: italic; opacity: 0.7;'>[{citation_text}]</span>"
        
        formatted_answer = re.sub(citation_pattern, format_citation, answer)
        st.markdown(formatted_answer, unsafe_allow_html=True)
        
        # Confidence badge (single emoji, removed duplicate)
        confidence_color = {
            "HIGH 🟢": "🟢",
            "MEDIUM 🟡": "🟡", 
            "LOW 🔴": "🔴"
        }
        badge_emoji = confidence_color.get(conf_level, "⚪")
        # Remove emoji from conf_level to avoid duplication
        conf_level_text = conf_level.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()
        
        # Build caption with context info
        caption_parts = [f"{badge_emoji} **Confidence:** {conf_pct}% ({conf_level_text})", f"**Sources:** {num_sources}"]
        
        # Add context mode indicator if applicable
        if result.get("context_mode"):
            context_turn = result.get("context_turn", 0)
            caption_parts.append(f"💬 **Turn:** {context_turn}")
            
        # Add reset notification if applicable
        if result.get("context_reset_note"):
            st.info(result["context_reset_note"])
        
        st.caption(" • ".join(caption_parts))
        
        # Expandable sources
        if sources:
            with st.expander("📚 View sources", expanded=False):
                for idx, src in enumerate(sources[:5], 1):
                    source_file = src.get("source", "Unknown")
                    title = src.get("title") or source_file.rsplit('.', 1)[0].replace('_', ' ')
                    section = src.get("section", "Main document")
                    st.markdown(f"**{idx}. {title}**")
                    st.caption(f"└─ {section}")
        
        if confidence_note:
            st.info(confidence_note)
        
        # Feedback buttons - tighter spacing with custom CSS
        st.markdown("""
        <style>
        div[data-testid="column"] {
            padding: 0 !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            gap: 0.25rem !important;
        }
        /* Remove all button borders and boxes */
        .stDownloadButton button, 
        div[data-testid="baseButton-secondary"],
        button[kind="secondary"] {
            padding: 0.25rem 0.5rem !important;
            min-width: 2.5rem !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        .stDownloadButton button:hover, 
        div[data-testid="baseButton-secondary"]:hover,
        button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            box-shadow: none !important;
        }
        
        /* FORCE horizontal layout on mobile - prevent stacking */
        @media (max-width: 768px) {
            div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
            }
            div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) > div[data-testid="column"] {
                flex: 0 0 auto !important;
                width: auto !important;
                min-width: auto !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        cols = st.columns([0.5, 0.5, 0.5, 0.5, 8])
        
        # Export button
        export_html = build_result_export_html(result)
        file_name = f"query_result_{message_index}.html"
        with cols[0]:
            st.download_button(
                "📥",
                data=export_html,
                file_name=file_name,
                mime="text/html",
                key=f"export_{message_index}",
                help="Download as HTML"
            )
        
        # Helpful button
        with cols[1]:
            if st.button("👍", key=f"helpful_{message_index}", help="Helpful"):
                app_state.feedback_system.log_feedback(result, "helpful")
                st.toast("✅ Feedback recorded", icon="👍")
        
        # Not helpful button
        with cols[2]:
            if st.button("👎", key=f"not_helpful_{message_index}", help="Not helpful"):
                app_state.feedback_system.log_feedback(result, "not_helpful")
                st.toast("📝 Feedback recorded", icon="👎")
        
        # Report issue button
        correction_toggle_key = f"correction_toggle_{message_index}"
        with cols[3]:
            if st.button("⚠️", key=f"incorrect_{message_index}", help="Report issue"):
                st.session_state[correction_toggle_key] = not st.session_state.get(correction_toggle_key, False)
        
        # Correction input (shown when toggled)
        if st.session_state.get(correction_toggle_key, False):
            correction = st.text_area(
                "What was wrong? What should the answer be?",
                key=f"correction_text_{message_index}",
                placeholder="Describe the issue...",
                height=100
            )
            if st.button("Submit correction", key=f"submit_correction_{message_index}"):
                if correction.strip():
                    app_state.feedback_system.log_feedback(result, "incorrect", correction.strip())
                    st.toast("✅ Correction submitted", icon="⚠️")
                    st.session_state[correction_toggle_key] = False
                    st.session_state.pop(f"correction_text_{message_index}", None)
                    _rerun_app()
                else:
                    st.warning("Please provide details before submitting.")


def render_app(
    app_state: AppState,
    *,
    read_only_mode: bool = False,
) -> None:
    st.set_page_config(
        page_title="MA.D.ASS: The Maritime Documentation Assistant",
        page_icon="⚓",
        layout="centered",  # Changed from "wide" to enable max-width control
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for centered chat layout (60% width)
    st.markdown("""
    <style>
    /* Hide Streamlit settings bar and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Override Streamlit's default block-container styling */
    section.main > div.block-container {
        max-width: 800px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1rem !important;
        margin: 0 auto !important;
    }

    /* Center and limit chat input width to match */
    div[data-testid="stChatInput"] {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* Fix floating input container if it exists */
    .stChatFloatingInputContainer {
        max-width: 800px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
    }

    /* Add padding to make input bar thicker */
    div[data-testid="stChatInput"] textarea {
        padding-top: 12px !important;
        padding-bottom: 12px !important;
    }

    /* Center the send button vertically - multiple approaches */
    div[data-testid="stChatInput"] button {
        align-self: center !important;
        margin-top: auto !important;
        margin-bottom: auto !important;
    }

    /* Alternative selector for send button icon */
    div[data-testid="stChatInput"] button[data-testid="baseButton-primary"] {
        align-self: center !important;
    }

    /* Ensure chat messages stay within bounds */
    div[data-testid="stChatMessageContainer"] {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* Center title and caption, move title higher */
    section.main h1 {
        text-align: center;
        margin-top: -2rem !important;
    }

    section.main > div > div > div > p {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a centered container for the title and caption
    title_html = """
    <div style="text-align: center;">
        <h1>⚓ MA.D.ASS</h1>
        <h2>The Maritime Documentation Assistant</h2>
        <h5 style="margin-top: -0.3em; color: #666;">Intelligent document search powered by dreams of electric sheep</h5>
    </div>
    """

    st.markdown(title_html, unsafe_allow_html=True)

    #st.title("⚓ MA.D.ASS: The Maritime Documentation Assistant")
    #st.caption("Intelligent document search powered by the dreams of electric sheep")

    # Ensure index is loaded
    if not app_state.ensure_index_loaded():
        st.error("⚠️ **No index found.** Please build the index first using the sidebar controls.")
        if not read_only_mode:
            if st.button("🔨 Build Index Now"):
                rebuild_index_parallel(app_state)
                _rerun_app()
        st.stop()
    
    # Ensure retrievers are ready
    if not app_state.is_ready_for_queries():
        st.error("⚠️ **System not ready.** Retrievers failed to initialize.")
        st.stop()
    
    # Ensure we have a session
    if not app_state.current_session_id:
        # Check if there are existing sessions
        manager = app_state.ensure_session_manager()
        sessions = manager.list_sessions(limit=1)
        
        if sessions:
            # Load most recent session
            app_state.switch_session(sessions[0].session_id)
        else:
            # Create first session
            app_state.create_new_session()


    # Load chat history
    app_state.ensure_history_loaded()
    
    # Session state defaults
    st.session_state.setdefault("retrieval_method", "hybrid")
    st.session_state.setdefault("rerank_enabled", True)
    st.session_state.setdefault("fortify_option", False)
    st.session_state.setdefault("auto_refine_option", False)
    st.session_state.setdefault("use_context", True)  # Default ON for context-aware chat
    st.session_state.setdefault("use_hierarchical", True)  # Default ON for hierarchical retrieval

    # Sidebar configuration
    with st.sidebar:
        # Admin/User mode toggle button at the top
        if read_only_mode:
            # In viewer mode - show Admin button
            if st.button("🔓 Admin Panel", use_container_width=True, key="mode_toggle", type="primary"):
                st.query_params["read_only"] = "false"
                _rerun_app()
        else:
            # In admin mode - show User button
            if st.button("👤 User", use_container_width=True, key="mode_toggle", type="secondary"):
                st.query_params["read_only"] = "true"
                _rerun_app()

        st.markdown("---")


        # Custom CSS
        st.markdown("""
        <style>

        /* Download button in dialog */
        [data-testid="stDialog"] .stDownloadButton button {
            background: rgba(10, 132, 255, 0.3) !important;
            border: 1px solid rgba(10, 132, 255, 0.5) !important;
            color: white !important;
        }

        [data-testid="stDialog"] .stDownloadButton button:hover {
            background: rgba(10, 132, 255, 0.5) !important;
        }


        /* Restore button styling for sidebar buttons only */
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[kind="secondary"] {
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            background: rgba(10, 132, 255, 0.1) !important;
            box-shadow: none !important;
            padding: 0.5rem 1rem !important;
        }
        
        section[data-testid="stSidebar"] button[kind="primary"]:hover,
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(10, 132, 255, 0.2) !important;
            border-color: rgba(10, 132, 255, 0.5) !important;
        }

        /* ── Doc file list: tertiary buttons as text links ── */
        section[data-testid="stSidebar"] button[kind="tertiary"] {
            border: none !important;
            background: transparent !important;
            padding: 0px 15px !important;
            box-shadow: none !important;
            height: auto !important;
            min-height: 0 !important;
            line-height: 2 !important;
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
            border-radius: 4px !important;
        }

        section[data-testid="stSidebar"] button[kind="tertiary"]:hover {
            color: rgba(143, 211, 255, 0.95) !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }

        section[data-testid="stSidebar"] button[kind="tertiary"] div,
        section[data-testid="stSidebar"] button[kind="tertiary"] p {
            display: block !important;
            text-align: left !important;
            margin: 0 !important;
            padding: 0 !important;
            font-size: 0.85rem !important;
            white-space: normal !important;
        }

        /* Kill the gap: target the CONTAINER wrapping each tertiary button */
        section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(button[kind="tertiary"]) {
            margin-top: -0.7rem !important;
            margin-bottom: 0 !important;
            padding: 0 !important;
        }

        /* Bullet via ::before with hanging indent */
        section[data-testid="stSidebar"] button[kind="tertiary"] p {
            padding-left: 1em !important;
            text-indent: -1em !important;
        }

        section[data-testid="stSidebar"] button[kind="tertiary"] p::before {
            content: "●" !important;
            font-size: 0.55em !important;
            vertical-align: middle !important;
            display: inline-block !important;
            width: 1em !important;
        }
        
        /* Fix session button height and prevent text wrapping - AGGRESSIVE */
        section[data-testid="stSidebar"] div[data-testid="column"]:first-child button {
            height: 2.5rem !important;
            min-height: 2.5rem !important;
            max-height: 2.5rem !important;
        }
        
        /* Prevent wrapping on ALL levels */
        section[data-testid="stSidebar"] div[data-testid="column"]:first-child button,
        section[data-testid="stSidebar"] div[data-testid="column"]:first-child button *,
        section[data-testid="stSidebar"] div[data-testid="column"]:first-child button p,
        section[data-testid="stSidebar"] div[data-testid="column"]:first-child button div {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        
        section[data-testid="stSidebar"] div[data-testid="column"]:first-child button p {
            margin: 0 !important;
            display: block !important;
        }
        
        /* Center icons in small button squares (export/delete) - force flex centering */
        section[data-testid="stSidebar"] div[data-testid="column"] button {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            min-width: 2.5rem !important;
        }
        
        section[data-testid="stSidebar"] div[data-testid="column"] button > div {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        section[data-testid="stSidebar"] .stDownloadButton button {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        section[data-testid="stSidebar"] .stDownloadButton button > div {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        /* Scrollable panel styling from original code */
        .sidebar-panel {
            font-size: 0.85rem;
            line-height: 1.25rem;
            color: rgba(233, 242, 249, 0.78);
            max-height: 280px;
            overflow-y: auto;
            padding-right: 0.35rem;
        }
        
        .sidebar-panel ul {
            margin: 0;
            padding-left: 1rem;
        }
        
        .sidebar-panel::-webkit-scrollbar {
            width: 6px;
        }
        
        .sidebar-panel::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.28);
            border-radius: 4px;
        }
        
        .sidebar-panel.docs {
            max-height: 320px;
        }
        
        /* ── Document file-list: clickable items ── */
        .doc-type-heading {
            margin: 0.6rem 0 0.2rem !important;
            font-size: 0.95rem !important;
            color: rgba(143, 211, 255, 0.9) !important;
            letter-spacing: 0.02em;
        }

        </style>
        """, unsafe_allow_html=True)
                
        if st.button("🔄 Start new chat", use_container_width=True, key="new_chat_btn", type="primary"):
            # Only create new session if current one has messages
            current_messages = app_state.get_current_session_messages()
            if current_messages:
                app_state.create_new_session()
            else:
                # Current session is empty, just reset it
                app_state.reset_session()
            _rerun_app()


        # Library management (only if not read-only)
        if not read_only_mode:
            with st.expander("📚 Library Management", expanded=False):
                # Load cache button
                if st.button("📥 Load cache", use_container_width=True, key="load_cache_btn"):
                    load_or_warn(app_state)
                
                st.divider()

                #st.write("**Rebuild Index (Parallel)**")
                clear_cache = st.checkbox(
                    "🗑️ Clear Gemini cache (re-extract all files)",
                    value=False,
                    key="rebuild_clear_cache",
                    help="Enable to re-extract all files via Gemini. Leave unchecked to use cached extractions (faster)."
                )                   
                
                # Rebuild section with checkbox
                if st.button("🔨 Rebuild index", use_container_width=True, key="rebuild_btn", type="primary"):
                    logged_in_tenant = st.session_state.get("tenant_id", "shared")
                    rebuild_index_parallel_execute(app_state, clear_cache, tenant_id=logged_in_tenant)        
                
                st.divider()
                
                # Sync button
                if st.button("🔄 Sync library", use_container_width=True, key="sync_btn"):
                    sync_library_with_ui(app_state)  # Use version with progress!
        

        # Documents on file - clickable with viewer dialog
        grouped = app_state.documents_grouped_by_type()
        with st.expander("📄 Documents on file", expanded=False):
            if grouped:
                order = ["FORM", "CHECKLIST", "PROCEDURE", "MANUAL", "POLICY", "REGULATION"]
                heading_map = {
                    "FORM": "Forms", "CHECKLIST": "Checklists",
                    "PROCEDURE": "Procedures", "MANUAL": "Manuals",
                    "POLICY": "Policies", "REGULATION": "Regulations",
                }

                with st.container(height=320, border=False):
                    for doc_type in sorted(
                        grouped,
                        key=lambda d: (order.index(d) if d in order else len(order), d),
                    ):
                        docs = grouped[doc_type]
                        if not docs:
                            continue
                        heading = heading_map.get(doc_type, doc_type.title())
                        st.markdown(
                            f"<h4 class='doc-type-heading'>{heading}</h4>",
                            unsafe_allow_html=True,
                        )
                        for display_title, source_filename in docs:
                            if st.button(
                                display_title,
                                key=f"docview_{source_filename}",
                                use_container_width=True,
                                type="tertiary",
                            ):
                                _view_document_dialog(source_filename, display_title)
            else:
                st.caption("No documents indexed yet.")

        # Sessions list
        with st.expander("💬 Sessions", expanded=True):
            with st.container(height=400, border=False):
                manager = app_state.ensure_session_manager()
                sessions = manager.list_sessions(limit=200)
                
                if sessions:
                    for session in sessions:
                        # Create unique key for each session button
                        button_key = f"session_{session.session_id}"
                        
                        # Show active indicator
                        is_current = session.session_id == app_state.current_session_id
                        prefix = "▶ " if is_current else "  "
                        
                        # Format title and message count
                        title_preview = session.title[:35] + "..." if len(session.title) > 35 else session.title
                        button_label = f"{prefix}{title_preview} ({session.message_count})"
                        
                        col1, col2, col3 = st.columns([3.5, 0.75, 0.75])
                        
                        with col1:
                            if st.button(button_label, key=button_key, use_container_width=True):
                                if not is_current:
                                    app_state.switch_session(session.session_id)
                                    _rerun_app()
                        
                        with col2:
                            # Export button - direct download without nested button
                            session_messages = manager.load_messages(session.session_id)
                            
                            # Convert to display format
                            messages_for_export = [
                                {
                                    "role": msg.role,
                                    "content": msg.content,
                                    "confidence_pct": msg.metadata.get("confidence_pct", 0),
                                    "confidence_level": msg.metadata.get("confidence_level", "N/A"),
                                    "confidence_note": msg.metadata.get("confidence_note", ""),
                                    "sources": msg.metadata.get("sources", []),
                                    "num_sources": msg.metadata.get("num_sources", 0),
                                    "retriever_type": msg.metadata.get("retriever_type", "unknown"),
                                }
                                for msg in session_messages
                            ]
                            
                            # Generate HTML
                            export_html = build_session_export_html(messages_for_export, session.title)
                            
                            # Direct download button
                            st.download_button(
                                label="📥",
                                data=export_html,
                                file_name=f"{session.title[:30].replace(' ', '_')}_session.html",
                                mime="text/html",
                                key=f"export_{session.session_id}",
                                help="Export session",
                                use_container_width=True,
                            )
                        
                        with col3:
                            if st.button("🗑️", key=f"delete_{session.session_id}", help="Delete session", use_container_width=True):
                                app_state.delete_session_with_uploads(session.session_id)
                                if is_current:
                                    # If deleting current session, create new one
                                    app_state.create_new_session()
                                _rerun_app()
                else:
                    st.caption("No sessions yet")

            with st.container(border=False):    
                # Clear all sessions button
                if sessions:
                    st.markdown("---")
                    if st.button("🗑️ Clear all sessions", use_container_width=True, type="primary"):
                        if st.session_state.get("confirm_clear_all"):
                            app_state.clear_all_sessions()
                            app_state.create_new_session()
                            st.session_state["confirm_clear_all"] = False
                            _rerun_app()
                        else:
                            st.session_state["confirm_clear_all"] = True
                            st.warning("⚠️ Click again to confirm deletion of ALL sessions")
        

        

    # Main chat interface
    st.markdown("---")
    
    # Ensure we have a current session
    if not app_state.current_session_id:
        app_state.create_new_session()

    # Render messages from current session
    messages = app_state.get_current_session_messages()

    last_user_query = None
    assistant_index = 0

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            if msg.get("is_upload_notice"):
                uploaded_files = msg.get("uploaded_files", [])
                with st.chat_message("user"):
                    st.markdown("**📎 Uploaded files**")
                    for record in uploaded_files:
                        name = html.escape(record.get("display_name", "Unknown file"))
                        size = record.get("size", 0)
                        chunks = record.get("num_chunks", 0)
                        st.markdown(
                            f"<div class='upload-card'><strong>{name}</strong><small>{_format_file_size(size)} · {chunks} chunks</small></div>",
                            unsafe_allow_html=True,
                        )
                continue

            last_user_query = msg.get("content", "")
            with st.chat_message("user"):
                st.markdown(last_user_query)

        elif role == "assistant":
            result = {
                "query": last_user_query or "Query",
                "answer": msg.get("content", ""),
                "confidence_pct": msg.get("confidence_pct", 0),
                "confidence_level": msg.get("confidence_level", "N/A"),
                "confidence_note": msg.get("confidence_note", ""),
                "sources": msg.get("sources", []),
                "num_sources": msg.get("num_sources", 0),
                "retriever_type": msg.get("retriever_type", "unknown"),
                "context_mode": msg.get("context_mode", False),
                "context_turn": msg.get("context_turn", 0),
                "context_reset_note": msg.get("context_reset_note"),
            }
            render_chat_message_with_feedback(app_state, result, assistant_index)
            assistant_index += 1

    session_uploads = app_state.get_session_upload_metadata()

    if "session_upload_nonce" not in st.session_state:
        st.session_state["session_upload_nonce"] = str(uuid.uuid4())

    uploader_key = f"session_upload_{app_state.current_session_id}_{st.session_state['session_upload_nonce']}"

    st.markdown("<div class='chat-input-wrapper'>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Attach documents",
        type=["pdf", "docx", "doc", "xlsx", "xls"],
        accept_multiple_files=True,
        key=uploader_key,
        label_visibility="collapsed",
        help=f"Attach up to {MAX_UPLOADS_PER_SESSION} files per session.",
    )

    # Initialize tracking PER SESSION
    session_hash_key = f"processed_upload_hashes_{app_state.current_session_id}"
    if session_hash_key not in st.session_state:
        st.session_state[session_hash_key] = set()

    if session_uploads:
        chips = []
        for record in session_uploads[:8]:
            chips.append(f"<span class='upload-chip'>{html.escape(record.get('display_name', 'File'))}</span>")
        remainder = len(session_uploads) - len(chips)
        if remainder > 0:
            chips.append(f"<span class='upload-chip more'>+{remainder} more</span>")
        st.markdown(f"<div class='chat-upload-chips'>{''.join(chips)}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='chat-upload-chips empty'>Attach PDFs, Word, or Excel files for this session.</div>",
            unsafe_allow_html=True,
        )

    if len(session_uploads) >= MAX_UPLOADS_PER_SESSION:
        st.markdown(
            f"<div class='upload-limit-note'>Upload limit of {MAX_UPLOADS_PER_SESSION} files reached.</div>",
            unsafe_allow_html=True,
        )

    # Chat input
    user_prompt = st.chat_input("Ask about the maritime library...")

    st.markdown("</div>", unsafe_allow_html=True)

    feedback_messages = st.session_state.pop("session_upload_feedback", None)
    if feedback_messages:
        for level, message in feedback_messages:
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            else:
                st.error(message)

    if uploaded_files:
        session_id = app_state.current_session_id
        if session_id:
            manager = app_state.ensure_session_upload_manager()
            new_records: List[Dict[str, Any]] = []
            feedback: List[Tuple[str, str]] = []
            
            total_files = len(uploaded_files)
            
            for idx, uploaded in enumerate(uploaded_files, 1):
                file_bytes = uploaded.read()
                if not file_bytes:
                    feedback.append(("warning", f"{uploaded.name} contained no data."))
                    continue
                # Check if already processed this file
                import hashlib
                file_hash = hashlib.md5(file_bytes).hexdigest()
                
                if file_hash in st.session_state[session_hash_key]:
                    LOGGER.info("Skipping %s - already processed", uploaded.name)
                    continue
            
                # NEW: Per-file spinner with progress counter
                with st.spinner(f"📎 Processing **{uploaded.name}** ({idx}/{total_files})..."):
                    result = manager.add_upload(
                        session_id,
                        uploaded.name,
                        file_bytes,
                        uploaded.type or "application/octet-stream",
                    )
            
                status = result.get("status")
                if status == "added":
                    # Mark as processed ONLY on success
                    st.session_state[session_hash_key].add(file_hash)
                    record = result.get("record", {})
                    new_records.append(record)
                    feedback.append((
                        "success",
                        f"Attached {record.get('display_name', uploaded.name)}",
                    ))
                elif status == "duplicate":
                    feedback.append(("warning", f"{uploaded.name} was already attached."))
                elif status == "limit":
                    feedback.append(("error", result.get("reason", "Upload limit reached.")))
                    break
                else:
                    feedback.append(("error", result.get("reason", f"Failed to process {uploaded.name}")))

            if new_records:
                app_state.refresh_session_upload_cache()
                summary_lines = [
                    f"📎 **{record.get('display_name', 'File')}**"
                    for record in new_records
                ]
                app_state.add_message_to_current_session(
                    "user",
                    "\n".join(summary_lines),
                    {"is_upload_notice": True, "uploaded_files": new_records},
                )

            if feedback:
                st.session_state["session_upload_feedback"] = feedback

            st.session_state["session_upload_nonce"] = str(uuid.uuid4())
            _rerun_app()

    if user_prompt:
        trimmed = user_prompt.strip()
        if not trimmed:
            st.warning("⚠️ Please enter a question.")
        else:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(trimmed)
            
            # Process query with streaming
            with st.chat_message("assistant"):
                try:
                    # Fancy status with updates
                    status = st.status("🔍 Processing your query...", expanded=True)
                    
                    with status:
                        result = orchestrated_query(
                            app_state,
                            trimmed,
                            use_conversation_context=True,
                            status_callback=lambda msg: st.write(msg),
                        )
                    
                    # Stream the response with typewriter effect
                    answer_stream = result.get("answer_stream")
                    if answer_stream:
                        # Buffer initial chunks to detect tables
                        initial_buffer = ""
                        chunks_collected = []
                        table_detected = False
                        
                        # Collect first ~500 chars or until table detected
                        for chunk in answer_stream:
                            chunks_collected.append(chunk)
                            initial_buffer += chunk
                            
                            # Check for table after collecting enough content
                            if len(initial_buffer) > 100:  # Need some content to detect
                                if detect_table_in_stream(initial_buffer):
                                    table_detected = True
                                    LOGGER.info("📊 Table detected in response, switching to buffered rendering")
                                    break
                            
                            # Stop buffering after reasonable amount if no table
                            if len(initial_buffer) > 500 and not table_detected:
                                break
                        
                        if table_detected:
                            LOGGER.info("Buffering complete response due to table content")
                            
                            MAX_RESPONSE_LEN = 50_000  # sane ceiling for any response
                            for chunk in answer_stream:
                                chunks_collected.append(chunk)
                                full_so_far = "".join(chunks_collected)
                                
                                # Bail out if response is suspiciously large
                                if len(full_so_far) > MAX_RESPONSE_LEN:
                                    LOGGER.warning("Response exceeded max length, likely degenerate — truncating")
                                    break
                                
                                # Bail out if we detect a degenerate line being built
                                # (any single line longer than 1000 chars is almost certainly broken)
                                last_newline = full_so_far.rfind('\n')
                                current_line = full_so_far[last_newline + 1:]
                                if len(current_line) > 1000 and current_line.count('-') > 500:
                                    LOGGER.warning("Degenerate table separator detected — truncating stream")
                                    break
                            
                            full_answer = "".join(chunks_collected)
                            full_answer = sanitize_markdown_tables(full_answer)
                            st.markdown(full_answer)
                            result["answer"] = full_answer
                        else:
                            # NORMAL PATH: Continue streaming with typewriter effect
                            # First show what we buffered
                            placeholder = st.empty()
                            placeholder.markdown(initial_buffer)
                            
                            # Then stream the rest
                            for chunk in answer_stream:
                                initial_buffer += chunk
                                placeholder.markdown(initial_buffer)
                            
                            full_answer = initial_buffer
                            result["answer"] = full_answer
                    else:
                        # No stream available, render complete
                        full_answer = result.get("answer", "")
                        st.markdown(full_answer)
                    
                    # NOW close the status after streaming starts
                    status.update(label="✅ Complete", state="complete", expanded=False)
                    
                    # CRITICAL: Remove generator from result before saving (can't serialize)
                    result.pop("answer_stream", None)
                    
                    # Display sources and confidence after streaming
                    conf_pct = result.get("confidence_pct", 0)
                    conf_level = result.get("confidence_level", "N/A")
                    num_sources = result.get("num_sources", 0)
                    
                    # Confidence badge
                    badge_emoji = "🟢" if "HIGH" in conf_level else "🟡" if "MEDIUM" in conf_level else "🔴"
                    conf_level_text = conf_level.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()
                    
                    caption_parts = [f"{badge_emoji} **Confidence:** {conf_pct}% ({conf_level_text})", f"**Sources:** {num_sources}"]
                    
                    if result.get("context_mode"):
                        context_turn = result.get("context_turn", 0)
                        caption_parts.append(f"💬 **Turn:** {context_turn}")
                    
                    if result.get("context_reset_note"):
                        st.info(result["context_reset_note"])
                    
                    st.caption(" • ".join(caption_parts))
                    
                    # Sources expander
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 View sources", expanded=False):
                            for idx, src in enumerate(sources[:5], 1):
                                source_file = src.get("source", "Unknown")
                                title = src.get("title") or source_file.rsplit('.', 1)[0].replace('_', ' ')
                                section = src.get("section", "Main document")
                                st.markdown(f"**{idx}. {title}**")
                                st.caption(f"└─ {section}")
                    
                    if result.get("confidence_note"):
                        st.info(result["confidence_note"])

                    # Add messages to session
                    app_state.add_message_to_current_session("user", trimmed)

                        # Extract metadata for assistant message
                    assistant_metadata = {
                        "confidence_pct": result.get("confidence_pct"),
                        "confidence_level": result.get("confidence_level"),
                        "confidence_note": result.get("confidence_note"),
                        "sources": result.get("sources"),
                        "num_sources": result.get("num_sources"),
                        "retriever_type": result.get("retriever_type"),
                        "context_mode": result.get("context_mode"),
                        "context_turn": result.get("context_turn"),
                        "context_reset_note": result.get("context_reset_note"),
                    }

                    app_state.add_message_to_current_session(
                        "assistant",
                        result.get("answer", ""),
                        assistant_metadata
                    )
                    
                    # Auto-generate session title after first real search (skip greetings/chitchat)
                    session = app_state.ensure_session_manager().get_session(app_state.current_session_id)
                    if session and session.title == "New Chat":
                        # Check if this was a real query (not greeting/chitchat)
                        retriever_type = result.get("retriever_type", "")
                        if retriever_type != "none":  # Real search happened - generate title now
                            app_state.ensure_session_manager().auto_generate_title(
                                app_state.current_session_id,
                                trimmed,
                                result.get("answer", "")[:200]
                            )

                    # DEPRECATED: Keep for backwards compatibility during transition
                    app_state.append_history(result)

                    LOGGER.info("Query processed: confidence=%d%%, sources=%d", 
                            result.get("confidence_pct", 0), result.get("num_sources", 0))
                    _rerun_app()
                    
                except Exception as exc:
                    LOGGER.exception("Query failed: %s", exc)
                    st.error(f"❌ **Search failed:** {exc}")
                    st.info("💡 **Try:**\n- Rephrasing your question\n- Using simpler terms\n- Checking if documents are indexed")


def render_viewer_app(app_state: AppState) -> None:
    """Restricted UI for read-only querying (no database management)."""
    render_app(app_state, read_only_mode=True)

def render_admin_panel(app_state: AppState) -> None:
    """
    Render full-width admin panel with all management features.
    
    NO CHAT INTERFACE - pure management UI.
    Toggle back to User mode for chat.
    """
    # === SUPERUSER GATE ===
    if not st.session_state.get("is_superuser", False):
        st.error("🚫 Admin panel requires superuser access.")
        st.markdown("""
        The admin panel is restricted to system administrators.
        
        If you need to:
        - **Upload documents**: Contact your administrator
        - **Delete documents**: Contact your administrator  
        - **Rebuild the index**: Contact your administrator
        
        Your chat history and queries are unaffected.
        """)
        
        if st.button("← Back to Chat", type="primary"):
            st.query_params["read_only"] = "true"
            st.rerun()
        
        # Stop rendering - don't show admin panel
        return
    
    # Mode toggle in sidebar
    with st.sidebar:

        if st.button("👤 Back to Chat", use_container_width=True, type="secondary"):
            st.query_params["read_only"] = "true"
            st.rerun()
        
        st.divider()

        st.markdown("### 🔧 Admin Panel")
        st.caption("Library management and configuration")
        
        st.divider()

        # === TENANT SELECTOR ===
        st.markdown("#### 👥 Manage Tenant")
        
        tenants = _get_tenant_list()
        tenant_names = _get_tenant_display_names()
        
        display_options = [tenant_names.get(t, t) for t in tenants]
        
        current_manage_tenant = st.session_state.get("manage_tenant", "shared")
        try:
            current_index = tenants.index(current_manage_tenant)
        except ValueError:
            current_index = 0
        
        selected_display = st.selectbox(
            "Select tenant to manage",
            options=display_options,
            index=current_index,
            key="manage_tenant_select",
            label_visibility="collapsed"
        )
        
        # Map display name back to tenant_id
        selected_tenant = tenants[display_options.index(selected_display)]
        
        if st.session_state.get("manage_tenant") != selected_tenant:
            st.session_state["manage_tenant"] = selected_tenant
            st.rerun()
        
        st.divider()
            
        # Show stats for selected tenant
        from .nodes import NodeRepository
        
        repo = NodeRepository(tenant_id=selected_tenant)
        tenant_nodes = repo.get_all_nodes()
        
        if selected_tenant != "shared":
            repo_shared = NodeRepository(tenant_id="shared")
            shared_nodes = repo_shared.get_all_nodes()
            visible_nodes = tenant_nodes + shared_nodes
        else:
            visible_nodes = tenant_nodes
            shared_nodes = []
        
        if visible_nodes:
            unique_sources = set(node.metadata.get("source") for node in visible_nodes)
            st.metric("Documents", len(unique_sources))
            
            if selected_tenant != "shared":
                st.caption(f"📁 {len(tenant_nodes)} tenant chunks")
                st.caption(f"🌐 {len(shared_nodes)} shared chunks")
            else:
                st.metric("Chunks", len(visible_nodes))
            
            try:
                manager = app_state.ensure_manager()
                total_db = manager.collection.count()
                st.caption(f"📊 {total_db} total in DB")
            except Exception as exc:
                LOGGER.warning("Failed to get DB count: %s", exc)
        else:
            st.info("No documents for this tenant")
            

    
    # Main panel
    st.title("🔧 MA.D.ASS - Admin Panel")
    
    # === LIBRARY MANAGEMENT ===
    with st.expander("📚 Library Management", expanded=True):
        _render_bulk_upload(app_state)
        
        st.divider()
        
        # Rebuild section
        st.markdown("### 🔨 Rebuild Index")
        st.markdown("""
        Process all documents and rebuild the search index from scratch.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            clear_cache = st.checkbox(
                "🔄 Clear Gemini cache and re-extract all files",
                value=False,
                help="Force re-extraction of all documents (slower but ensures fresh data)"
            )
            
            if clear_cache:
                st.warning("⚠️ All files will be re-extracted via Gemini API.")
        
        with col2:
            st.write("")
            st.write("")

            rebuild_clicked = st.button("🔨 Rebuild Index", type="primary", use_container_width=True)

        if rebuild_clicked:
            manage_tenant = st.session_state.get("manage_tenant", "shared")
            rebuild_index_parallel_execute(app_state, clear_cache, tenant_id=manage_tenant)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 🌲 Rebuild Document Tree")
            st.markdown("""
            Document trees enable **hierarchical retrieval** - fetching complete sections with their subsections.
            Use this button to rebuild trees without re-indexing.
            """)

        with col2:
            if st.button(
                "🌲 Rebuild Document Tree",
                help="Fast rebuild of hierarchical structure from existing cache without re-indexing"
            ):
                rebuild_trees_only(app_state)
        
        st.divider()
        
        #Database Health section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔄 Database Health")
            st.markdown("Rebuild memory from ChromaDB when memory is bloated.")
        
        with col2:
            st.write("")
            if st.button("🔄 Sync Memory to DB", use_container_width=True):
                _sync_memory_to_db(app_state)

        st.divider()
        
        # Sync section
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### 🔄 Sync Library")
            st.markdown("Check for new, modified, or deleted files and update incrementally.")

        with col2:
            st.write("")
            sync_clicked = st.button("🔄 Sync Library", use_container_width=True)

        # FIX: Call OUTSIDE columns so it renders full-width below
        if sync_clicked:
            sync_library_with_ui(app_state)

        st.divider()
        
        # Processing report
        if "last_processing_report" in st.session_state:
            report = st.session_state["last_processing_report"]
            render_processing_status_table(report)
        else:
            with st.expander("📊 Processing Status Report", expanded=False):
                st.info("No recent processing report. Build or sync to generate.")
                
                config = AppConfig.get()
                report_path = config.paths.cache_dir / "last_processing_report.json"
                
                if report_path.exists():
                    if st.button("📂 Load Last Report from Disk"):
                        from app.processing_status import load_processing_report
                        report = load_processing_report(report_path)
                        if report:
                            st.session_state["last_processing_report"] = report
                            st.rerun()
    
    # === DOCUMENTS ON FILE ===
    with st.expander("📄 Documents on File", expanded=True):
        _render_documents_with_delete(app_state)
    
    # === FEEDBACK ANALYTICS ===
    with st.expander("📊 Feedback Analytics", expanded=False):
        render_feedback_stats_panel(app_state)
    
    # === FORM SCHEMA CONFIGURATION ===
    with st.expander("🔧 Form Schema Configuration", expanded=True):
        _render_form_schema_editor()

    # === DOCUMENT INSPECTOR === (NEW - ADD THIS)
    with st.expander("📋 Document Inspector", expanded=False):
        render_document_inspector(app_state)


# ==============================================================================
# BULK UPLOAD
# ==============================================================================

def _render_bulk_upload(app_state: AppState) -> None:
    """Bulk file upload widget with enhanced file management."""
    
    st.markdown("### 📤 Bulk Document Upload")
    st.markdown("Upload multiple documents. Supported: **PDF, DOCX, DOC, XLSX, XLS, TXT**")
    
    # Use keyed uploader so we can clear it
    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = 0
    
    # Track files to exclude
    if "excluded_files" not in st.session_state:
        st.session_state.excluded_files = set()
    
    uploaded_files = st.file_uploader(
        "Select documents",
        type=["pdf", "docx", "doc", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"bulk_uploader_{st.session_state.upload_widget_key}"
    )
    
    if not uploaded_files:
        st.info("No files selected")
        # Clear exclusions when no files
        if st.session_state.excluded_files:
            st.session_state.excluded_files = set()
        return
    
    # Filter out excluded files
    uploaded_files = [f for f in uploaded_files if f.name not in st.session_state.excluded_files]
    
    if not uploaded_files:
        st.warning("All selected files were removed from upload list")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Reset removals", use_container_width=True):
                st.session_state.excluded_files = set()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear all", use_container_width=True):
                st.session_state.excluded_files = set()
                st.session_state.upload_widget_key += 1
                st.rerun()
        return
    
    # Show summary
    total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
    st.caption(f"📦 **{len(uploaded_files)} files** ({total_size_mb:.2f} MB total)")
    
    # Check for duplicates
    config = AppConfig.get()
    manage_tenant = st.session_state.get("manage_tenant", "shared")
    docs_path = config.docs_path_for(manage_tenant)
    existing_files = {f.name for f in docs_path.glob("*") if f.is_file()}
    duplicates = {f.name for f in uploaded_files if f.name in existing_files}
    
    # ALWAYS show our custom file list (not just for 3+)
    st.markdown("---")
    st.markdown("**📋 Files to upload:**")
    
    # Custom CSS for button styling only
    st.markdown("""
    <style>
    /* Make remove buttons minimal and properly sized */
    button[key^="remove_file_"] {
        background: transparent !important;
        border: none !important;
        padding: 0.3rem 0.5rem !important;
        box-shadow: none !important;
        min-width: auto !important;
        width: 2rem !important;
        height: 2rem !important;
        font-size: 1.1rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    button[key^="remove_file_"]:hover {
        background: rgba(255, 0, 0, 0.15) !important;
        border-radius: 0.3rem !important;
    }
    
    /* Force visible scrollbar on the container */
    div[data-testid="stVerticalBlock"] > div[style*="height: 400px"] {
        overflow-y: scroll !important;  /* Always show scrollbar */
        scrollbar-width: auto !important;  /* Firefox */
        scrollbar-color: rgba(255, 255, 255, 0.3) rgba(255, 255, 255, 0.05) !important;  /* Firefox */
    }
    
    /* Webkit scrollbar styling (Chrome, Edge, Safari) */
    div[data-testid="stVerticalBlock"] > div[style*="height: 400px"]::-webkit-scrollbar {
        width: 12px !important;
        display: block !important;
    }
    
    div[data-testid="stVerticalBlock"] > div[style*="height: 400px"]::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 6px !important;
        margin: 4px !important;
    }
    
    div[data-testid="stVerticalBlock"] > div[style*="height: 400px"]::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3) !important;
        border-radius: 6px !important;
        border: 2px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    div[data-testid="stVerticalBlock"] > div[style*="height: 400px"]::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use Streamlit's container with height parameter (Streamlit 1.31+)
    with st.container(height=400, border=True):
        # Display files with remove buttons
        for idx, uploaded_file in enumerate(uploaded_files):
            is_duplicate = uploaded_file.name in duplicates
            
            col1, col2, col3 = st.columns([6, 1.5, 0.5])
            
            with col1:
                # Highlight duplicates in red
                if is_duplicate:
                    st.markdown(f"🔴 **{uploaded_file.name}** _(exists on server)_")
                else:
                    st.markdown(f"📄 {uploaded_file.name}")
            
            with col2:
                st.text(_format_file_size(uploaded_file.size))
            
            with col3:
                # Plain text X that acts as button
                if st.button("❌", key=f"remove_file_{idx}_{uploaded_file.name}", 
                           help="Remove from upload", use_container_width=False):
                    st.session_state.excluded_files.add(uploaded_file.name)
                    st.rerun()
    
    st.markdown("---")
    
    # Handle duplicates
    if duplicates:
        st.warning(f"⚠️ **{len(duplicates)} duplicate filename(s) detected**")
        
        # Show overwrite option with explanation
        overwrite = st.checkbox(
            "🔄 Overwrite & force re-index",
            value=False,
            help="Delete old versions (including DB chunks) then upload new ones"
        )
        
        if not overwrite:
            st.error("❌ Upload blocked. Either:\n- Enable overwrite above, or\n- Click ❌ next to duplicates to remove them")
            return
        else:
            st.info("✓ Will delete old files completely, then upload and index new ones")
    
    # Check if ANY documents exist in the system (not just this tenant's folder)
    try:
        manager = app_state.ensure_manager()
        system_has_index = manager.collection.count() > 0
    except Exception:
        system_has_index = False
    
    if system_has_index:
        st.info("📚 Library exists → Will run **incremental sync**")
    else:
        st.info("🏗️ Empty library → Will run **full rebuild**")
    
    st.markdown("---")
    st.markdown("#### 📁 Document Ownership")

    tenants = _get_tenant_list()

    manage_tenant = st.session_state.get("manage_tenant", "shared")
    try:
        default_index = tenants.index(manage_tenant)
    except ValueError:
        default_index = 0
    
    selected_tenant = st.selectbox(
        "Upload to",
        options=tenants,
        index=default_index,
        help="'shared' = available to all tenants. Select a specific tenant for company-specific documents.",
        key="upload_tenant_select"
    )
    
    if selected_tenant == "shared":
        st.info("📋 These documents will be visible to **all tenants** (regulations, common procedures)")
    else:
        st.info(f"🔒 These documents will only be visible to **{selected_tenant}**")

    st.markdown("---")
    st.markdown("#### 🏷️ Document Type")

    doc_type_options = ["Automatic"] + ALLOWED_DOC_TYPES
    selected_doc_type = st.selectbox(
        "Classify as",
        options=doc_type_options,
        index=0,
        help="'Automatic' lets the AI classify each document. Select a specific type to override for ALL files in this batch.",
        key="upload_doc_type_select"
    )

    if selected_doc_type == "Automatic":
        st.caption("🤖 Each document will be classified individually by Gemini during extraction.")
    else:
        st.warning(f"⚠️ All documents in this batch will be tagged as **{selected_doc_type}**, regardless of AI classification.")

    # Map UI value to code value (None = automatic)
    doc_type_override = None if selected_doc_type == "Automatic" else selected_doc_type

    # Upload button
    if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
        _execute_bulk_upload(
            uploaded_files, app_state, system_has_index,
            overwrite_duplicates=list(duplicates) if duplicates and overwrite else [],
            tenant_id=selected_tenant,
            doc_type_override=doc_type_override
        )

def _execute_bulk_upload(
    uploaded_files: List,
    app_state: AppState,
    system_has_index: bool,
    overwrite_duplicates: List[str] = None,
    tenant_id: str = "shared",
    doc_type_override: str | None = None
) -> None:
    """Execute bulk upload and processing."""
    
    config = AppConfig.get()
    docs_path = config.docs_path_for(tenant_id)
    
    overwrite_duplicates = overwrite_duplicates or []
    
    # Step 1: Delete old versions if overwriting
    if overwrite_duplicates:
        st.write("### 🗑️ Deleting old versions...")
        delete_progress = st.progress(0.0)
        
        deleted = delete_duplicate_files(overwrite_duplicates, docs_path)
        delete_progress.progress(1.0)
        st.success(f"✅ Deleted {deleted} old files")
        
        # Run sync to remove from DB (if library exists)
        if system_has_index:
            st.write("Cleaning up database...")
            manager = app_state.ensure_manager()
            manager.nodes = app_state.nodes
            sync_result, _ = manager.sync_library(app_state.index, force_retry_errors=False)
            app_state.nodes = manager.nodes
            app_state.invalidate_node_map_cache()
            app_state.ensure_retrievers()
            st.success(f"✅ Removed {len(sync_result.deleted)} old entries from database")
        
        st.write("---")
    
    # Step 2: Copy new files
    st.write("### 📁 Copying new files...")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    # Use service helper
    copied_count, skipped_count, failed = copy_uploaded_files(uploaded_files, docs_path)
    
    # Update progress (files were copied, show completion)
    progress_bar.progress(1.0)
    status_text.text(f"Copied {copied_count} files")
    
    st.success(f"✅ Copied {copied_count} new files")
    
    if skipped_count > 0:
        st.warning(f"⚠️ Skipped {skipped_count} files due to errors")
        for f in failed:
            st.error(f"❌ Failed: `{f}`")
    
    # Step 3: Process
    st.write("---")
    
    if system_has_index:
        st.write("### 🔄 Running incremental sync...")
        sync_library_with_ui(app_state, tenant_id=tenant_id, doc_type_override=doc_type_override)
    else:
        st.write("### 🔨 Building index...")
        rebuild_index_parallel_execute(app_state, clear_gemini_cache=False, tenant_id=tenant_id, doc_type_override=doc_type_override)
    
    st.write("---")
    st.write("### 🌲 Updating document trees...")
    
    try:
        rebuild_trees_only(app_state)
        LOGGER.info("✅ Document trees automatically rebuilt after bulk upload")
    except Exception as exc:
        LOGGER.warning("⚠️ Tree rebuild failed after bulk upload: %s", exc)
        st.warning(f"⚠️ Tree rebuild failed: {exc}")
        st.info("💡 You can manually rebuild trees using the button in Admin panel")
    
    # Clear state
    st.session_state.excluded_files = set()
    st.session_state.upload_widget_key += 1
    LOGGER.info("Cleared file uploader after successful processing")
    
    # Force rerun to show fresh uploader
    st.rerun()


# ==============================================================================
# DOCUMENT DELETION
# ==============================================================================

def _render_documents_with_delete(app_state: AppState) -> None:
    """Admin document list with edit, delete, and batch operations."""

    # Get tenant to manage
    manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
    
    # Load nodes for managed tenant ONLY (shared has its own entry in dropdown)
    from .nodes import NodeRepository
    
    repo = NodeRepository(tenant_id=manage_tenant)
    managed_nodes = repo.get_all_nodes()
    
    st.markdown("### 📄 Documents on File")
    st.caption(f"Viewing: **{_get_tenant_display_names().get(manage_tenant, manage_tenant)}**")
    
    if not managed_nodes:
        st.info("No documents indexed for this tenant.")
        return
    
    # Check if we need to reset batch mode (from previous operation)
    if st.session_state.get("reset_batch_mode_flag", False):
        st.session_state["batch_mode_toggle"] = False
        st.session_state.batch_mode_enabled = False
        st.session_state.batch_selected_docs = set()
        st.session_state["reset_batch_mode_flag"] = False
    
    # Initialize session states
    if "editing_doc" not in st.session_state:
        st.session_state.editing_doc = None
    
    if "pending_edits" not in st.session_state:
        st.session_state.pending_edits = {}
    
    if "delete_confirmations" not in st.session_state:
        st.session_state.delete_confirmations = set()
    
    if "batch_mode_enabled" not in st.session_state:
        st.session_state.batch_mode_enabled = False
    if "batch_selected_docs" not in st.session_state:
        st.session_state.batch_selected_docs = set()
    if "batch_delete_confirm" not in st.session_state:
        st.session_state.batch_delete_confirm = False
    if "batch_edit_expanded" not in st.session_state:
        st.session_state.batch_edit_expanded = False
    
    if st.session_state.get("deletion_completed", False):
        st.session_state.delete_confirmations = set()
        st.session_state.deletion_completed = False
    
    # Group managed_nodes by doc_type (not app_state.nodes!)
    from collections import defaultdict
    docs_by_type: Dict[str, set] = defaultdict(set)
    for node in managed_nodes:
        metadata = node.metadata
        doc_type = str(metadata.get("doc_type", "UNCATEGORIZED")).upper()
        title = metadata.get("title") or metadata.get("source") or "Untitled"
        if doc_type == "FORM":
            form_number = metadata.get("form_number")
            if form_number:
                form_normalized = form_number.replace(" ", "").upper()
                title_start = title.split("-")[0].strip().replace(" ", "").upper()
                if not title_start.startswith(form_normalized):
                    title = f"{form_number} - {title}"
        docs_by_type[doc_type].add(title)
    docs_by_type = {k: sorted(list(v)) for k, v in docs_by_type.items()}
    
    total_docs = sum(len(titles) for titles in docs_by_type.values())
    st.caption(f"📚 **{total_docs} documents** in library")
    
    # NEW: Batch mode toggle
    batch_mode = st.checkbox(
        "Batch select mode",
        value=st.session_state.batch_mode_enabled,
        key="batch_mode_toggle",
        help="Enable to select multiple documents for batch operations"
    )
    
    # Update batch mode state
    if batch_mode != st.session_state.batch_mode_enabled:
        st.session_state.batch_mode_enabled = batch_mode
        # Clear selections when toggling mode
        st.session_state.batch_selected_docs = set()
        st.session_state.batch_delete_confirm = False
        st.session_state.batch_edit_expanded = False
        st.rerun()
    
    st.markdown("---")
    
    # Build mapping: display_title -> (source_filename, metadata) 
    display_to_meta: Dict[str, tuple] = {}
    for node in managed_nodes:
        metadata = node.metadata
        source = metadata.get("source", "")
        if not source:
            continue
        
        # Use same title logic as documents_grouped_by_type
        doc_type = str(metadata.get("doc_type", "")).upper()
        title = metadata.get("title") or source
        
        if doc_type == "FORM":
            form_number = metadata.get("form_number")
            if form_number:
                form_normalized = form_number.replace(" ", "").upper()
                title_start = title.split("-")[0].strip().replace(" ", "").upper()
                if not title_start.startswith(form_normalized):
                    title = f"{form_number} - {title}"
        
        display_to_meta[title] = (source, metadata)
    
    # Determine which expander should stay open (for editing)
    editing_doc = st.session_state.get("editing_doc")
    editing_doc_type = None
    if editing_doc:
        for node in managed_nodes:
            if node.metadata.get("source") == editing_doc:
                editing_doc_type = str(node.metadata.get("doc_type", "UNCATEGORIZED")).upper()
                break
    # FIX: Add "Select All" functionality for batch mode
    # Render by document type (MODIFIED: Add batch mode logic)
    for doc_type, titles in sorted(docs_by_type.items()):
        should_expand = (doc_type == editing_doc_type)
        with st.expander(f"**{doc_type}** ({len(titles)} docs)", expanded=should_expand):

            # ============================================================
            # Select All checkbox (only in batch mode)
            if batch_mode:
                # Get all filenames and their keys for this category
                category_files = []  # List of (display_title, source, unique_key)
                for display_title in titles:
                    source, metadata = display_to_meta.get(display_title, (display_title, {}))
                    unique_key = f"{doc_type}_{source}"
                    category_files.append((display_title, source, unique_key))
                
                # Check if all individual checkboxes are currently checked
                all_checked = all(
                    st.session_state.get(f"batch_sel_{uk}", False) 
                    for _, _, uk in category_files
                )
                
                # Track previous value of the select-all checkbox widget
                select_all_key = f"select_all_{doc_type}"
                prev_value_key = f"prev_{select_all_key}"
                
                # Select All checkbox
                col1, col2 = st.columns([0.3, 5.7], vertical_alignment="center")
                
                with col1:
                    select_all = st.checkbox(
                        "Select all",
                        value=all_checked,
                        key=select_all_key,
                        label_visibility="collapsed"
                    )
                
                with col2:
                    st.markdown("**Select all**")
                
                # ONLY update if the checkbox widget value actually changed
                # (meaning user clicked it, not just page rerendered)
                prev_value = st.session_state.get(prev_value_key, all_checked)
                
                if select_all != prev_value:
                    # User clicked the select-all checkbox!
                    for _, _, unique_key in category_files:
                        checkbox_key = f"batch_sel_{unique_key}"
                        st.session_state[checkbox_key] = select_all
                    
                    # Store current value as previous for next render
                    st.session_state[prev_value_key] = select_all
                
                st.markdown("---")
            # ============================================================

            for display_title in sorted(titles):
                source, metadata = display_to_meta.get(display_title, (display_title, {}))
                source_filename = Path(source).name
                unique_key = f"{doc_type}_{source}"
                
                # BATCH MODE vs NORMAL MODE
                if batch_mode:
                    # BATCH MODE: Show checkbox + title only
                    col1, col2 = st.columns([0.3, 5.7], vertical_alignment="center")
                    
                    with col1:
                        is_selected = source_filename in st.session_state.batch_selected_docs
                        if st.checkbox("Select", value=is_selected, key=f"batch_sel_{unique_key}", label_visibility="collapsed"):
                            st.session_state.batch_selected_docs.add(source_filename)
                        else:
                            st.session_state.batch_selected_docs.discard(source_filename)
                    
                    with col2:
                        st.markdown(f"📄 {display_title}")
                
                else:
                    # NORMAL MODE: Existing edit/delete logic (UNCHANGED)
                    is_editing = st.session_state.editing_doc == source
                    is_confirming_delete = unique_key in st.session_state.delete_confirmations
                    
                    if is_editing:
                        # EDITING STATE - Show cancel button only
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**{display_title}**")
                        with col2:
                            if st.button("❌", key=f"cancel_{source}", help="Cancel editing"):
                                st.session_state.editing_doc = None
                                if source in st.session_state.pending_edits:
                                    del st.session_state.pending_edits[source]
                                st.rerun()
                        
                        # Show edit form
                        st.markdown("─" * 50)
                        _render_metadata_edit_form(source, metadata, app_state)
                        st.markdown("─" * 50)
                    
                    elif is_confirming_delete:
                        # DELETE CONFIRMATION STATE
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"📄 {display_title}")
                        with col2:
                            if st.button("✅", key=f"confirm_{unique_key}", use_container_width=True):
                                success = _delete_document_by_source(source, display_title, app_state)
                                if success:
                                    st.session_state.deletion_completed = True
                                st.rerun()
                    
                    else:
                        # NORMAL STATE - Show edit and delete buttons
                        col1, col2, col3 = st.columns([4, 1, 1])
                        with col1:
                            st.markdown(f"📄 {display_title}")
                        with col2:
                            if st.button("✏️", key=f"edit_{unique_key}", use_container_width=True, help="Edit metadata"):
                                st.session_state.editing_doc = source
                                
                                # Load current values
                                manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
                                form_categories = load_form_categories(manage_tenant)
                                category_value = metadata.get("form_category_name", "")
                                
                                # Map code to full name if it's a shortcode
                                if category_value and category_value in form_categories:
                                    category_value = form_categories[category_value]
                                
                                st.session_state.pending_edits[source] = {
                                    "doc_type": str(metadata.get("doc_type", "DOCUMENT")).upper(),
                                    "title": metadata.get("title", ""),
                                    "form_number": metadata.get("form_number", ""),
                                    "form_category_name": category_value,
                                    "tenant_id": metadata.get("tenant_id", "shared"),
                                }
                                st.session_state[f"expander_{doc_type}"] = True
                                st.rerun()
                        with col3:
                            if st.button("🗑️", key=f"delete_{unique_key}", use_container_width=True):
                                st.session_state.delete_confirmations.add(unique_key)
                                st.rerun()
    
    # NEW: Batch operations buttons (only in batch mode)
    if batch_mode and st.session_state.batch_selected_docs:
        st.markdown("---")
        
        num_selected = len(st.session_state.batch_selected_docs)
        st.caption(f"**{num_selected} document(s) selected**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Batch delete
            if st.session_state.batch_delete_confirm:
                if st.button(f"✅ Confirm Delete {num_selected}", type="primary", use_container_width=True):
                    _batch_delete_documents(list(st.session_state.batch_selected_docs), app_state)
                    st.session_state.batch_selected_docs = set()
                    st.session_state.batch_delete_confirm = False
                    st.rerun()
            else:
                if st.button(f"🗑️ Delete Selected ({num_selected})", use_container_width=True):
                    st.session_state.batch_delete_confirm = True
                    st.rerun()
        
        with col2:
            if st.button(f"✏️ Edit Type/Ownership ({num_selected})", use_container_width=True):
                st.session_state.batch_edit_expanded = not st.session_state.batch_edit_expanded
                st.rerun()
        
        # Cancel delete confirmation
        if st.session_state.batch_delete_confirm:
            if st.button("❌ Cancel Delete", use_container_width=True):
                st.session_state.batch_delete_confirm = False
                st.rerun()
        
        # Batch edit expander
        if st.session_state.batch_edit_expanded:
            with st.expander("✏️ Edit Document Type", expanded=True):
                _render_batch_edit_form(list(st.session_state.batch_selected_docs), app_state)
    
    # Nuclear option (EXISTING CODE - UNCHANGED)
    st.markdown("---")
    st.markdown("### ⚠️ Danger Zone")
    
    is_confirming_nuclear = "confirm_nuclear_delete" in st.session_state
    
    if is_confirming_nuclear:
        st.warning("⚠️ **Delete ALL documents?** This cannot be undone!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Delete Everything", type="primary", use_container_width=True):
                _delete_entire_library(app_state)
                del st.session_state["confirm_nuclear_delete"]
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                del st.session_state["confirm_nuclear_delete"]
                st.rerun()
    else:
        if st.button("⚠️ Delete Entire Library", use_container_width=True):
            st.session_state["confirm_nuclear_delete"] = True
            st.rerun()


def _render_metadata_edit_form(source: str, metadata: Dict, app_state: AppState) -> None:
    """Render inline metadata editing form."""
    pending = st.session_state.pending_edits.get(source, {})
    
    # Filename (read-only)
    st.text_input("Filename", value=source, disabled=True, key=f"filename_{source}")
    
    # Title
    new_title = st.text_input(
        "Title",
        value=pending.get("title", ""),
        key=f"title_{source}"
    )
    pending["title"] = new_title
    
    # Document Type
    current_doc_type = str(pending.get("doc_type", "DOCUMENT")).upper()
    try:
        type_index = ALLOWED_DOC_TYPES.index(current_doc_type)
    except ValueError:
        LOGGER.warning("Unknown doc_type '%s' for %s, defaulting to first option", current_doc_type, source)
        type_index = 0
    
    new_doc_type = st.selectbox(
        "Document Type",
        options=ALLOWED_DOC_TYPES,
        index=type_index,
        key=f"doctype_{source}"
    )
    pending["doc_type"] = new_doc_type
    
    # Ownership (tenant_id)
    tenants = _get_tenant_list()
    current_tenant = pending.get("tenant_id", "shared")
    try:
        tenant_index = tenants.index(current_tenant)
    except ValueError:
        tenant_index = 0
    
    new_tenant = st.selectbox(
        "Ownership",
        options=tenants,
        index=tenant_index,
        key=f"tenant_{source}",
        help="'shared' = visible to all tenants"
    )
    pending["tenant_id"] = new_tenant
    
    # Form-specific fields (show for FORM and CHECKLIST)
    if new_doc_type in ["FORM", "CHECKLIST"]:
        new_form_number = st.text_input(
            "Form Number",
            value=pending.get("form_number", ""),
            key=f"formnum_{source}"
        )
        pending["form_number"] = new_form_number
        
        manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
        form_categories = load_form_categories(manage_tenant)
        category_options = [""] + sorted(form_categories.values())
        
        current_category = pending.get("form_category_name", "")
        try:
            cat_index = category_options.index(current_category)
        except ValueError:
            cat_index = 0
        
        new_category = st.selectbox(
            "Form Category",
            options=category_options,
            index=cat_index,
            key=f"formcat_{source}"
        )
        pending["form_category_name"] = new_category
    
    st.session_state.pending_edits[source] = pending
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Changes", key=f"save_{source}", type="primary", use_container_width=True):
            corrections = {}
            
            if pending["title"] != metadata.get("title", ""):
                corrections["title"] = pending["title"]
            
            old_doc_type = str(metadata.get("doc_type", "")).upper()
            if pending["doc_type"] != old_doc_type:
                corrections["doc_type"] = pending["doc_type"]
            
            # Check tenant change
            old_tenant = metadata.get("tenant_id", "shared")
            if pending["tenant_id"] != old_tenant:
                corrections["tenant_id"] = pending["tenant_id"]
            
            if new_doc_type in ["FORM", "CHECKLIST"]:
                if pending["form_number"] != metadata.get("form_number", ""):
                    corrections["form_number"] = pending["form_number"]
                
                selected_category = pending["form_category_name"]
                old_category = metadata.get("form_category_name", "")
                
                manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
                form_categories = load_form_categories(manage_tenant)
                reverse_map = {v: k for k, v in form_categories.items()}
                
                category_code = reverse_map.get(selected_category, selected_category)
                old_category_code = old_category if old_category in form_categories else reverse_map.get(old_category, old_category)
                
                if category_code != old_category_code:
                    corrections["form_category_name"] = category_code
            
            if corrections:
                config = AppConfig.get()
                
                success = update_metadata_everywhere(
                    source,
                    corrections,
                    app_state.nodes,
                    config.paths.chroma_path
                )
                
                if success:
                    # Update SQLite nodes
                    from .nodes import NodeRepository
                    from .database import db_connection
                    
                    with db_connection() as conn:
                        if "tenant_id" in corrections:
                            from .metadata_updates import save_tenant_id, transfer_document_ownership
                            
                            # Transfer physical file and JSONL record to new tenant
                            transfer_document_ownership(source, corrections["tenant_id"])
                            
                            # Update tenant_id in JSONL record (now in new tenant's cache)
                            save_tenant_id(source, corrections["tenant_id"])
                            
                            conn.execute(
                                "UPDATE nodes SET tenant_id = ? WHERE doc_id = ?",
                                (corrections["tenant_id"], source)
                            )
                        # Update metadata JSON for other fields
                        manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
                        repo = NodeRepository(tenant_id=manage_tenant)
                        managed_nodes = repo.get_all_nodes()
                        import json
                        for node in managed_nodes:
                            if node.metadata.get("source") == source:
                                # Apply corrections to the fresh node metadata
                                for field, value in corrections.items():
                                    node.metadata[field] = value
                                conn.execute(
                                    "UPDATE nodes SET metadata = ? WHERE doc_id = ? AND node_id = ?",
                                    (json.dumps(node.metadata), source, node.node_id)
                                )
                    
                    st.success("✅ Metadata updated successfully!")
                    st.session_state.editing_doc = None
                    if source in st.session_state.pending_edits:
                        del st.session_state.pending_edits[source]
                    
                    # Force reload nodes on next access
                    app_state.nodes = []
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to update metadata")
            else:
                st.info("No changes detected")
    
    with col2:
        if st.button("Cancel", key=f"cancel2_{source}", use_container_width=True):
            st.session_state.editing_doc = None
            if source in st.session_state.pending_edits:
                del st.session_state.pending_edits[source]
            st.rerun()


def _render_batch_edit_form(filenames: List[str], app_state: AppState) -> None:
    """Render batch edit form for changing doc_type and ownership of multiple documents."""
    
    st.markdown(f"**Editing {len(filenames)} document(s)**")
    
    # Doc type dropdown
    new_doc_type = st.selectbox(
        "New Document Type",
        options=["(no change)"] + ALLOWED_DOC_TYPES,
        key="batch_edit_doc_type"
    )
    
    # Ownership dropdown
    tenants = _get_tenant_list()
    new_tenant = st.selectbox(
        "New Ownership",
        options=["(no change)"] + tenants,
        key="batch_edit_tenant",
        help="'shared' = visible to all tenants"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Changes", type="primary", use_container_width=True, key="batch_save"):
            if new_doc_type == "(no change)" and new_tenant == "(no change)":
                st.warning("No changes selected")
                return
            
            success_count = 0
            
            with st.spinner(f"Updating {len(filenames)} documents..."):
                # Load nodes for managed tenant
                manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
                from .nodes import NodeRepository
                repo = NodeRepository(tenant_id=manage_tenant)
                managed_nodes = repo.get_all_nodes()
                
                for filename in filenames:
                    try:
                        source = None
                        metadata = None
                        for node in managed_nodes:
                            node_source = node.metadata.get("source", "")
                            if Path(node_source).name == filename:
                                source = node_source
                                metadata = node.metadata
                                break
                        
                        if source and metadata:
                            corrections = {}
                            
                            # Only add doc_type if actually changing
                            if new_doc_type != "(no change)":
                                corrections["doc_type"] = new_doc_type
                            
                            # Only add tenant_id if actually changing
                            if new_tenant != "(no change)":
                                corrections["tenant_id"] = new_tenant
                            
                            if not corrections:
                                continue
                            
                            config = AppConfig.get()
                            
                            # Update in-memory nodes, Gemini cache, and ChromaDB
                            success = update_metadata_everywhere(
                                source,
                                corrections,
                                app_state.nodes,
                                config.paths.chroma_path
                            )
                            
                            if success:
                                # Update SQLite
                                from .database import db_connection
                                import json
                                
                                with db_connection() as conn:
                                    if "tenant_id" in corrections:
                                        conn.execute(
                                            "UPDATE nodes SET tenant_id = ? WHERE doc_id = ?",
                                            (corrections["tenant_id"], source)
                                        )
                                        from .metadata_updates import save_tenant_id, transfer_document_ownership
                                        transfer_document_ownership(source, corrections["tenant_id"])
                                        save_tenant_id(source, corrections["tenant_id"])
                                    
                                    # Update metadata JSON in SQLite too
                                    managed_repo = NodeRepository(tenant_id=manage_tenant)
                                    batch_managed_nodes = managed_repo.get_all_nodes()
                                    for node in batch_managed_nodes:
                                        if node.metadata.get("source") == source:
                                            for field, value in corrections.items():
                                                node.metadata[field] = value
                                            conn.execute(
                                                "UPDATE nodes SET metadata = ? WHERE node_id = ?",
                                                (json.dumps(node.metadata), node.node_id)
                                            )
                                
                                success_count += 1
                        else:
                            LOGGER.warning(f"Could not find source for {filename}")
                    
                    except Exception as exc:
                        LOGGER.exception(f"Failed to update {filename}")
                        st.error(f"❌ Failed: {filename}")
            
            if success_count > 0:
                changes = []
                if new_doc_type != "(no change)":
                    changes.append(f"type={new_doc_type}")
                if new_tenant != "(no change)":
                    changes.append(f"owner={new_tenant}")
                
                st.success(f"✅ Updated {success_count}/{len(filenames)} documents ({', '.join(changes)})")
                
                # Force reload nodes
                app_state.nodes = []
            
            st.session_state.batch_selected_docs = set()
            st.session_state.batch_edit_expanded = False
            st.session_state["reset_batch_mode_flag"] = True
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True, key="batch_cancel"):
            st.session_state.batch_edit_expanded = False
            st.rerun()


def _batch_delete_documents(filenames: List[str], app_state: AppState) -> None:
    """Delete multiple documents and sync once at the end."""
    
    from .config import AppConfig
    
    config = AppConfig.get()
    manage_tenant = st.session_state.get("manage_tenant", 
                    st.session_state.get("tenant_id", "shared"))
    docs_path = config.docs_path_for(manage_tenant)
    
    # Step 1: Delete all files from disk
    deleted_count = 0
    with st.spinner(f"Deleting {len(filenames)} files..."):
        for filename in filenames:
            file_path = docs_path / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_count += 1
                    LOGGER.info(f"Deleted file: {filename}")
                except Exception as exc:
                    LOGGER.exception(f"Failed to delete {filename}")
                    st.error(f"❌ Failed to delete: {filename}")
            else:
                LOGGER.warning(f"File not found: {filename}")
    
    st.success(f"✅ Deleted {deleted_count}/{len(filenames)} files from disk")
    
    # Step 2: Sync library once to clean up database
    if deleted_count > 0:
        with st.spinner("Syncing library to update database..."):
            manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
            manager = app_state.ensure_manager()
            manager.tenant_id = manage_tenant
            manager.nodes = app_state.nodes
            
            sync_result, _ = manager.sync_library(
                app_state.index,
                force_retry_errors=False
            )
            
            # Update app state
            app_state.nodes = manager.nodes
            app_state.invalidate_node_map_cache()
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
        
        st.success(f"✅ Removed {len(sync_result.deleted)} entries from database")
        st.session_state.batch_mode_enabled = False
        LOGGER.info(f"Batch deleted {deleted_count} files, removed {len(sync_result.deleted)} from DB")
        st.session_state["reset_batch_mode_flag"] = True   # Uncheck the checkbox
        #st.session_state.batch_selected_docs = set()    # Clear selections

def _delete_document_by_source(source_filename: str, display_title: str, app_state: AppState) -> bool:
    """UI wrapper for delete_document_by_source service."""
    
    if not source_filename:
        st.error(f"❌ Invalid source filename for: {display_title}")
        LOGGER.error("Empty source filename for display_title: %s", display_title)
        return False
    
    manage_tenant = st.session_state.get("manage_tenant", st.session_state.get("tenant_id", "shared"))
    
    with st.spinner(f"Deleting {display_title}..."):
        result = delete_document_by_source(source_filename, app_state, tenant_id=manage_tenant)
    
    if result.success:
        st.success(f"✅ Deleted: {display_title}")
        if result.error:  # Partial success (sync failed)
            st.warning(f"⚠️ {result.error}")
        return True
    else:
        st.error(f"❌ Failed to delete: {result.error}")
        return False


def _delete_entire_library(app_state: AppState) -> None:
    """UI wrapper for delete_entire_library service."""
    
    with st.spinner("🔥 Deleting entire library..."):
        result = delete_entire_library(app_state)
    
    if result.success:
        st.success(f"✅ Deleted {result.files_deleted} files")
        st.info("💡 Library is now empty. Upload new documents to rebuild.")
    else:
        st.error(f"❌ Failed: {result.error}")


# ==============================================================================
# FORM SCHEMA EDITOR
# ==============================================================================

def _render_form_schema_editor() -> None:
    """Form schema configuration editor with polished UX."""
    
    # Get tenant to manage
    manage_tenant = st.session_state.get("manage_tenant", "shared")
    tenant_display = _get_tenant_display_names().get(manage_tenant, manage_tenant)
    
    st.markdown("### 🔧 Form Schema Configuration")
    st.caption(f"Editing codes for: **{tenant_display}**")
    
    if manage_tenant == "shared":
        st.info("📋 Shared codes are used as defaults for all tenants.")
    else:
        st.info(f"📋 These codes are specific to **{tenant_display}**. Falls back to shared if empty.")
    
    # Always load fresh from JSON for managed tenant
    current_categories = load_form_categories(manage_tenant)
    
    # Initialize state
    if "form_schema_editor" not in st.session_state:
        st.session_state.form_schema_editor = {
            "confirm_delete": set(),
            "last_saved": current_categories.copy(),
            "input_counter": 0,
        }
    
    editor_state = st.session_state.form_schema_editor
    categories = current_categories
    
    st.caption(f"**{len(categories)} codes**")
    st.markdown("---")
    
    # Render existing categories
    for code, description in sorted(categories.items()):
        col1, col2, col3 = st.columns([1.5, 5, 1])
        
        with col1:
            st.text_input(
                "Code", 
                value=code, 
                key=f"display_code_{code}",
                disabled=True, 
                label_visibility="collapsed"
            )
        
        with col2:
            st.text_input(
                "Desc", 
                value=description, 
                key=f"display_desc_{code}",
                disabled=True, 
                label_visibility="collapsed"
            )
        
        with col3:
            is_confirming = code in editor_state["confirm_delete"]
            
            if is_confirming:
                if st.button("✅", key=f"confirm_del_{code}", use_container_width=True, type="primary"):
                    if code in categories:
                        del categories[code]
                        LOGGER.info(f"Deleted form code: {code} (tenant={manage_tenant})")
                    
                    if save_form_categories(categories, manage_tenant):
                        editor_state["last_saved"] = categories.copy()
                        LOGGER.info("Saved after deletion")
                    
                    editor_state["confirm_delete"].discard(code)
                    
                    for key in [f"display_code_{code}", f"display_desc_{code}", 
                               f"delete_btn_{code}", f"confirm_del_{code}"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.rerun()
            else:
                if st.button("🗑️", key=f"delete_btn_{code}", use_container_width=True):
                    editor_state["confirm_delete"].add(code)
                    st.rerun()
    
    st.markdown("---")
    
    # Add new code section
    st.markdown("#### ➕ Add New Code")
    
    input_key_suffix = editor_state["input_counter"]
    
    col1, col2, col3 = st.columns([1.5, 5, 1])
    
    with col1:
        new_code = st.text_input(
            "Code", 
            placeholder="e.g., X", 
            max_chars=10, 
            label_visibility="collapsed",
            key=f"new_code_input_{input_key_suffix}"
        )
    
    with col2:
        new_description = st.text_input(
            "Description", 
            placeholder="e.g., Example", 
            label_visibility="collapsed",
            key=f"new_desc_input_{input_key_suffix}"
        )
    
    with col3:
        add_button = st.button("➕", use_container_width=True, type="secondary", key="add_new_btn")
    
    if add_button:
        new_code_clean = new_code.strip().upper()
        new_desc_clean = new_description.strip()
        
        if not new_code_clean or not new_desc_clean:
            st.warning("⚠️ Both required")
        elif new_code_clean in categories:
            st.error(f"⚠️ `{new_code_clean}` exists!")
        else:
            categories[new_code_clean] = new_desc_clean
            LOGGER.info(f"Added form code: {new_code_clean} (tenant={manage_tenant})")
            
            if save_form_categories(categories, manage_tenant):
                editor_state["last_saved"] = categories.copy()
                LOGGER.info("Saved after addition")
            
            editor_state["input_counter"] += 1
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💾 **Changes save automatically**")
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary", key="clear_all_btn"):
            st.session_state["confirm_clear_all"] = True
            st.rerun()
    
    if st.session_state.get("confirm_clear_all", False):
        st.warning("⚠️ **Delete ALL form codes?** This will clear the entire schema!")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes, Clear All", type="primary", use_container_width=True, key="confirm_clear_all_yes"):
                empty_schema = {}
                if save_form_categories(empty_schema, manage_tenant):
                    editor_state["last_saved"] = {}
                    st.success("✅ All codes cleared!")
                    LOGGER.info(f"Cleared all form codes (tenant={manage_tenant})")
                
                del st.session_state["confirm_clear_all"]
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True, key="confirm_clear_all_no"):
                del st.session_state["confirm_clear_all"]
                st.rerun()

__all__ = [
    "compose_result_markdown",
    "save_result_as_html",
    "render_app",
    "render_viewer_app",
    "render_form_schema_editor",
    "render_bulk_upload",
    "render_documents_with_delete",
]
