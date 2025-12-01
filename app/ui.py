"""Streamlit UI helpers and rendering functions."""

from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
import html
from markdown import markdown as md
import streamlit as st

from .config import AppConfig
from .indexing import build_index_from_library, build_index_from_library_parallel, load_cached_nodes_and_index
from .query import query_with_confidence, cohere_client
from .state import AppState
from .logger import LOGGER
from .constants import MAX_CONTEXT_TURNS  # Import for context display
from .session_uploads import MAX_UPLOADS_PER_SESSION
from .processing_status import ProcessingReport, DocumentProcessingStatus, StageStatus


_DEF_EXTS = ["extra", "tables", "fenced_code", "sane_lists"]



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
                    "Chunks": status.chunks_created if status.chunks_created else "-",
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
            rebuild_index_parallel_execute(app_state, clear_cache)
    
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

def rebuild_index_parallel_execute(app_state: AppState, clear_gemini_cache: bool = False) -> None:
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
            clear_gemini_cache=clear_gemini_cache
        )

        # Store report in session state
        st.session_state["last_processing_report"] = report

        # Update app state
        app_state.nodes = nodes
        app_state.index = index
        app_state.invalidate_node_map_cache()  # Clear stale cache
        app_state.vector_retriever = None
        app_state.bm25_retriever = None
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

def sync_library(app_state: AppState) -> None:
    manager = app_state.ensure_manager()
    manager.nodes = app_state.nodes
    with st.spinner("Syncing library..."):
        try:
            changes = manager.sync_library(app_state.index)
            st.success(
                f"✅ Sync complete. Added {len(changes.added)}, modified {len(changes.modified)}, deleted {len(changes.deleted)}."
            )
            app_state.nodes = manager.nodes
            app_state.invalidate_node_map_cache()  # Clear stale cache
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
            LOGGER.info("Library synced: +%d, ~%d, -%d", len(changes.added), len(changes.modified), len(changes.deleted))
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
        <h5 style="margin-top: -0.3em; color: #666;">Intelligent document search powered by the dreams of electric sheep</h5>
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
    st.session_state.setdefault("rerank_enabled", cohere_client is not None)
    st.session_state.setdefault("fortify_option", False)
    st.session_state.setdefault("auto_refine_option", False)
    st.session_state.setdefault("use_context", True)  # Default ON for context-aware chat
    st.session_state.setdefault("use_hierarchical", True)  # Default ON for hierarchical retrieval

    # Sidebar configuration
    with st.sidebar:
        # Admin/User mode toggle button at the top
        if read_only_mode:
            # In viewer mode - show Admin button
            if st.button("🔓 Admin", use_container_width=True, key="mode_toggle", type="primary"):
                st.query_params["read_only"] = "false"
                _rerun_app()
        else:
            # In admin mode - show User button
            if st.button("👤 User", use_container_width=True, key="mode_toggle", type="secondary"):
                st.query_params["read_only"] = "true"
                _rerun_app()

        st.markdown("---")

        # Custom CSS for button styling (not feedback buttons)
        st.markdown("""
        <style>
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
        
        .sidebar-panel.docs .doc-type {
            margin-bottom: 0.75rem;
        }
        
        .sidebar-panel.docs .doc-type h4 {
            margin: 0 0 0.3rem;
            font-size: 0.95rem;
            color: rgba(143, 211, 255, 0.9);
            letter-spacing: 0.02em;
        }
        
        .sidebar-panel.docs .doc-type li {
            list-style: disc;
            margin-bottom: 0.2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.header("⚙️ Settings")
        
        if st.button("🔄 Start new chat", use_container_width=True, key="new_chat_btn", type="primary"):
            # Only create new session if current one has messages
            current_messages = app_state.get_current_session_messages()
            if current_messages:
                app_state.create_new_session()
            else:
                # Current session is empty, just reset it
                app_state.reset_session()
            _rerun_app()
        
        with st.expander("🔍 Retrieval Options", expanded=True):
            retrieval_type = st.selectbox(
                "Method",
                ["hybrid", "vector", "bm25"],
                key="retrieval_method",
                help="Hybrid combines vector and keyword search"
            )
            
            rerank_available = cohere_client is not None
            rerank_option = st.checkbox(
                "Enable reranking",
                key="rerank_enabled",
                disabled=not rerank_available,
                help="Re-rank results with Cohere (requires API key)"
            )
            
            fortify_option = st.checkbox(
                "Fortify query",
                key="fortify_option",
                help="Enhance query with Gemini before searching"
            )
            
            auto_refine_option = st.checkbox(
                "Auto-refine queries",
                key="auto_refine_option",
                help="Automatically rephrase low-confidence queries"
            )
            
            # NEW: Context-aware conversation toggle
            use_context = st.checkbox(
                "💬 Context-aware chat",
                key="use_context",
                help=f"Remember previous exchanges (resets after {MAX_CONTEXT_TURNS} turns)"
            )
            
            # Show context status if enabled
            if use_context and app_state.context_turn_count > 0:
                st.caption(f"📍 Turn {app_state.context_turn_count}/{MAX_CONTEXT_TURNS}")
                if app_state.context_turn_count >= MAX_CONTEXT_TURNS - 1:
                    st.caption("⚠️ Next query will start fresh")

            # Hierarchical retrieval toggle (A/B testing)
            use_hierarchical = st.checkbox(
                "🔍 Hierarchical retrieval",
                value=True,
                key="use_hierarchical",
                help="Use section-level retrieval for procedures"
            )

        # Library management (only if not read-only)
        if not read_only_mode:
            with st.expander("📚 Library Management", expanded=False):
                # Load cache button
                if st.button("📥 Load cache", use_container_width=True, key="load_cache_btn"):
                    load_or_warn(app_state)
                
                st.divider()
                
                # Rebuild section with checkbox
                st.write("**Rebuild Index (Parallel)**")
                clear_cache = st.checkbox(
                    "🗑️ Clear Gemini cache (re-extract all files)",
                    value=False,
                    key="rebuild_clear_cache",
                    help="Enable to re-extract all files via Gemini. Leave unchecked to use cached extractions (faster)."
                )
                
                if st.button("🔨 Rebuild index", use_container_width=True, key="rebuild_btn", type="primary"):
                    rebuild_index_parallel_execute(app_state, clear_cache)
                
                st.divider()
                
                # Sync button
                if st.button("🔄 Sync library", use_container_width=True, key="sync_btn"):
                    sync_library(app_state)
        

        # Sessions list
        with st.expander("💬 Sessions", expanded=True):
            manager = app_state.ensure_session_manager()
            sessions = manager.list_sessions(limit=20)
            
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
        
        # Documents on file - using original HTML approach with scrollable panel
        grouped = app_state.documents_grouped_by_type()
        with st.expander("📄 Documents on file", expanded=False):
            if grouped:
                order = ["FORM", "CHECKLIST", "PROCEDURE", "MANUAL", "POLICY", "REGULATION"]
                heading_map = {
                    "FORM": "Forms",
                    "CHECKLIST": "Checklists",
                    "PROCEDURE": "Procedures",
                    "MANUAL": "Manuals",
                    "POLICY": "Policies",
                    "REGULATION": "Regulations",
                }
                sections = []
                for doc_type in sorted(grouped, key=lambda d: (order.index(d) if d in order else len(order), d)):
                    titles = grouped[doc_type]
                    if not titles:
                        continue
                    heading = heading_map.get(doc_type, doc_type.title())
                    items = "".join(f"<li>{title}</li>" for title in titles)
                    sections.append(f"<div class='doc-type'><h4>{heading}</h4><ul>{items}</ul></div>")
                docs_html = "".join(sections)
                st.markdown(
                    f"<div class='sidebar-panel docs'>{docs_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No documents indexed yet.")
        
        # Feedback stats (ONLY for admin, not viewer)
        if not read_only_mode:
            with st.expander("📊 Feedback stats", expanded=False):
                render_feedback_stats_panel(app_state)

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
            
            # Process query with spinner
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    try:
                        result = query_with_confidence(
                            app_state,
                            trimmed,
                            retriever_type=retrieval_type,
                            auto_refine=auto_refine_option,
                            fortify=fortify_option,
                            rerank=rerank_option,
                            use_conversation_context=use_context,
                            enable_hierarchical=use_hierarchical,
                        )

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


__all__ = [
    "compose_result_markdown",
    "save_result_as_html",
    "render_app",
    "render_viewer_app",
]
