"""Streamlit UI helpers and rendering functions."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, List
import html
from markdown import markdown as md

import streamlit as st

from .config import AppConfig
from .indexing import build_index_from_library, load_cached_nodes_and_index
from .query import query_with_confidence, cohere_client
from .state import AppState
from .logger import LOGGER


_DEF_EXTS = ["extra", "tables", "fenced_code", "sane_lists"]


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
        if key.startswith("correction_toggle_") or key.startswith("correction_text_"):
            st.session_state.pop(key)


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
            source_file = src.get("source", "Unknown")
            title = src.get("title") or source_file.rsplit('.', 1)[0].replace('_', ' ')
            doc_type = (src.get("doc_type") or "").upper()
            type_label = type_labels.get(doc_type, doc_type)
            display_title = f"{title} ({type_label})" if type_label else title

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
    """DEPRECATED: Use app_state.ensure_index_loaded() instead."""
    if not app_state.ensure_index_loaded():
        st.warning("⚠️ No cached index found. Please build the index first.")


def rebuild_index(app_state: AppState) -> None:
    with st.spinner("Building index from library (this may take several minutes)..."):
        try:
            nodes, index = build_index_from_library()
            app_state.nodes = nodes
            app_state.index = index
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
            app_state.ensure_manager().nodes = nodes
            st.success(f"✅ Rebuilt index with {len(nodes)} chunks.")
            LOGGER.info("Index rebuilt successfully: %d nodes", len(nodes))
        except Exception as exc:
            LOGGER.exception("Failed to rebuild index")
            st.error(f"❌ Failed to rebuild index: {exc}")


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

    st.write("### Feedback analytics")
    st.write(f"Total feedback: {analysis['total_feedback']}")
    st.write(f"Satisfaction rate: {analysis['satisfaction_rate']:.1f}%")
    st.write(f"Incorrect rate: {analysis['incorrect_rate']:.1f}%")

    cal = analysis["confidence_calibration"]
    st.write("#### Confidence calibration")
    st.write(f"High confidence correct: {cal['high_conf_accurate']}")
    st.write(f"High confidence wrong: {cal['high_conf_wrong']}")
    st.write(f"Overconfidence rate: {cal['overconfidence_rate']:.1f}%")
    st.write(f"Low confidence correct: {cal['low_conf_accurate']}")
    st.write(f"Low confidence wrong: {cal['low_conf_wrong']}")
    st.write(f"Underconfidence rate: {cal['underconfidence_rate']:.1f}%")

    ref = analysis["query_refinement"]
    st.write("#### Query refinement")
    st.write(f"Queries refined: {ref['total_refined']}")
    st.write(f"Refinement success: {ref['refinement_success_rate']:.1f}%")

    if analysis["recommendations"]:
        st.write("#### Recommendations")
        for rec in analysis["recommendations"]:
            st.write(f"- {rec}")

    problems = app_state.feedback_system.get_problem_queries(limit=3)
    if problems:
        st.write("#### Recent problem queries")
        for idx, item in enumerate(problems, 1):
            st.write(f"{idx}. \"{item['query']}\" - {item['confidence_pct']}% ({item['confidence_level']})")
            if item.get("correction"):
                st.write(f"   User feedback: {item['correction'][:100]}")


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
        st.caption(f"{badge_emoji} **Confidence:** {conf_pct}% ({conf_level_text}) • **Sources:** {num_sources}")
        
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
        page_title="Maritime RAG Assistant",
        page_icon="⚓",
        layout="centered",  # Changed from "wide" to enable max-width control
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for centered chat layout (60% width, like Claude)
    st.markdown("""
    <style>
    /* Override Streamlit's default block-container styling */
    section.main > div.block-container {
        max-width: 800px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
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
    
    /* Ensure chat messages stay within bounds */
    div[data-testid="stChatMessageContainer"] {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    
    /* Center title and caption */
    section.main h1,
    section.main > div > div > div > p {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚓ Maritime RAG Assistant")
    st.caption("Intelligent document search powered by Gemini + LlamaIndex")

    # Ensure index is loaded
    if not app_state.ensure_index_loaded():
        st.error("⚠️ **No index found.** Please build the index first using the sidebar controls.")
        if not read_only_mode:
            if st.button("🔨 Build Index Now"):
                rebuild_index(app_state)
                _rerun_app()
        st.stop()
    
    # Ensure retrievers are ready
    if not app_state.is_ready_for_queries():
        st.error("⚠️ **System not ready.** Retrievers failed to initialize.")
        st.stop()
    
    # Load chat history
    app_state.ensure_history_loaded()
    
    # Session state defaults
    st.session_state.setdefault("retrieval_method", "hybrid")
    st.session_state.setdefault("rerank_enabled", cohere_client is not None)
    st.session_state.setdefault("fortify_option", False)
    st.session_state.setdefault("auto_refine_option", False)

    # Sidebar configuration with sticky header and scrollable sections
    with st.sidebar:
        # Custom CSS for sticky header and scrollable sections
        st.markdown("""
        <style>
        /* Make sidebar content container use flexbox */
        section[data-testid="stSidebar"] > div {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        /* Sticky header - fixed at top */
        .sidebar-sticky-header {
            flex-shrink: 0;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            background: var(--background-color);
            position: sticky;
            top: 0;
            z-index: 999;
        }
        
        /* Scrollable container */
        .sidebar-scrollable {
            flex-grow: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding-top: 1rem;
            padding-right: 0.5rem;
        }
        
        /* Custom scrollbar */
        .sidebar-scrollable::-webkit-scrollbar {
            width: 6px;
        }
        
        .sidebar-scrollable::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
        }
        
        .sidebar-scrollable::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        /* Limit documents expander content height */
        .sidebar-scrollable div[data-testid="stExpander"] > div > div {
            max-height: 300px;
            overflow-y: auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # STICKY HEADER - wrapped in a container
        st.markdown("<div class='sidebar-sticky-header'>", unsafe_allow_html=True)
        
        st.header("⚙️ Settings")
        
        if st.button("🔄 Start new chat", use_container_width=True, key="new_chat_btn"):
            _reset_chat_state(app_state)
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
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # SCROLLABLE SECTION - wrapped in scrollable container
        st.markdown("<div class='sidebar-scrollable'>", unsafe_allow_html=True)
        
        # Library management (only if not read-only)
        if not read_only_mode:
            with st.expander("📚 Library Management", expanded=False):
                if st.button("📥 Load cache", use_container_width=True):
                    load_or_warn(app_state)
                if st.button("🔨 Rebuild index", use_container_width=True):
                    rebuild_index(app_state)
                if st.button("🔄 Sync library", use_container_width=True):
                    sync_library(app_state)
        
        # Chat history
        with st.expander("💬 Chat History", expanded=False):
            if app_state.query_history:
                for idx, hist in enumerate(reversed(app_state.query_history), 1):
                    query_preview = hist.get("query", "Untitled")[:50]
                    st.caption(f"{idx}. {query_preview}...")
            else:
                st.caption("No queries yet")
            
            if st.button("🗑️ Clear history", use_container_width=True):
                app_state.clear_history()
                _rerun_app()
        
        # Documents on file - with internal scrolling
        grouped = app_state.documents_grouped_by_type()
        with st.expander("📄 Documents on file", expanded=False):
            if grouped:
                order = ["FORM", "CHECKLIST", "PROCEDURE", "MANUAL", "POLICY", "REGULATION"]
                for doc_type in sorted(grouped, key=lambda d: (order.index(d) if d in order else len(order), d)):
                    titles = grouped[doc_type]
                    if not titles:
                        continue
                    st.markdown(f"**{doc_type}** ({len(titles)})")
                    for title in titles:
                        st.caption(f"• {title}")
            else:
                st.caption("No documents indexed yet.")
        
        # Feedback stats (ONLY for admin, not viewer)
        if not read_only_mode:
            with st.expander("📊 Feedback stats", expanded=False):
                render_feedback_stats_panel(app_state)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Main chat interface
    st.markdown("---")
    
    # Render chat history using native st.chat_message
    for idx, result in enumerate(app_state.query_history):
        # User message
        with st.chat_message("user"):
            st.markdown(result.get("query", "Untitled query"))
        
        # Assistant message with feedback
        render_chat_message_with_feedback(app_state, result, idx)
    
    # Chat input
    user_prompt = st.chat_input("Ask about the maritime library...")
    
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
                        )
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