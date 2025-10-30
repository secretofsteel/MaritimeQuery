"""Streamlit UI helpers and rendering functions."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, List
import html

import streamlit as st

from .config import AppConfig
from .indexing import build_index_from_library, load_cached_nodes_and_index
from .query import query_with_confidence, cohere_client
from .state import AppState


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
    _rerun_app()


def hide_streamlit_branding() -> None:
    """Remove Streamlit header/footer branding."""
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

_CUSTOM_THEME = """
<style>
    .stApp {
        background: radial-gradient(circle at top, #0f3d58 0%, #07141d 45%, #03080c 100%);
        color: #e9f2f9;
    }
    .stApp [data-testid="stSidebar"] {
        background: rgba(10, 26, 36, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        width: 350px;
        min-width: 350px;
        max-width: 350px;
        flex: 0 0 350px;
    }
    .stApp [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
        width: 100%;
    }
    .stApp [data-testid="stSidebar"] .stButton {
        width: 100%;
    }
    .stApp [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        display: inline-flex;
        justify-content: center;
    }
    .sidebar-top {
        position: sticky;
        top: 0;
        z-index: 5;
        background: rgba(10, 26, 36, 0.95);
        padding-bottom: 1rem;
    }
    .sidebar-scroll {
        max-height: calc(100vh - 5.2rem);
        overflow-y: auto;
        padding-right: 0.35rem;
    }
    .sidebar-scroll::-webkit-scrollbar {
        width: 6px;
    }
    .sidebar-scroll::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.28);
        border-radius: 4px;
    }
    .chat-history-list {
        margin: 0.4rem 0 0;
        padding: 0;
        max-height: 260px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        background: rgba(4, 16, 24, 0.65);
    }
    .chat-history-list ol {
        list-style: decimal;
        padding: 0.85rem 1.25rem;
        margin: 0;
        color: rgba(233, 242, 249, 0.85);
        font-size: 0.9rem;
        line-height: 1.45rem;
    }
    .chat-history-list ol li {
        margin-bottom: 0.35rem;
        word-break: break-word;
    }
    .chat-history-list::-webkit-scrollbar {
        width: 6px;
    }
    .chat-history-list::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.24);
        border-radius: 4px;
    }
    .chat-thread {
        display: flex;
        flex-direction: column;
        gap: 1.6rem;
        margin-top: 1.5rem;
        padding-bottom: 4rem;
    }
    .chat-message {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        max-width: min(780px, calc(100% - 140px));
    }
    .chat-message.user {
        align-self: flex-end;
        text-align: left;
    }
    .chat-message.assistant {
        align-self: flex-start;
    }
    .chat-message .bubble {
        border-radius: 22px;
        padding: 1rem 1.3rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
        line-height: 1.55rem;
    }
    .chat-message.user .bubble {
        background: linear-gradient(135deg, #0a84ff, #13c4ff);
        color: #fff;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .chat-message.assistant .bubble {
        background: rgba(7, 24, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e9f2f9;
    }
    .chat-message.assistant .bubble .result-markdown {
        background: transparent;
        padding: 0;
    }
    .chat-message.assistant .bubble .result-markdown .sources-list {
        font-size: 0.78rem;
    }
    .chat-feedback-marker {
        display: none;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) {
        gap: 0.6rem;
        margin-top: 0.75rem;
        justify-content: flex-start;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) > div[data-testid="column"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) button {
        width: 100%;
        border-radius: 999px;
        border: 1px solid rgba(19, 196, 255, 0.35);
        background: rgba(19, 196, 255, 0.16);
        color: #e9f2f9;
        font-size: 0.82rem;
        padding: 0.45rem 0.6rem;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) button:hover {
        background: rgba(19, 196, 255, 0.35);
        box-shadow: 0 0 14px rgba(19, 196, 255, 0.25);
    }
    .chat-feedback-correction {
        margin-top: 0.75rem;
        background: rgba(4, 16, 24, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 0.9rem;
    }
    .chat-feedback-correction textarea {
        background: rgba(8, 30, 43, 0.78) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #e9f2f9 !important;
        font-size: 0.9rem !important;
    }
    .chat-feedback-correction .stButton>button {
        width: 100%;
        border-radius: 999px;
        background: linear-gradient(135deg, #1dd1a1, #10ac84);
        border: none;
        color: #041017;
        font-weight: 600;
    }
    .chat-feedback-correction .stButton>button:hover {
        box-shadow: 0 0 16px rgba(29, 209, 161, 0.35);
    }
    .stApp .stButton>button {
        border-radius: 999px;
        border: none;
        padding: 0.6rem 1.6rem;
        background: linear-gradient(135deg, #0a84ff, #13c4ff);
        color: #fff;
        font-weight: 600;
    }
    .stApp .stButton>button:hover {
        box-shadow: 0 0 18px rgba(19, 196, 255, 0.35);
    }
    .stApp .stTextArea textarea,
    .stApp .stSelectbox div[data-baseweb="select"] {
        background: rgba(8, 30, 43, 0.68);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e9f2f9;
    }
    .stApp .stTextArea textarea:focus,
    .stApp .stSelectbox div[data-baseweb="select"]:focus {
        border-color: #13c4ff;
        box-shadow: 0 0 0 2px rgba(19, 196, 255, 0.25);
    }
    .stApp .stMarkdown h2, .stApp .stMarkdown h3, .stApp .stMarkdown h4 {
        color: #f2f9ff;
    }
    .stApp .stAlert {
        border-radius: 12px;
        background: rgba(8, 30, 43, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stApp .stDownloadButton button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #1dd1a1, #10ac84);
        color: #041017;
        font-weight: 600;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100%;
    }
    .config-panel {
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 1rem;
        background: rgba(7, 24, 34, 0.6);
        font-size: 0.9rem;
        line-height: 1.4rem;
        color: rgba(233, 242, 249, 0.85);
    }
    .config-panel .label {
        font-weight: 600;
        color: #8fd3ff;
    }
    .result-markdown p,
    .result-markdown li {
        font-size: 20px;
        line-height: 1.8rem;
    }
    .result-markdown .sources-list {
        font-size: 14px;
        line-height: 1.35rem;
        color: rgba(233, 242, 249, 0.85);
        margin: 0;
        padding-left: 1.4rem;
    }
    .result-markdown .sources-list li {
        margin-bottom: 0.35rem;
    }
    .result-markdown .source-location {
        font-size: 12px;
        opacity: 0.78;
    }
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
    .sidebar-panel.feedback {
        max-height: 220px;
        overflow-y: auto;
        font-size: 0.82rem;
        line-height: 1.15rem;
        padding-right: 0.2rem;
    }
    .sidebar-panel.feedback *,
    .sidebar-panel.feedback .stMarkdown p,
    .sidebar-panel.feedback .stMarkdown li {
        font-size: 0.82rem !important;
        line-height: 1.2rem !important;
    }
    .sidebar-panel.feedback h3,
    .sidebar-panel.feedback h4 {
        font-size: 0.9rem !important;
        margin-bottom: 0.35rem !important;
    }
</style>
"""

_GITHUBISH_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif; margin: 0; padding: 0; }
.container { max-width: 900px; margin: 2rem auto; padding: 2rem; }
.markdown-body { line-height: 1.75; font-size: 18px; }
.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 { margin-top: 1.5em; }
.markdown-body h3 { border-bottom: 1px solid rgba(127,127,127,.25); padding-bottom: .3em; }
.markdown-body pre { background: rgba(127,127,127,.1); padding: 1rem; overflow: auto; border-radius: 8px; }
.markdown-body code { padding: .1em .3em; background: rgba(127,127,127,.15); border-radius: 4px; }
.markdown-body blockquote { margin: 1em 0; padding: .5em 1em; border-left: 4px solid rgba(127,127,127,.4); background: rgba(127,127,127,.08); border-radius: 4px; }
.markdown-body table { border-collapse: collapse; width: 100%; margin: 1em 0; }
.markdown-body th, .markdown-body td { border: 1px solid rgba(127,127,127,.3); padding: .5em .75em; }
hr { border: 0; border-top: 1px solid rgba(127,127,127,.25); margin: 2rem 0; }
.smallmeta { color: rgba(127,127,127,.9); font-size: 12px; margin-top: 2rem; }
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


def ensure_markdown_deps() -> None:
    try:  # pragma: no cover - dependency management
        import markdown  # noqa: F401
    except Exception:  # pragma: no cover - dependency management
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"], stdout=subprocess.DEVNULL)


def compose_result_markdown(result: Dict) -> str:
    original_query = result.get("query", "No original query")
    conf_pct = result.get("confidence_pct", "N/A")
    conf_level = result.get("confidence_level", "N/A")
    num_sources = result.get("num_sources", "N/A")
    attempts = result.get("attempts")
    best_attempt = result.get("best_attempt")
    refinement_history = result.get("refinement_history", [])
    answer = result.get("answer", "No answer available.")
    sources = result.get("sources", [])
    confidence_note = result.get("confidence_note", "")

    header_md = f"""
### ðŸ§­ Query Result

- **Query:** {original_query}
- **Confidence:** {conf_pct}% ({conf_level})
- **Sources analyzed:** {num_sources}
"""
    if refinement_history:
        header_md += f"**Query refinements:** {attempts} attempts  \n"
        header_md += f"**Best result:** Attempt {best_attempt}\n"

    if refinement_history:
        refinements_md = "\n#### ðŸ”„ Query Refinement History\n"
        refinements_md += f"- Original: â€œ{result.get('query','')}â€\n"
        for item in refinement_history[1:]:
            refinements_md += f"- Attempt {item.get('attempt')}: â€œ{item.get('query')}â€ (confidence: {item.get('confidence')}%)\n"
        if best_attempt:
            refinements_md += f"- âœ… Best: Attempt {best_attempt} (confidence: {conf_pct}%)\n"
    else:
        refinements_md = ""

    answer_md = f"""
#### ðŸ“ Answer
{answer}
"""

    if sources:
        sources_md = "\n#### ðŸ“š Sources\n<ol class=\"sources-list\">"
        for idx, src in enumerate(sources[:5], 1):
            source_file = src.get("source", "Unknown")
            title = src.get("title", "")
            doc_type = src.get("doc_type", "")
            hierarchy = src.get("hierarchy", "")
            section = src.get("section", "")
            tab_name = src.get("tab_name", "")
            form_number = src.get("form_number", "")
            form_category_name = src.get("form_category_name", "")

            type_labels = {
                "PROC": "PROCEDURE",
                "FORM": "FORM",
                "REG": "REGULATION",
                "POLICY": "POLICY",
                "REPORT": "REPORT",
                "CHECKLIST": "CHECKLIST",
            }
            type_label = type_labels.get(doc_type, doc_type.upper() if doc_type else "")

            display_title = title if title else source_file.rsplit(".", 1)[0].replace("_", " ")
            title_line = display_title
            if type_label:
                title_line += f" ({type_label})"

            if tab_name:
                location = f"Tab: {tab_name}"
            elif hierarchy and hierarchy.strip():
                location = hierarchy
            elif doc_type == "FORM":
                if form_number and form_category_name:
                    location = f"Form {form_number} - {form_category_name}"
                elif form_number:
                    location = f"Form {form_number}"
                elif section and section not in ["Document Content", ""]:
                    clean_section = section.split("|")[0].strip() if "|" in section else section
                    location = clean_section[:60] if len(clean_section) > 60 else clean_section
                else:
                    location = "Form document"
            else:
                if section and section not in ["Document Content", "Main document", ""]:
                    clean_section = section.split("|")[0].strip() if "|" in section else section
                    location = clean_section[:60] if len(clean_section) > 60 else clean_section
                else:
                    location = "Main document"

            truncated_location = (location[:147] + "...") if len(location) > 150 else location
            sources_md += (
                f"<li><strong>{idx}. {title_line}</strong><br><span class='source-location'>Location: {truncated_location}</span></li>"
            )
        sources_md += "</ol>"
    else:
        sources_md = "\n_No sources available._\n"

    note_md = f"\n> âš ï¸ {confidence_note}\n" if confidence_note else ""
    return (
        "<div class='result-markdown'>\n"
        f"{header_md}\n---\n{refinements_md}\n{answer_md}\n---\n{sources_md}\n---\n{note_md}\n"
        "</div>"
    )


def apply_custom_theme() -> None:
    st.markdown(_CUSTOM_THEME, unsafe_allow_html=True)


def build_result_export_html(result: Dict, title: str = "RAG Query Result") -> str:
    """Generate an HTML export representation for a query result."""
    ensure_markdown_deps()
    from markdown import markdown

    markdown_body = compose_result_markdown(result)
    html_body = markdown(markdown_body, extensions=_DEF_EXTS)
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
    nodes, index = load_cached_nodes_and_index()
    if nodes and index:
        app_state.nodes = nodes
        app_state.index = index
        app_state.vector_retriever = None
        app_state.bm25_retriever = None
        app_state.ensure_retrievers()
        app_state.ensure_manager().nodes = nodes
        st.success(f"Loaded {len(nodes)} cached nodes.")
    else:
        st.warning("No cached index found. Rebuild to initialize.")


def rebuild_index(app_state: AppState) -> None:
    with st.spinner("Building index from library (this may take several minutes)..."):
        nodes, index = build_index_from_library()
    app_state.nodes = nodes
    app_state.index = index
    app_state.vector_retriever = None
    app_state.bm25_retriever = None
    app_state.ensure_retrievers()
    app_state.ensure_manager().nodes = nodes
    st.success(f"Rebuilt index with {len(nodes)} chunks.")


def sync_library(app_state: AppState) -> None:
    manager = app_state.ensure_manager()
    manager.nodes = app_state.nodes
    with st.spinner("Syncing library..."):
        changes = manager.sync_library(app_state.index)
    st.success(
        f"Sync complete. Added {len(changes.added)}, modified {len(changes.modified)}, deleted {len(changes.deleted)}."
    )
    app_state.nodes = manager.nodes
    app_state.vector_retriever = None
    app_state.bm25_retriever = None
    app_state.ensure_retrievers()


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


def render_chat_feedback_row(app_state: AppState, result: Dict, key_prefix: str) -> None:
    """Render inline export + feedback controls for a chat response."""
    st.markdown("<div class='chat-feedback-marker'></div>", unsafe_allow_html=True)
    controls = st.columns([1, 1, 1, 1], gap="small")

    export_html = build_result_export_html(result)
    file_name = f"query_result_{key_prefix}.html"
    controls[0].download_button(
        "\u2193 Download",
        data=export_html,
        file_name=file_name,
        mime="text/html",
        key=f"export_{key_prefix}",
    )

    if controls[1].button("👍 Helpful", key=f"helpful_{key_prefix}"):
        app_state.feedback_system.log_feedback(result, "helpful")
        st.success("Thanks! Feedback recorded.")

    if controls[2].button("👎 Needs work", key=f"not_helpful_{key_prefix}"):
        app_state.feedback_system.log_feedback(result, "not_helpful")
        st.info("Thanks! Feedback recorded.")

    incorrect_key = f"incorrect_{key_prefix}"
    correction_toggle_key = f"correction_toggle_{key_prefix}"
    if controls[3].button("⚠️ Incorrect", key=incorrect_key):
        st.session_state[correction_toggle_key] = not st.session_state.get(correction_toggle_key, False)

    if st.session_state.get(correction_toggle_key, False):
        st.markdown("<div class='chat-feedback-correction'>", unsafe_allow_html=True)
        correction = st.text_area(
            "What was wrong? What should the answer be?",
            key=f"correction_text_{key_prefix}",
            placeholder="Tell us what needs to be fixed...",
            label_visibility="collapsed",
        )
        if st.button("Submit correction", key=f"submit_correction_{key_prefix}"):
            if correction.strip():
                app_state.feedback_system.log_feedback(result, "incorrect", correction.strip())
                st.success("Thank you! Your correction has been recorded.")
                st.session_state[correction_toggle_key] = False
                st.session_state.pop(f"correction_text_{key_prefix}", None)
            else:
                st.warning("Please provide correction details before submitting.")
        st.markdown("</div>", unsafe_allow_html=True)

def render_app(
    app_state: AppState,
    *,
    allow_library_controls: bool = True,
    rerank_checkbox_disabled: bool = False,
) -> None:
    st.set_page_config(page_title="Maritime RAG Assistant", layout="wide")
    apply_custom_theme()
    st.title("Maritime RAG Assistant")
    st.caption("Modular Streamlit interface powered by Gemini + LlamaIndex")

    paths = AppConfig.get().paths

    if not hasattr(app_state, "ensure_history_loaded"):
        app_state = AppState()
        st.session_state["app_state"] = app_state

    if "sidebar_loaded_cache" not in st.session_state:
        load_or_warn(app_state)
        st.session_state["sidebar_loaded_cache"] = True

    app_state.ensure_history_loaded()

    st.session_state.setdefault("retrieval_method", "hybrid")
    st.session_state.setdefault("rerank_enabled", cohere_client is not None)
    st.session_state.setdefault("fortify_option", False)
    st.session_state.setdefault("auto_refine_option", False)

    top_container = st.sidebar.container()
    with top_container:
        st.markdown("<div class='sidebar-top'>", unsafe_allow_html=True)
        if st.button("Start new chat", use_container_width=True):
            _reset_chat_state(app_state)
            _rerun_app()
        with st.expander("Assistant options", expanded=True):
            retrieval_type = st.selectbox(
                "Retrieval method",
                ["hybrid", "vector", "bm25"],
                key="retrieval_method",
            )

            rerank_available = cohere_client is not None
            rerank_disabled = rerank_checkbox_disabled or not rerank_available
            rerank_option = st.checkbox(
                "Enable reranking (Cohere)",
                key="rerank_enabled",
                help="Re-rank top candidates with Cohere for improved relevance." if rerank_available else "Cohere key not configured.",
                disabled=rerank_disabled,
            )
            if rerank_disabled:
                rerank_option = False
                st.session_state["rerank_enabled"] = False

            fortify_option = st.checkbox("Fortify query with Gemini", key="fortify_option")
            auto_refine_option = st.checkbox("Auto-refine low confidence queries", key="auto_refine_option")
        st.markdown("</div>", unsafe_allow_html=True)

    scroll_container = st.sidebar.container()
    with scroll_container:
        st.markdown("<div class='sidebar-scroll'>", unsafe_allow_html=True)

        with st.expander("Chat history", expanded=True):
            history_entries = app_state.history_log
            if history_entries:
                items: List[str] = []
                for result in reversed(history_entries):
                    label = result.get("query", "Untitled query").strip() or "Untitled query"
                    safe_label = label if len(label) <= 120 else f"{label[:117]}..."
                    items.append(f"<li>{safe_label}</li>")
                list_html = "<div class='chat-history-list'><ol>{}</ol></div>".format("".join(items))
                st.markdown(list_html, unsafe_allow_html=True)
            else:
                st.caption("No queries yet.")

            if st.button("Clear history", key="clear_history_button", use_container_width=True):
                app_state.clear_history()
                _rerun_app()

        if allow_library_controls:
            with st.expander("Library management", expanded=False):
                button_kwargs = {"use_container_width": True}
                if st.button("Load cache", **button_kwargs):
                    load_or_warn(app_state)
                if st.button("Rebuild index", **button_kwargs):
                    rebuild_index(app_state)
                if st.button("Sync library", **button_kwargs):
                    sync_library(app_state)

            with st.expander("Customize paths", expanded=False):
                docs_input = st.text_input("Documents directory", value=str(paths.docs_path))
                chroma_input = st.text_input("Chroma directory", value=str(paths.chroma_path))
                cache_input = st.text_input("Cache directory", value=str(paths.cache_dir))
                if st.button("Apply paths", use_container_width=True):
                    try:
                        AppConfig.get().update_paths(Path(docs_input), Path(chroma_input), Path(cache_input))
                        st.success("Paths updated. Reload or resync to apply changes.")
                        _rerun_app()
                    except Exception as exc:  # pragma: no cover - defensive path
                        st.error(f"Failed to update paths: {exc}")

        grouped = app_state.documents_grouped_by_type()
        with st.expander("Documents on file", expanded=False):
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

        with st.expander("Feedback stats", expanded=False):
            st.markdown("<div class='sidebar-panel feedback'>", unsafe_allow_html=True)
            render_feedback_stats_panel(app_state)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    main_col = st.container()

    with main_col:
        chat_container = st.container()
        with chat_container:
            if app_state.query_history:
                st.markdown("<div class='chat-thread'>", unsafe_allow_html=True)
                for idx, result in enumerate(app_state.query_history):
                    query_text = html.escape(result.get("query", "").strip() or "Untitled query")
                    st.markdown(
                        f"<div class='chat-message user'><div class='bubble'>{query_text}</div></div>",
                        unsafe_allow_html=True,
                    )
                    assistant_body = compose_result_markdown(result)
                    st.markdown(
                        f"<div class='chat-message assistant'><div class='bubble'>{assistant_body}</div></div>",
                        unsafe_allow_html=True,
                    )
                    render_chat_feedback_row(app_state, result, key_prefix=str(idx))
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Ask a question to get started.")

        user_prompt = st.chat_input("Ask about the maritime library...")
        if user_prompt is not None:
            trimmed = user_prompt.strip()
            if not trimmed:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Searching..."):
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
                        _rerun_app()
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")

    if app_state.query_history:
        app_state.last_result = app_state.query_history[-1]
    else:
        app_state.last_result = None


def render_viewer_app(app_state: AppState) -> None:
    """Restricted UI for read-only testing of retrieval."""
    hide_streamlit_branding()
    if not app_state.nodes or not app_state.index:
        load_or_warn(app_state)
    render_app(
        app_state,
        allow_library_controls=False,
        rerank_checkbox_disabled=False,
    )


__all__ = [
    "compose_result_markdown",
    "save_result_as_html",
    "render_app",
    "render_viewer_app",
]



