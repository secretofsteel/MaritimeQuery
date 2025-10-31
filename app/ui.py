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
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(circle at top, #0f3d58 0%, #07141d 45%, #03080c 100%);
        color: #e9f2f9;
    }
    .stApp main .block-container {
        max-width: 100%;
        padding: 2.4rem 3rem 2.8rem;
        display: flex;
        flex-direction: column;
        gap: 1.6rem;
        min-height: 100vh;
    }
    .chat-header {
        background: linear-gradient(135deg, rgba(10, 26, 36, 0.9), rgba(5, 16, 26, 0.9));
        border: 1px solid rgba(148, 226, 255, 0.18);
        border-radius: 22px;
        padding: 1.4rem 1.8rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1.6rem;
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.35);
    }
    .chat-header h1 {
        margin: 0;
        font-size: 1.8rem;
        letter-spacing: 0.02em;
        color: #f4fbff;
    }
    .chat-header p {
        margin: 0.35rem 0 0;
        color: rgba(233, 242, 249, 0.75);
        font-size: 0.95rem;
    }
    .chat-header .chat-tags {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .status-pill {
        background: linear-gradient(135deg, #0a84ff, #13c4ff);
        color: #fff;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        box-shadow: 0 12px 26px rgba(19, 196, 255, 0.35);
    }
    .status-pill.off {
        background: rgba(8, 30, 43, 0.85);
        color: rgba(233, 242, 249, 0.65);
        border: 1px solid rgba(148, 226, 255, 0.22);
        box-shadow: none;
    }
    .status-pill.muted {
        background: rgba(12, 40, 56, 0.82);
        color: rgba(233, 242, 249, 0.75);
        border: 1px solid rgba(148, 226, 255, 0.16);
        box-shadow: none;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type {
        gap: 2.2rem !important;
        align-items: stretch;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"] {
        align-items: stretch;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child {
        max-width: 360px;
        flex: 0 0 360px !important;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child > div > div[data-testid="stVerticalBlock"] {
        background: rgba(7, 24, 34, 0.9);
        border: 1px solid rgba(148, 226, 255, 0.18);
        border-radius: 24px;
        box-shadow: 0 26px 60px rgba(0, 0, 0, 0.38);
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1.2rem;
        height: calc(100vh - 13rem);
        position: sticky;
        top: 8.4rem;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:first-child {
        flex: 0 0 auto;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2) {
        flex: 1 1 auto;
        overflow-y: auto;
        padding-right: 0.4rem;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2)::-webkit-scrollbar {
        width: 6px;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:first-child > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2)::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.28);
        border-radius: 4px;
    }
    .stApp .stExpander {
        background: rgba(5, 18, 28, 0.9);
        border: 1px solid rgba(148, 226, 255, 0.14);
        border-radius: 16px;
        box-shadow: none;
    }
    .stApp .stExpander > div {
        background: transparent;
    }
    .stApp .stExpander .streamlit-expanderHeader {
        color: #cde9ff;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .chat-history-list {
        margin: 0.4rem 0 0;
        padding: 0;
        max-height: 260px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        background: rgba(4, 16, 24, 0.7);
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
    .panel-scroll {
        font-size: 0.85rem;
        line-height: 1.25rem;
        color: rgba(233, 242, 249, 0.78);
    }
    .panel-scroll::-webkit-scrollbar {
        width: 6px;
    }
    .panel-scroll::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.24);
        border-radius: 4px;
    }
    .panel-scroll.docs {
        max-height: 320px;
        overflow-y: auto;
        padding-right: 0.35rem;
    }
    .panel-scroll.docs .doc-type {
        margin-bottom: 0.75rem;
    }
    .panel-scroll.docs .doc-type h4 {
        margin: 0 0 0.3rem;
        font-size: 0.95rem;
        color: rgba(143, 211, 255, 0.9);
        letter-spacing: 0.02em;
    }
    .panel-scroll.docs .doc-type li {
        list-style: disc;
        margin-bottom: 0.2rem;
    }
    .panel-scroll.feedback {
        font-size: 0.82rem;
        line-height: 1.15rem;
        max-height: 220px;
        overflow-y: auto;
        padding-right: 0.2rem;
    }
    .panel-scroll.feedback *,
    .panel-scroll.feedback .stMarkdown p,
    .panel-scroll.feedback .stMarkdown li {
        font-size: 0.82rem !important;
        line-height: 1.2rem !important;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] {
        background: rgba(6, 18, 28, 0.92);
        border: 1px solid rgba(148, 226, 255, 0.14);
        border-radius: 30px;
        box-shadow: 0 30px 64px rgba(0, 0, 0, 0.45);
        padding: 1.9rem 2.1rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        height: calc(100vh - 13rem);
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:first-child {
        flex: 0 0 auto;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2) {
        flex: 1 1 auto;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2)::-webkit-scrollbar {
        width: 6px;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(2)::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.24);
        border-radius: 4px;
    }
    .stApp main .block-container div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:nth-child(3) {
        flex: 0 0 auto;
    }
    .chat-thread-anchor {
        display: none;
    }
    .stApp div[data-testid="stVerticalBlock"]:has(> .chat-thread-anchor) {
        display: flex;
        flex-direction: column;
        gap: 1.4rem;
    }
    .chat-message {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        max-width: min(780px, calc(100% - 120px));
    }
    .chat-message.user {
        align-self: flex-end;
        text-align: left;
    }
    .chat-message.assistant {
        align-self: stretch;
    }
    .chat-message .bubble {
        border-radius: 24px;
        padding: 1rem 1.35rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
        line-height: 1.55rem;
    }
    .chat-message.user .bubble {
        background: linear-gradient(135deg, #0a84ff, #13c4ff);
        color: #fff;
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-left: 8rem;
    }
    .chat-message.assistant .bubble {
        background: rgba(4, 16, 24, 0.78);
        border: 1px solid rgba(148, 226, 255, 0.22);
        color: #e9f2f9;
        margin-right: 2rem;
    }
    .chat-message.assistant .bubble .result-markdown {
        background: transparent;
        padding: 0;
    }
    .chat-message.assistant .bubble .result-markdown .sources-list {
        font-size: 0.82rem;
    }
    .chat-feedback-marker {
        display: none;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) {
        gap: 0.45rem;
        margin-top: 0.45rem;
        justify-content: flex-end;
        align-items: center;
        align-self: flex-end;
        margin-left: auto;
        margin-right: 0;
        flex-wrap: nowrap;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) > div[data-testid="column"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
        flex: 0 0 auto !important;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) button,
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) .stDownloadButton button {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 999px;
        border: 1px solid rgba(233, 242, 249, 0.18);
        background: rgba(5, 18, 28, 0.85);
        color: rgba(233, 242, 249, 0.9);
        font-size: 1.1rem;
        padding: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) button:hover,
    .stApp div[data-testid="stHorizontalBlock"]:has(.chat-feedback-marker) .stDownloadButton button:hover {
        background: rgba(19, 196, 255, 0.18);
        border-color: rgba(19, 196, 255, 0.45);
        color: #f6fbff;
        box-shadow: none;
    }
    .chat-feedback-correction {
        margin-top: 0.75rem;
        background: rgba(4, 16, 24, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 0.95rem;
    }
    .chat-feedback-correction textarea {
        background: rgba(8, 30, 43, 0.78) !important;
        border-radius: 12px !important;
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
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e9f2f9;
    }
    .stApp .stTextArea textarea:focus,
    .stApp .stSelectbox div[data-baseweb="select"]:focus {
        border-color: #13c4ff;
        box-shadow: 0 0 0 2px rgba(19, 196, 255, 0.25);
    }
    .stApp [data-testid="stChatInput"] textarea {
        background: rgba(8, 30, 43, 0.82);
        border-radius: 14px;
        border: 1px solid rgba(148, 226, 255, 0.24);
        color: #e9f2f9;
        font-size: 0.95rem;
    }
    .stApp [data-testid="stChatInput"] textarea:focus {
        border-color: #0a84ff;
        box-shadow: 0 0 0 2px rgba(19, 196, 255, 0.3);
    }
    .stApp [data-testid="stChatInput"] button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #1dd1a1, #10ac84);
        color: #041017;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
    }
    .stApp [data-testid="stChatInput"] button:hover {
        box-shadow: 0 0 16px rgba(29, 209, 161, 0.35);
    }
    .stApp .stMarkdown h2,
    .stApp .stMarkdown h3,
    .stApp .stMarkdown h4 {
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
    .result-markdown {
        display: flex;
        flex-direction: column;
        gap: 1.1rem;
        color: #f2f7ff;
    }
    .result-markdown h3,
    .result-markdown h4 {
        margin: 0;
        font-weight: 600;
        letter-spacing: 0.01em;
        color: #b9e0ff;
    }
    .result-markdown h3 {
        font-size: 1.05rem;
    }
    .result-markdown h4 {
        font-size: 0.95rem;
        color: #9ed4ff;
    }
    .result-markdown p,
    .result-markdown li {
        font-size: 0.95rem;
        line-height: 1.55rem;
        color: #f2f7ff;
    }
    .result-markdown .summary-grid {
        margin-top: 0.55rem;
        display: grid;
        grid-template-columns: auto 1fr;
        row-gap: 0.35rem;
        column-gap: 0.75rem;
    }
    .result-markdown .summary-row dt {
        font-weight: 600;
        color: #7acbff;
    }
    .result-markdown .summary-row dd {
        margin: 0;
        color: #f2f7ff;
    }
    .result-markdown .refinement-list {
        margin: 0.5rem 0 0;
        padding-left: 1.2rem;
        color: #f2f7ff;
    }
    .result-markdown .ref-label {
        font-weight: 600;
        color: #8dd2ff;
    }
    .result-markdown .answer-body {
        font-size: 0.95rem;
        line-height: 1.6rem;
        background: rgba(4, 16, 24, 0.85);
        border: 1px solid rgba(148, 226, 255, 0.18);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        color: #f6fbff;
    }
    .result-markdown .answer-body * {
        color: inherit;
    }
    .result-markdown .sources-list {
        font-size: 0.95rem;
        line-height: 1.4rem;
        color: #ddeeff;
        margin: 0;
        padding-left: 1.2rem;
    }
    .result-markdown .sources-list li {
        margin-bottom: 0.35rem;
    }
    .result-markdown .source-location {
        font-size: 12px;
        opacity: 0.82;
        color: #c5daff;
    }
    .result-markdown .confidence-note {
        font-size: 0.9rem;
        opacity: 0.75;
        font-style: italic;
        color: #f2f7ff;
    }
    .result-markdown .empty-sources {
        opacity: 0.75;
        color: #d5e9ff;
    }
    .result-sources details {
        background: rgba(5, 18, 28, 0.78);
        border: 1px solid rgba(148, 226, 255, 0.2);
        border-radius: 16px;
        padding: 0.8rem 1rem;
    }
    .result-sources details summary {
        cursor: pointer;
        list-style: none;
        font-weight: 600;
        color: #9ed4ff;
        font-size: 0.95rem;
    }
    .result-sources details summary::-webkit-details-marker {
        display: none;
    }
    .result-sources details[open] summary {
        color: #c3e7ff;
    }
</style>
"""

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
    parts.append("<details class='sources-details' open>")
    parts.append("<summary>Sources</summary>")
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
    parts.append("</details>")
    parts.append("</section>")

    if confidence_note:
        parts.append(f"<p class='confidence-note'>Note: {html.escape(confidence_note)}</p>")

    parts.append("</div>")
    return "\n".join(parts)



def apply_custom_theme() -> None:
    st.markdown(_CUSTOM_THEME, unsafe_allow_html=True)


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
    controls = st.columns(4, gap="small")

    export_html = build_result_export_html(result)
    file_name = f"query_result_{key_prefix}.html"
    controls[0].download_button(
        "⬇️",
        data=export_html,
        file_name=file_name,
        mime="text/html",
        key=f"export_{key_prefix}",
        help="Download this response as an HTML file.",
    )

    if controls[1].button(
        "👍",
        key=f"helpful_{key_prefix}",
        help="Mark this response as helpful.",
    ):
        app_state.feedback_system.log_feedback(result, "helpful")
        st.success("Thanks! Feedback recorded.")

    if controls[2].button(
        "👎",
        key=f"not_helpful_{key_prefix}",
        help="Mark this response as not helpful.",
    ):
        app_state.feedback_system.log_feedback(result, "not_helpful")
        st.info("Thanks! Feedback recorded.")

    incorrect_key = f"incorrect_{key_prefix}"
    correction_toggle_key = f"correction_toggle_{key_prefix}"
    if controls[3].button(
        "⚠️",
        key=incorrect_key,
        help="Report an issue and provide a correction.",
    ):
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

    retrieval_type = st.session_state["retrieval_method"]
    rerank_option = st.session_state["rerank_enabled"]
    fortify_option = st.session_state["fortify_option"]
    auto_refine_option = st.session_state["auto_refine_option"]

    layout_block = st.container()
    with layout_block:
        control_col, chat_col = st.columns([0.38, 0.62], gap="large")

        with control_col:
            panel_wrapper = st.container()
            with panel_wrapper:
                top_panel = st.container()
                with top_panel:
                    if st.button("Start new chat", use_container_width=True):
                        st.session_state["__chat_reset_requested"] = True
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
                            help="Re-rank top candidates with Cohere for improved relevance."
                            if rerank_available
                            else "Cohere key not configured.",
                            disabled=rerank_disabled,
                        )
                        if rerank_disabled:
                            rerank_option = False
                            st.session_state["rerank_enabled"] = False

                        fortify_option = st.checkbox(
                            "Fortify query with Gemini",
                            key="fortify_option",
                        )
                        auto_refine_option = st.checkbox(
                            "Auto-refine low confidence queries",
                            key="auto_refine_option",
                        )

                scroll_panel = st.container()
                with scroll_panel:
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
                                f"<div class='panel-scroll docs'>{docs_html}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("No documents indexed yet.")

                    with st.expander("Feedback stats", expanded=False):
                        st.markdown("<div class='panel-scroll feedback'>", unsafe_allow_html=True)
                        render_feedback_stats_panel(app_state)
                        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.pop("__chat_reset_requested", False):
            _reset_chat_state(app_state)
            _rerun_app()

        with chat_col:
            chat_wrapper = st.container()
            with chat_wrapper:
                header_section = st.container()
                body_section = st.container()
                input_section = st.container()

                with header_section:
                    retrieval_label = retrieval_type.title()
                    rerank_class = "status-pill" if rerank_option else "status-pill off"
                    fortify_class = "status-pill" if fortify_option else "status-pill muted"
                    refine_class = "status-pill" if auto_refine_option else "status-pill muted"
                    st.markdown(
                        f"""
                        <div class="chat-header">
                            <div>
                                <h1>Maritime RAG Assistant</h1>
                                <p>Modular Streamlit interface powered by Gemini + LlamaIndex</p>
                            </div>
                            <div class="chat-tags">
                                <span class="status-pill">{retrieval_label} retrieval</span>
                                <span class="{rerank_class}">Rerank {'on' if rerank_option else 'off'}</span>
                                <span class="{fortify_class}">Fortify {'on' if fortify_option else 'off'}</span>
                                <span class="{refine_class}">Auto refine {'on' if auto_refine_option else 'off'}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with body_section:
                    chat_placeholder = st.container()

                def draw_chat_thread() -> None:
                    placeholder = chat_placeholder.empty()
                    with placeholder.container():
                        if app_state.query_history:
                            st.markdown("<div class='chat-thread-anchor'></div>", unsafe_allow_html=True)
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
                        else:
                            st.info("Ask a question to get started.")

                draw_chat_thread()

                with input_section:
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
                                    draw_chat_thread()
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



