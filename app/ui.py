"""Streamlit UI helpers and rendering functions."""
from __future__ import annotations
import datetime
import os
import json
from pathlib import Path
from typing import Dict, List
import html
from markdown import markdown as md
import streamlit as st
from .config import AppConfig
from .indexing import build_index_from_library, load_cached_nodes_and_index
from .query import query_with_confidence, cohere_client
from .state import AppState
from .feedback import FeedbackSystem
from .constants import FORM_CATEGORIES # Assuming you might want to display category codes

# --- Constants ---
_DEF_EXTS = ["extra", "tables", "fenced_code", "sane_lists"]

# --- Helper Functions ---

def _rerun_app() -> None:
    """Compat helper for rerunning the Streamlit script across versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def apply_custom_theme() -> None:
    """Apply the cohesive dark theme CSS."""
    st.markdown(_DARK_THEME_CSS, unsafe_allow_html=True)

def compose_result_markdown(result: Dict) -> str:
    """Generate Markdown for a single query result, including sources and metadata."""
    # --- Header Section ---
    header_md = f"### 📋 Query Result\n\n"
    header_md += f"**Query:** {result.get('query', 'N/A')}\n\n"
    header_md += f"**Confidence:** {result.get('confidence_pct', 0):.1f}% ({result.get('confidence_level', 'N/A')})\n\n"
    header_md += f"**Retrieved Sources:** {len(result.get('sources', []))}\n\n"
    header_md += f"**Retrieved Nodes:** {result.get('num_sources', 0)}\n\n"
    header_md += f"**Retrieval Method:** {result.get('retriever_type', 'N/A')}\n\n"
    header_md += f"**Total Attempts:** {result.get('attempts', 1)}\n\n"

    # --- Refinement History Section ---
    refinement_history = result.get('refinement_history', [])
    if refinement_history and len(refinement_history) > 1: # Only show if there were attempts beyond the first
        header_md += "#### 🔄 Query Refinement History\n"
        original_query = result.get('query', 'N/A')
        header_md += f"- **Original:** \"{original_query}\"\n"
        # Start from the second attempt (index 1)
        for ref in refinement_history[1:]:
            attempt_num = ref.get('attempt', 'N/A')
            query_text = ref.get('query', 'N/A')
            conf = ref.get('confidence', 'N/A')
            header_md += f"- **Attempt {attempt_num}:** \"{query_text}\" (confidence: {conf}%)\n"
        # Indicate the best attempt if applicable
        best_attempt_num = result.get('best_attempt', None)
        if best_attempt_num:
            conf_pct = result.get('confidence_pct', 0)
            header_md += f"- ✅ **Best:** Attempt {best_attempt_num} (confidence: {conf_pct}%)\n\n"
    else:
        header_md += "\n" # Add space if no refinement history

    # --- Answer Section ---
    answer = result.get('answer', 'No answer generated.')
    answer_md = f"#### 📝 Answer\n\n{answer}\n\n"

    # --- Sources Section ---
    sources = result.get('sources', [])
    if sources:
        num_sources_to_show = min(5, len(sources)) # Show top 5 sources by default
        sources_md = f"#### 📚 Top {num_sources_to_show} Sources Analysed\n\n"
        sources_md += "<details>\n<summary>Click here to see sources</summary>\n\n"
        for i, source in enumerate(sources[:num_sources_to_show]):
            sources_md += f"**Source {i+1}**\n\n"
            sources_md += f"- **File:** `{source.get('source', 'N/A')}`\n"
            sources_md += f"- **Section:** `{source.get('section', 'N/A')}`\n"
            sources_md += f"- **Page:** `{source.get('page_label', 'N/A')}`\n" # Use page_label if available
            sources_md += f"- **Score:** `{source.get('score', 'N/A'):.4f}`\n"
            # Add category if available and map it
            category_code = source.get('category', 'N/A')
            if category_code in FORM_CATEGORIES:
                 category_name = FORM_CATEGORIES[category_code]
                 sources_md += f"- **Category:** `{category_code} - {category_name}`\n"
            else:
                 sources_md += f"- **Category:** `{category_code}`\n" # Fallback if code not in map
            sources_md += f"- **Content Preview:**\n```\n{source.get('content_preview', 'N/A')[:200]}...\n```\n\n" # Limit preview length
        sources_md += "</details>\n\n"
    else:
        sources_md = "**No sources retrieved.**\n\n"

    # Combine all sections
    full_md = header_md + answer_md + sources_md
    return full_md

def export_result_to_html(result: Dict, out_dir: Path) -> str | None:
    """Export a single query result as a self-contained HTML file."""
    # Convert Markdown to HTML
    md_content = compose_result_markdown(result)
    html_body = md(md_content, extensions=_DEF_EXTS)



    # Define the HTML template
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Maritime RAG Result - {result.get('query', 'N/A')[:50]}...</title>
    <style>{_GITHUBISH_CSS}</style>
</head>
<body>
    <div class="container markdown-body">
        {html_body}
        <hr />
        <div class="smallmeta">
            Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            Query: {html.escape(result.get('query', 'N/A'))}
        </div>
    </div>
</body>
</html>"""

    # Determine output filename
    safe_query = "".join(c for c in result.get('query', 'unnamed') if c.isalnum() or c in (' ', '-', '_')).rstrip()
    base_name = safe_query[:50] if safe_query else "result"
    ext = ".html"
    counter = 1
    new_output_file = f"{base_name}{ext}"
    while (out_dir / new_output_file).exists():
        new_output_file = f"{base_name}({counter}){ext}"
        counter += 1

    export_path = out_dir / new_output_file
    export_path.write_text(full_html, encoding="utf-8")
    return str(export_path)

def load_or_warn(app_state: AppState) -> None:
    """Load cached index or warn the user."""
    nodes, index = load_cached_nodes_and_index()
    if nodes and index:
        app_state.nodes = nodes
        app_state.index = index
        app_state.vector_retriever = None  # Reset retrievers to force recreation
        app_state.bm25_retriever = None
        app_state.ensure_retrievers() # Ensure retrievers are created after loading index
        app_state.ensure_manager().nodes = nodes # Update manager's node list
        st.success(f"✅ Loaded {len(nodes)} cached nodes and index.")
    else:
        st.warning("⚠️ No cached index found. Rebuild or sync to initialize the system.")

def rebuild_index(app_state: AppState) -> None:
    """Rebuild the index from the library."""
    with st.spinner("🔄 Building index from library (this may take several minutes)..."):
        try:
            nodes, index = build_index_from_library()
            app_state.nodes = nodes
            app_state.index = index
            app_state.vector_retriever = None  # Reset retrievers
            app_state.bm25_retriever = None
            app_state.ensure_retrievers() # Ensure retrievers are created after rebuilding index
            app_state.ensure_manager().nodes = nodes # Update manager's node list
            st.success(f"✅ Rebuilt index with {len(nodes)} chunks.")
        except Exception as e:
            st.error(f"❌ Failed to rebuild index: {e}")
            import traceback
            traceback.print_exc()

def sync_library(app_state: AppState) -> None:
    """Sync the library using the IncrementalIndexManager."""
    manager = app_state.ensure_manager()
    manager.nodes = app_state.nodes # Ensure manager has the latest nodes from app_state

    with st.spinner("🔄 Syncing library..."):
        try:
            changes = manager.sync_library(app_state.index)
            st.success(f"✅ Sync complete. Added: {len(changes.added)}, Modified: {len(changes.modified)}, Deleted: {len(changes.deleted)}.")
            # Update app_state nodes after sync
            app_state.nodes = manager.nodes
            # Reset and recreate retrievers after sync
            app_state.vector_retriever = None
            app_state.bm25_retriever = None
            app_state.ensure_retrievers()
        except Exception as e:
            st.error(f"❌ Failed to sync library: {e}")
            import traceback
            traceback.print_exc()

# --- UI Rendering Functions ---

def render_library_controls(app_state: AppState) -> None:
    """Render controls for library management in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("📚 Library Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Sync Library", use_container_width=True):
            with st.spinner("Syncing library..."):
                try:
                    changes = app_state.ensure_manager().sync_library(app_state.index)
                    st.success(f"Sync complete!\n- Added: {len(changes.added)}\n- Modified: {len(changes.modified)}\n- Deleted: {len(changes.deleted)}")
                    _rerun_app()
                except Exception as e:
                    st.error(f"Sync failed: {e}")
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            app_state.clear_history()
            st.success("History cleared!")

    if st.button("💾 Save Index Now"):
        try:
            app_state.ensure_manager()._save_nodes_pickle()
            st.success("Index saved to cache!")
        except Exception as e:
            st.error(f"Failed to save index: {e}")

def render_document_list(app_state: AppState) -> None:
    """Render the list of indexed documents in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("📋 Indexed Documents")

    if not app_state.nodes:
        st.sidebar.info("No documents indexed yet.")
        return

    # Group nodes by source filename
    doc_info_map = {}
    for node in app_state.nodes:
        source = node.metadata.get("source", "Unknown")
        if source not in doc_info_map:
            doc_info_map[source] = {
                "sections": set(),
                "pages": set(),
                "categories": set(),
                "chunk_count": 0
            }
        doc_info_map[source]["sections"].add(node.metadata.get("section", "General"))
        doc_info_map[source]["pages"].add(node.metadata.get("page_label", "N/A")) # Use page_label
        doc_info_map[source]["categories"].add(node.metadata.get("category", "N/A"))
        doc_info_map[source]["chunk_count"] += 1

    # Display document info
    for source, info in doc_info_map.items():
        with st.sidebar.expander(f"📄 {Path(source).name}"):
            st.markdown(f"**Chunks:** {info['chunk_count']}")
            categories_str = ", ".join(sorted(info['categories']))
            if categories_str != "N/A":
                 # Map category codes to names if possible
                 mapped_names = []
                 for code in sorted(info['categories']):
                     if code in FORM_CATEGORIES:
                         mapped_names.append(f"{code} - {FORM_CATEGORIES[code]}")
                     else:
                         mapped_names.append(code)
                 categories_str = ", ".join(mapped_names)
            st.markdown(f"**Categories:** {categories_str}")
            pages_str = ", ".join(sorted([p for p in info['pages'] if p != "N/A"]))
            if pages_str:
                st.markdown(f"**Pages:** {pages_str}")
            # Sections might be too long, so just show count or a sample
            # st.markdown(f"**Sections:** {', '.join(list(info['sections'])[:3])}...") # Example with sample

def render_feedback_stats(app_state: AppState) -> None:
    """Render feedback statistics in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("📊 Feedback Stats")
    analysis = app_state.feedback_system.analyze_feedback()

    # Check if there's an error in the analysis result
    if "error" in analysis:
        st.sidebar.info("No feedback data yet. Start using the system!")
        return

    # Use the correct keys from the analysis dictionary
    total_feedback = analysis.get('total_feedback', 0)
    helpful = analysis.get('satisfaction_rate', 0)
    incorrect = analysis.get('incorrect_rate', 0)

    st.sidebar.write(f"Total Feedback: {total_feedback}")
    st.sidebar.write(f"Helpful: {helpful:.1f}%")
    st.sidebar.write(f"Not Helpful: {incorrect:.1f}%")

    if st.sidebar.button("🔍 Show Detailed Analysis"):
        with st.sidebar.expander("Detailed Analysis"):
            cal = analysis["confidence_calibration"]
            st.write("#### Confidence calibration")
            st.write(f"High confidence correct: {cal['high_conf_accurate']}")
            st.write(f"High confidence wrong: {cal['high_conf_wrong']}")
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
                    st.write(f"{idx}. \"{item['query']}\"")
                    st.write(f" Confidence: {item['confidence_pct']}% ({item['confidence_level']})")
                    if item.get("correction"):
                        st.write(f" User feedback: {item['correction'][:100]}...")

def render_query_settings() -> tuple:
    """Render query settings in the sidebar and return their values."""
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Query Settings")

    retrieval_type = st.sidebar.selectbox("Retrieval Method", ["hybrid", "vector", "bm25"], index=0)
    auto_refine_option = st.sidebar.toggle("Auto-refine Low Confidence", value=True)
    fortify_option = st.sidebar.toggle("Query Fortification", value=False)

    # Use the global cohere_client to check if the API key is available
    cohere_available = cohere_client is not None
    if not cohere_available:
        st.sidebar.info("Cohere API key not found. Reranking disabled.")
        rerank_option = False
        rerank_checkbox_disabled = True
    else:
        rerank_option = st.sidebar.toggle("Cohere Reranking", value=False, disabled=not cohere_available)
        rerank_checkbox_disabled = not cohere_available

    return retrieval_type, auto_refine_option, fortify_option, rerank_option, rerank_checkbox_disabled

def render_chat_feedback_row(app_state: AppState, result: Dict, key_prefix: str) -> None:
    """Render inline feedback buttons for a specific result."""
    # Use the result's unique identifier or a combination of query and timestamp for the key prefix
    # This ensures feedback is linked to the specific message instance
    feedback_key = f"fb_{key_prefix}"
    correction_toggle_key = f"corr_{key_prefix}"

    st.markdown("<div style='margin-top: 10px; padding: 8px; border-radius: 8px; background-color: rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("👍 Helpful", key=f"like_{feedback_key}"):
            app_state.feedback_system.log_feedback(result, "helpful")
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 Not Helpful", key=f"dislike_{feedback_key}"):
            app_state.feedback_system.log_feedback(result, "not_helpful")
            st.info("Thanks for letting us know.")
    with col3:
        if st.button("❌ Incorrect", key=f"incorrect_{feedback_key}"):
            st.session_state[correction_toggle_key] = not st.session_state.get(correction_toggle_key, False)

    # Correction input box, shown only if the 'Incorrect' toggle is True
    if st.session_state.get(correction_toggle_key, False):
        correction = st.text_area(
            "What should the answer be?",
            key=f"correction_text_{key_prefix}",
            placeholder="Tell us what needs to be fixed...",
            label_visibility="collapsed",
        )
        if st.button("Submit correction", key=f"submit_correction_{key_prefix}"):
            if correction.strip():
                app_state.feedback_system.log_feedback(result, "incorrect", correction.strip())
                st.success("Thank you! Your correction has been recorded.")
                st.session_state[correction_toggle_key] = False
                st.session_state.pop(f"correction_text_{key_prefix}", None) # Clear the text input state
            else:
                st.warning("Please provide correction details before submitting.")

    st.markdown("</div>", unsafe_allow_html=True)

def render_app(app_state: AppState, *, allow_library_controls: bool = True, rerank_checkbox_disabled: bool = False) -> None:
    """Main rendering function for the Streamlit app."""
    st.set_page_config(page_title="Maritime RAG Assistant", layout="wide")
    apply_custom_theme()
    st.title("Maritime RAG Assistant")
    st.caption("Modular Streamlit interface powered by Gemini + LlamaIndex")

    # Sidebar setup
    if allow_library_controls:
        render_library_controls(app_state)
    render_document_list(app_state)
    render_feedback_stats(app_state)
    retrieval_type, auto_refine_option, fortify_option, rerank_option, rerank_checkbox_disabled = render_query_settings()

    # Main container setup
    main_container = st.container()

    # Initialize session state for chat messages if not already done
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Initialize a flag to prevent re-querying on scroll/js update
    if "last_query_processed" not in st.session_state:
        st.session_state.last_query_processed = False

    # === MAIN CHAT INTERFACE ===
    with main_container:
        # Create a container specifically for the chat history (the SCROLLABLE BOX)
        chat_history_container = st.container()

        # Render all existing chat messages using st.chat_message
        with chat_history_container:
            # Optional: Add a header or separator if desired
            # st.subheader("Chat History")
            for idx, message in enumerate(st.session_state.chat_messages):
                # Create a unique key for feedback components linked to this message
                msg_key = f"{idx}_{message['role'][:1]}_{hash(message['content']) % 10000}" # Use index, role initial, and hash for uniqueness

                if message["role"] == "user":
                    # User message
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:  # role == "assistant"
                    # Assistant message
                    with st.chat_message("assistant"):
                        # Render the answer content using the existing compose_result_markdown function
                        # The 'content' in the message dictionary now holds the full result dictionary
                        result_dict = message["content"]
                        # Display the formatted markdown answer
                        st.markdown(compose_result_markdown(result_dict))

                        # Add feedback controls below the answer for this specific message
                        render_chat_feedback_row(app_state, result_dict, msg_key)

        # Add a spacer to ensure the input box is at the bottom
        st.markdown("---")

        # === INPUT BOX ===
        # Use st.chat_input for the bottom input bar
        user_prompt = st.chat_input("Ask about the maritime library...")

        if user_prompt and not st.session_state.last_query_processed:
            # Add user message to session state
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            st.session_state.last_query_processed = True # Set flag to prevent re-processing on reruns caused by JS or other widgets

            # Show a spinner while processing the assistant's response
            # Note: The spinner will appear *after* the user message in the container
            # because the container re-renders. This is the standard Streamlit behavior for chat_input.
            with st.spinner("Searching and thinking..."):
                try:
                    # Perform the query (your existing logic)
                    result = query_with_confidence(
                        app_state,
                        user_prompt.strip(),
                        retriever_type=retrieval_type,
                        auto_refine=auto_refine_option,
                        fortify=fortify_option,
                        rerank=rerank_option,
                    )

                    # Add assistant message to session state
                    # Store the full result dictionary as the 'content' for the assistant message
                    st.session_state.chat_messages.append({"role": "assistant", "content": result})

                    # Save the result to app_state.history_log for persistence
                    app_state.append_history(result)

                    # Update last_result if needed
                    app_state.last_result = result

                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    # Optionally, add an error message as an assistant message
                    error_result = {
                        "query": user_prompt,
                        "answer": f"❌ An error occurred during processing: {str(exc)}",
                        "confidence_pct": 0,
                        "confidence_level": "Error",
                        "sources": [],
                        "num_sources": 0,
                        "retriever_type": retrieval_type,
                        "attempts": 1,
                        "refinement_history": []
                    }
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_result})

            # After processing, reset the flag so the next input can be handled
            st.session_state.last_query_processed = False

            # --- IMPORTANT: Auto-scroll to bottom using JavaScript ---
            # Inject JavaScript to scroll to the bottom of the page after a new message pair (user + assistant) is added.
            st.components.v1.html("""
            <script>
                // Wait a bit for the DOM to fully update after spinner finishes
                setTimeout(function() {
                    // Scroll to the bottom of the page/document body
                    window.scrollTo(0, document.body.scrollHeight);
                }, 250); // Delay in milliseconds, adjust if needed
            </script>
            """, height=0) # height=0 hides the HTML component itself


def render_viewer_app(app_state: AppState) -> None:
    """Restricted UI for read-only testing of retrieval."""
    # Apply theme and title
    st.set_page_config(page_title="Maritime RAG Viewer", layout="wide")
    apply_custom_theme()
    st.title("Maritime RAG Viewer")
    st.caption("Read-only mode for query testing.")

    # Load cache if not already loaded
    if not app_state.nodes or not app_state.index:
        load_or_warn(app_state) # Assumes load_or_warn is accessible or defined similarly

    # Call the main render_app function but restrict controls
    render_app(app_state, allow_library_controls=False, rerank_checkbox_disabled=False)

# --- CSS Definitions ---
# (Keep your existing CSS definitions here)
_DARK_THEME_CSS = """
<style>
    /* Enforce dark mode */
    :root { color-scheme: dark; }

    /* Main body styling */
    body {
        font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;
        margin: 0;
        padding: 0;
        background: #040b14; /* Deep blue background */
        color: #f2f7ff; /* Light text */
    }

    /* Main container styling */
    .container {
        max-width: 900px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: linear-gradient(155deg, rgba(5, 20, 35, 0.96), rgba(2, 12, 22, 0.9)); /* Slightly lighter blue gradient */
        border: 1px solid rgba(120, 210, 255, 0.12); /* Subtle blue border */
        border-radius: 24px;
        box-shadow: 0 32px 70px rgba(0, 0, 0, 0.45); /* Deep shadow for depth */
    }

    /* Markdown content styling */
    .markdown-body {
        line-height: 1.75;
        font-size: 18px;
        color: #f2f7ff; /* Ensure text color inside markdown is light */
    }
    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
        margin-top: 1.5em;
        color: #a0d2ff; /* Light blue headers */
    }
    .markdown-body h3 {
        border-bottom: 1px solid rgba(127,127,127,.25);
        padding-bottom: .3em;
    }
    .markdown-body pre {
        background: rgba(10, 30, 50, 0.7); /* Darker code block background */
        padding: 1rem;
        overflow: auto;
        border-radius: 8px;
        border: 1px solid rgba(120, 210, 255, 0.1); /* Subtle border for code blocks */
    }
    .markdown-body code {
        padding: .1em .3em;
        background: rgba(10, 30, 50, 0.5); /* Slightly transparent code inline background */
        border-radius: 4px;
    }
    .markdown-body blockquote {
        margin: 1em 0;
        padding: .5em 1em;
        border-left: 4px solid rgba(120, 210, 255, 0.4); /* Blue highlight for quotes */
        background: rgba(120, 210, 255, 0.08); /* Subtle blue background for quotes */
        border-radius: 4px;
    }
    .markdown-body table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    .markdown-body th, .markdown-body td {
        border: 1px solid rgba(127,127,127,.3);
        padding: .5em .75em;
    }
    /* Scrollbar styling for a more modern look */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(10, 30, 50, 0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(120, 210, 255, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(120, 210, 255, 0.5);
    }
    /* Streamlit specific overrides for chat messages if needed */
    [data-testid="stChatMessage"] {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 12px;
    }
    [data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] {
        background-color: rgba(5, 20, 35, 0.85); /* Darker background for user messages */
    }
    [data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] {
        background-color: rgba(2, 12, 22, 0.9); /* Slightly different for assistant */
    }
    /* Ensure input box area is distinct */
    section[data-testid="stForm"] {
        background-color: rgba(5, 20, 35, 0.7);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(120, 210, 255, 0.1);
    }
    /* Style for the chat input container */
    [data-testid="stChatInputContainer"] {
        border-top: 1px solid rgba(120, 210, 255, 0.1);
        padding-top: 1rem;
    }
</style>
"""

_GITHUBISH_CSS = """:root { color-scheme: dark; }body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif; margin: 0; padding: 0; background: #040b14; color: #f2f7ff; }.container { max-width: 900px; margin: 2rem auto; padding: 2.5rem; background: linear-gradient(155deg, rgba(5, 20, 35, 0.96), rgba(2, 12, 22, 0.9)); border: 1px solid rgba(120, 210, 255, 0.12); border-radius: 24px; box-shadow: 0 32px 70px rgba(0, 0, 0, 0.45); }.markdown-body { line-height: 1.75; font-size: 18px; color: #f2f7ff; }.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 { margin-top: 1.5em; color: #a0d2ff; }.markdown-body h3 { border-bottom: 1px solid rgba(127,127,127,.25); padding-bottom: .3em; }.markdown-body pre { background: rgba(10, 30, 50, 0.7); padding: 1rem; overflow: auto; border-radius: 8px; border: 1px solid rgba(120, 210, 255, 0.1); }.markdown-body code { padding: .1em .3em; background: rgba(10, 30, 50, 0.5); border-radius: 4px; }.markdown-body blockquote { margin: 1em 0; padding: .5em 1em; border-left: 4px solid rgba(120, 210, 255, 0.4); background: rgba(120, 210, 255, 0.08); border-radius: 4px; }.markdown-body table { border-collapse: collapse; width: 100%; margin: 1em 0; }.markdown-body th, .markdown-body td { border: 1px solid rgba(127,127,127,.3); padding: .5em .75em; }hr { border: 0; border-top: 1px solid rgba(127,127,127,.25); margin: 2rem 0; }.smallmeta { color: rgba(127,127,127,.9); font-size: 12px; margin-top: 2rem; }"""