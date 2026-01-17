"""Modular Streamlit entrypoint using refactored components."""

from __future__ import annotations

import streamlit as st

from app.config import AppConfig
from app.logger import LOGGER
from app.state import AppState
from app.ui import (
    render_app,  # Existing chat interface for USER mode
    render_admin_panel,  # New admin panel for ADMIN mode
)


def main() -> None:
    """Main entrypoint - routes to chat or admin based on query param."""
    
    # Page config (must be first Streamlit command)
    st.set_page_config(
        page_title="MA.D.ASS - Admin Panel",
        page_icon="⚓",
        layout="centered",  # Centered for both modes
        initial_sidebar_state="expanded"
    )
    
    # Load config
    AppConfig.get()
    
    # Version check: recreate AppState if structure changed
    APP_STATE_VERSION = "2.0"
    
    # Force clear old state if version mismatch
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        LOGGER.info("Cleared session state due to version change")
    
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
        st.session_state["app_state_version"] = APP_STATE_VERSION
        LOGGER.info("Created new AppState (version %s)", APP_STATE_VERSION)
    
    app_state: AppState = st.session_state["app_state"]
    
    # Load cached index if available
    if not app_state.nodes:
        with st.spinner("Loading cached index..."):
            if app_state.ensure_index_loaded():
                LOGGER.info("Loaded cached index: %d nodes", len(app_state.nodes))
            else:
                LOGGER.info("No cached index found")
    
    # Data consistency check: warn if memory and DB are out of sync
    if app_state.nodes:
        try:
            manager = app_state.ensure_manager()
            memory_count = len(app_state.nodes)
            db_count = manager.collection.count()
            diff = abs(memory_count - db_count)
            
            if diff > 10:
                LOGGER.warning(
                    "Data inconsistency detected: Memory=%d chunks, DB=%d chunks, diff=%d",
                    memory_count, db_count, diff
                )
                # Also log which is larger for debugging
                if memory_count > db_count:
                    LOGGER.warning("Memory has %d MORE chunks than DB (memory bloat)", diff)
                else:
                    LOGGER.warning("DB has %d MORE chunks than memory (orphaned chunks)", diff)
        except Exception as exc:
            # Don't fail startup if consistency check fails
            LOGGER.warning("Failed to check data consistency: %s", exc)
    
    # Check query parameter for mode
    read_only = st.query_params.get("read_only", "true").lower() == "true"
    
    # Route to appropriate UI
    if read_only:
        # USER MODE: Chat interface
        render_app(app_state, read_only_mode=True)
    else:
        # ADMIN MODE: Full admin panel, NO chat
        render_admin_panel(app_state)


if __name__ == "__main__":
    main()