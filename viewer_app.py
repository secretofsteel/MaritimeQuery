"""Read-only Streamlit entrypoint for retrieval stress testing."""

from __future__ import annotations

import streamlit as st

from app.config import AppConfig
from app.state import AppState
from app.ui import render_app, render_viewer_app


def main() -> None:
    AppConfig.get()
    
    # Version check: recreate AppState if structure changed
    APP_STATE_VERSION = "2.0"  # Increment when AppState structure changes
    
    # Force clear old state if version mismatch
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        # Clear ALL session state to ensure clean slate
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
        st.session_state["app_state_version"] = APP_STATE_VERSION

    # Check query parameter for read-only mode (defaults to True for viewer app)
    read_only = st.query_params.get("read_only", "true").lower() == "true"
    render_viewer_app(st.session_state["app_state"]) if read_only else render_app(st.session_state["app_state"], read_only_mode=False)


if __name__ == "__main__":
    main()
