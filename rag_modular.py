"""Modular Streamlit entrypoint using refactored components."""

from __future__ import annotations

import streamlit as st

from app.config import AppConfig
from app.logger import LOGGER
from app.state import AppState
from app.ui import render_app


def main() -> None:
    AppConfig.get()
    
    # Version check: recreate AppState if structure changed
    APP_STATE_VERSION = "2.0"  # Increment when AppState structure changes
    
    # Force clear old state if version mismatch
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        # Clear ALL session state to ensure clean slate
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        LOGGER.info("Cleared session state due to version change")
    
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
        st.session_state["app_state_version"] = APP_STATE_VERSION
        LOGGER.info("Created new AppState (version %s)", APP_STATE_VERSION)

    # Check query parameter for read-only mode
    read_only = st.query_params.get("read_only", "false").lower() == "true"
    render_app(st.session_state["app_state"], read_only_mode=read_only)


if __name__ == "__main__":
    main()
