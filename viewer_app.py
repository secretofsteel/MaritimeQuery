"""Read-only Streamlit entrypoint for retrieval stress testing."""

from __future__ import annotations

import streamlit as st

from app.config import AppConfig
from app.state import AppState
from app.ui import render_viewer_app


def main() -> None:
    AppConfig.get()
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    render_viewer_app(st.session_state["app_state"])


if __name__ == "__main__":
    main()
