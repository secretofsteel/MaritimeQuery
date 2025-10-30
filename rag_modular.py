"""Modular Streamlit entrypoint using refactored components."""

from __future__ import annotations

import streamlit as st

from app.config import AppConfig
from app.logger import LOGGER
from app.state import AppState
from app.ui import render_app


def main() -> None:
    AppConfig.get()  # Initialise configuration
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    render_app(st.session_state["app_state"])


if __name__ == "__main__":
    main()
