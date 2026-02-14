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
    """Main entrypoint - authenticates user, then routes to chat or admin."""
    
    # Page config (must be first Streamlit command)
    st.set_page_config(
        page_title="MA.D.ASS",
        page_icon="⚓",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # === AUTHENTICATION GATE ===
    from app.auth import get_authenticator, get_tenant_for_user, is_superuser
    
    authenticator = get_authenticator()

    # Initialize auth state keys (required by streamlit-authenticator 0.4.x)
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "name" not in st.session_state:
        st.session_state["name"] = None
    if "logout" not in st.session_state:
        st.session_state["logout"] = None
    
    # Render login form and check status
    authenticator.login(location="main")

    name = st.session_state.get("name")
    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")
    
    if auth_status is False:
        st.error("❌ Invalid username or password")
        st.stop()
    
    if auth_status is None:
        # Not yet attempted - show welcome message with login form
        st.markdown("## ⚓ Maritime Document Assistant")
        st.markdown("Please log in to continue.")
        st.stop()
    
    # === AUTHENTICATION SUCCESSFUL ===
    # Store auth context in session state for use throughout the app
    try:
        st.session_state["tenant_id"] = get_tenant_for_user(username)
        st.session_state["is_superuser"] = is_superuser(username)
        st.session_state["username"] = username
        st.session_state["display_name"] = name
    except (KeyError, ValueError) as e:
        st.error(f"❌ Authentication error: {e}")
        LOGGER.error("Failed to get tenant for user %s: %s", username, e)
        authenticator.logout("Logout")
        st.stop()
    
    # === LOGOUT BUTTON IN SIDEBAR ===
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## {name}")
            if st.session_state.get("is_superuser"):
                st.caption("🔑 Administrator")
        with col2:
            authenticator.logout("↩️ Logout", "sidebar")
        
        st.write(" ")
    
    # === LOAD CONFIG ===
    AppConfig.get()
    
    # === VERSION CHECK: Recreate AppState if structure changed ===
    APP_STATE_VERSION = "2.1"  # Bump this when AppState structure changes
    
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        # Clear old state
        keys_to_keep = {"tenant_id", "is_superuser", "username", "display_name", 
                        "authenticator", "auth_config", "app_state_version"}
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        LOGGER.info("Cleared session state due to version change")
    
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState(tenant_id=st.session_state.get("tenant_id", "shared"))
        st.session_state["app_state_version"] = APP_STATE_VERSION
        LOGGER.info("Created new AppState (version %s) for tenant %s", 
                    APP_STATE_VERSION, st.session_state.get("tenant_id"))
    
    app_state: AppState = st.session_state["app_state"]
    
    # === LOAD CACHED INDEX ===
    current_tenant = st.session_state.get("tenant_id", "shared")
    cached_tenant = st.session_state.get("_loaded_tenant")
    
    # Reload if: no nodes, OR tenant changed, OR retrievers not ready
    needs_reload = (
        not app_state.nodes 
        or cached_tenant != current_tenant
        or not app_state.is_ready_for_queries()
    )
    
    if needs_reload:
        # Clear old state if tenant changed
        if cached_tenant and cached_tenant != current_tenant:
            app_state.invalidate_node_map_cache()
            app_state._index_load_attempted = False
        
        with st.spinner("Loading library..."):
            if app_state.ensure_index_loaded():
                app_state.ensure_nodes_loaded()
                st.session_state["_loaded_tenant"] = current_tenant
                LOGGER.info("Loaded cached index: %d nodes in SQLite (tenant=%s)", )
            else:
                LOGGER.info("No cached index found")
    
    # Ensure retrievers exist even if index was already loaded
    if app_state.index is not None and not app_state.is_ready_for_queries():
        app_state.ensure_retrievers()
    
    # === DATA CONSISTENCY CHECK ===
    # With multi-tenancy, memory != DB is expected (memory has tenant+shared, DB has all)
    # Only log, don't warn unless memory > DB (impossible = real bug)
    if app_state.nodes:
            try:
                manager = app_state.ensure_manager()
                memory_count = len(app_state.nodes)
                info = manager.qdrant_client.get_collection(manager.collection_name)
                db_count = info.points_count
                
                if memory_count > db_count:
                    LOGGER.warning(
                        "Data inconsistency: Memory=%d > Qdrant=%d (shouldn't happen)",
                        memory_count, db_count
                    )
            except Exception as exc:
                LOGGER.debug("Consistency check skipped: %s", exc)
    
    # === ROUTE TO APPROPRIATE UI ===
    read_only = st.query_params.get("read_only", "true").lower() == "true"
    
    if read_only:
        render_app(app_state, read_only_mode=True)
    else:
        render_admin_panel(app_state)

if __name__ == "__main__":
    main()