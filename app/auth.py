"""Authentication module using streamlit-authenticator.

Provides tenant-scoped authentication for multi-company access.
Each user belongs to a tenant, and all their data is scoped accordingly.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import streamlit_authenticator as stauth
import yaml

from .logger import LOGGER


def get_auth_config_path() -> Path:
    """Get path to authentication config file."""
    from .config import AppConfig
    config = AppConfig.get()
    return config.paths.base_dir / "config" / "users.yaml"


def create_default_config(config_path: Path) -> Dict[str, Any]:
    """
    Create a default authentication config file.
    
    Generates a secure cookie key and placeholder admin account.
    The admin password hash is a placeholder - must be replaced.
    
    Returns:
        The default configuration dict.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate secure cookie key
    cookie_key = secrets.token_hex(16)
    
    default_config = {
        "credentials": {
            "usernames": {
                "admin": {
                    "name": "Administrator",
                    "password": "$2b$12$PLACEHOLDER_REPLACE_ME_WITH_REAL_HASH",
                    "tenant_id": "admin",
                    "role": "superuser",
                }
            }
        },
        "cookie": {
            "name": "maritime_rag_auth",
            "key": cookie_key,
            "expiry_days": 30,
        },
        "preauthorized": {
            "emails": []
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    LOGGER.warning(
        "Created default auth config at %s - UPDATE THE ADMIN PASSWORD!", 
        config_path
    )
    
    return default_config


def load_auth_config() -> Dict[str, Any]:
    """
    Load authentication configuration from YAML file.
    
    Creates default config if file doesn't exist.
    
    Returns:
        Configuration dictionary for streamlit-authenticator.
    
    Raises:
        ValueError: If config file exists but is invalid.
    """
    config_path = get_auth_config_path()
    
    if not config_path.exists():
        LOGGER.info("Auth config not found, creating default: %s", config_path)
        return create_default_config(config_path)
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        if not config.get("credentials", {}).get("usernames"):
            raise ValueError("Config missing credentials.usernames section")
        if not config.get("cookie"):
            raise ValueError("Config missing cookie section")
        
        LOGGER.debug("Loaded auth config: %d users", 
                     len(config["credentials"]["usernames"]))
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in auth config: {e}")


def get_authenticator() -> stauth.Authenticate:
    """
    Get configured Streamlit authenticator instance.
    
    Caches the authenticator in session state to persist across reruns.
    
    Returns:
        Configured Authenticate instance ready for login().
    """
    # Cache in session state
    if "authenticator" not in st.session_state:
        config = load_auth_config()
        
        st.session_state["authenticator"] = stauth.Authenticate(
            credentials=config["credentials"],
            cookie_name=config["cookie"]["name"],
            cookie_key=config["cookie"]["key"],
            cookie_expiry_days=config["cookie"]["expiry_days"],
            preauthorized=config.get("preauthorized", {}).get("emails", []),
        )
        
        # Also cache the config for tenant lookup
        st.session_state["auth_config"] = config
    
    return st.session_state["authenticator"]


def get_tenant_for_user(username: str) -> str:
    """
    Look up tenant_id for a given username.
    
    Args:
        username: The authenticated username.
    
    Returns:
        The tenant_id associated with the user.
    
    Raises:
        KeyError: If username not found in config.
        ValueError: If user has no tenant_id configured.
    """
    config = st.session_state.get("auth_config")
    
    if config is None:
        config = load_auth_config()
    
    user_data = config["credentials"]["usernames"].get(username)
    
    if user_data is None:
        raise KeyError(f"User not found in config: {username}")
    
    tenant_id = user_data.get("tenant_id")
    
    if not tenant_id:
        raise ValueError(f"User {username} has no tenant_id configured")
    
    return tenant_id


def is_superuser(username: str) -> bool:
    """
    Check if user has superuser role.
    
    Args:
        username: The authenticated username.
    
    Returns:
        True if user has role 'superuser', False otherwise.
    """
    config = st.session_state.get("auth_config")
    
    if config is None:
        config = load_auth_config()
    
    user_data = config["credentials"]["usernames"].get(username, {})
    return user_data.get("role") == "superuser"


def hash_password(password: str) -> str:
    """
    Generate bcrypt hash for a password.
    
    Convenience function for creating new user entries.
    
    Args:
        password: Plain text password.
    
    Returns:
        Bcrypt hash string suitable for users.yaml.
    """
    return stauth.Hasher([password]).generate()[0]


def render_login() -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    """
    Render login form and return authentication status.
    
    This is a convenience wrapper around authenticator.login()
    that handles the common pattern.
    
    Returns:
        Tuple of (name, authentication_status, username)
        - name: Display name if authenticated, None otherwise
        - authentication_status: True if authenticated, False if failed, None if not attempted
        - username: Username if authenticated, None otherwise
    """
    authenticator = get_authenticator()
    return authenticator.login(location="main")


__all__ = [
    "get_authenticator",
    "get_tenant_for_user",
    "is_superuser",
    "load_auth_config",
    "hash_password",
    "render_login",
]
