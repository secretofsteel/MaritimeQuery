"""PostgreSQL connection pool and utilities.

Provides a synchronous connection pool using psycopg3.
Async upgrade deferred to Step 5 (when Streamlit is removed).

Usage:
    from app.pg_database import get_pg_pool, pg_connection

    # In application startup (FastAPI lifespan):
    pool = get_pg_pool()

    # In any function that needs DB access:
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ...")
            rows = cur.fetchall()
"""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .logger import LOGGER

_PG_POOL: ConnectionPool | None = None


def init_pg_pool(database_url: str | None = None) -> ConnectionPool:
    """Initialize the module-level connection pool. Idempotent."""
    global _PG_POOL
    if _PG_POOL is not None:
        return _PG_POOL

    dsn = database_url or os.getenv(
        "DATABASE_URL",
        "postgresql://maritime:maritime_dev@localhost:5432/maritime_rag"
    )

    LOGGER.info("Initializing PostgreSQL connection pool...")
    try:
        _PG_POOL = ConnectionPool(
            conninfo=dsn,
            min_size=2,
            max_size=10,
            open=True,
            kwargs={"row_factory": dict_row}
        )
        LOGGER.info("PostgreSQL connection pool created.")
    except Exception as e:
        LOGGER.error("Failed to create PostgreSQL connection pool: %s", e)
        raise

    return _PG_POOL


def get_pg_pool() -> ConnectionPool:
    """Get the connection pool. Raises RuntimeError if not initialized."""
    if _PG_POOL is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call init_pg_pool() first.")
    return _PG_POOL


def close_pg_pool() -> None:
    """Close the connection pool. Safe to call multiple times."""
    global _PG_POOL
    if _PG_POOL is not None:
        LOGGER.info("Closing PostgreSQL connection pool...")
        _PG_POOL.close()
        _PG_POOL = None


@contextmanager
def pg_connection() -> Generator[psycopg.Connection, None, None]:
    """Context manager: checkout connection, auto-commit/rollback."""
    pool = get_pg_pool()
    with pool.connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def check_pg_connection() -> bool:
    """Quick health check. Returns True if PostgreSQL is reachable."""
    try:
        if _PG_POOL is None:
            return False
        with _PG_POOL.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False
