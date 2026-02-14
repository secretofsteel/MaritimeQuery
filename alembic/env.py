"""Alembic environment configuration.

Uses psycopg3 directly — no SQLAlchemy ORM dependency.
Reads DATABASE_URL from environment or falls back to dev default.
"""

import os
from logging.config import fileConfig

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None  # No SQLAlchemy models — raw SQL migrations

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://maritime:maritime_dev@localhost:5432/maritime_rag"
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL script)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    # Try connecting with psycopg3 directly
    try:
        from psycopg import connect
        connection = connect(DATABASE_URL, autocommit=True)
        
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()
            
        connection.close()
        
    except Exception as e:
        # Fallback for some Alembic versions/environments if DBAPI2 connection isn't accepted directly
        # or if psycopg isn't behaving as expected by Alembic
        print(f"Direct psycopg connection failed, trying SQLAlchemy wrapper: {e}")
        from sqlalchemy import create_engine
        
        # Ensure create_engine uses the correct driver prefix if needed (e.g. postgresql+psycopg://)
        # But standard postgresql:// usually defaults to psycopg2. 
        # For psycopg3, we might need postgresql+psycopg://
        url = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")
        
        engine = create_engine(url)
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
