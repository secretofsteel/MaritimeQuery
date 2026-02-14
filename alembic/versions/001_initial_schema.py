"""initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-02-14 20:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A. Schema version tracking
    op.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO schema_info (key, value) VALUES ('version', '1')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
    """)

    # B. Node storage with tsvector (Layer 1 — RAG)
    op.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata JSONB,
            section_id TEXT,
            tenant_id TEXT NOT NULL DEFAULT 'shared',
            tsv tsvector,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_doc ON nodes(doc_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_tenant ON nodes(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_section ON nodes(section_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_tsv ON nodes USING GIN(tsv);
    """)

    # tsvector trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION nodes_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('simple', COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trg_nodes_tsv
            BEFORE INSERT OR UPDATE OF text ON nodes
            FOR EACH ROW
            EXECUTE FUNCTION nodes_tsv_trigger();
    """)

    # C. Chat sessions
    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            message_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    """)

    # D. Feedback
    op.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT,
            confidence_pct INTEGER,
            confidence_level TEXT,
            num_sources INTEGER,
            top_sources JSONB,
            retriever_type TEXT,
            feedback TEXT NOT NULL,
            correction TEXT,
            attempts INTEGER DEFAULT 1,
            was_refined BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC);
    """)

    # E. Users (placeholder)
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    """)

    # F. Layer 2 foundation tables
    op.execute("""
        CREATE TABLE IF NOT EXISTS vessels (
            id SERIAL PRIMARY KEY,
            company_id TEXT NOT NULL,
            name TEXT NOT NULL,
            imo_number TEXT UNIQUE,
            flag_state TEXT,
            class_society TEXT,
            vessel_type TEXT,
            dwt NUMERIC,
            bwm_system_type TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS certificates (
            id SERIAL PRIMARY KEY,
            vessel_id INTEGER REFERENCES vessels(id) ON DELETE CASCADE,
            certificate_type TEXT NOT NULL,
            certificate_name TEXT NOT NULL,
            issue_date DATE,
            expiry_date DATE,
            issuing_authority TEXT,
            status TEXT DEFAULT 'active',
            source_document_id TEXT,
            extracted_fields JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS document_inventory (
            id SERIAL PRIMARY KEY,
            vessel_id INTEGER REFERENCES vessels(id) ON DELETE SET NULL,
            company_id TEXT NOT NULL,
            document_type TEXT,
            title TEXT NOT NULL,
            revision_number TEXT,
            approval_date DATE,
            next_review_date DATE,
            ingestion_status TEXT DEFAULT 'pending'
                CHECK (ingestion_status IN ('pending', 'processed', 'archived')),
            layer_assignment TEXT
                CHECK (layer_assignment IN ('L1_full', 'L1_summary', 'L3_archive')),
            file_path TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS inspections (
            id SERIAL PRIMARY KEY,
            vessel_id INTEGER REFERENCES vessels(id) ON DELETE CASCADE,
            inspection_type TEXT NOT NULL
                CHECK (inspection_type IN ('PSC', 'flag', 'vetting', 'internal')),
            date DATE NOT NULL,
            port TEXT,
            inspector TEXT,
            overall_result TEXT,
            source_document_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS deficiencies (
            id SERIAL PRIMARY KEY,
            inspection_id INTEGER REFERENCES inspections(id) ON DELETE CASCADE,
            vessel_id INTEGER REFERENCES vessels(id) ON DELETE CASCADE,
            code TEXT,
            description TEXT NOT NULL,
            severity TEXT,
            status TEXT DEFAULT 'open'
                CHECK (status IN ('open', 'in_progress', 'closed', 'verified')),
            due_date DATE,
            responsible_person TEXT,
            closure_date DATE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_vessels_company ON vessels(company_id);
        CREATE INDEX IF NOT EXISTS idx_certificates_vessel ON certificates(vessel_id);
        CREATE INDEX IF NOT EXISTS idx_certificates_expiry ON certificates(expiry_date);
        CREATE INDEX IF NOT EXISTS idx_document_inventory_company ON document_inventory(company_id);
        CREATE INDEX IF NOT EXISTS idx_inspections_vessel ON inspections(vessel_id);
        CREATE INDEX IF NOT EXISTS idx_deficiencies_inspection ON deficiencies(inspection_id);
        CREATE INDEX IF NOT EXISTS idx_deficiencies_vessel ON deficiencies(vessel_id);
        CREATE INDEX IF NOT EXISTS idx_deficiencies_status ON deficiencies(status);
    """)


def downgrade() -> None:
    # Drop in reverse dependency order
    op.execute("DROP TABLE IF EXISTS deficiencies;")
    op.execute("DROP TABLE IF EXISTS inspections;")
    op.execute("DROP TABLE IF EXISTS document_inventory;")
    op.execute("DROP TABLE IF EXISTS certificates;")
    op.execute("DROP TABLE IF EXISTS vessels;")
    op.execute("DROP TABLE IF EXISTS users;")
    op.execute("DROP TABLE IF EXISTS feedback;")
    op.execute("DROP TABLE IF EXISTS messages;")
    op.execute("DROP TABLE IF EXISTS sessions;")
    op.execute("DROP TRIGGER IF EXISTS trg_nodes_tsv ON nodes;")
    op.execute("DROP FUNCTION IF EXISTS nodes_tsv_trigger();")
    op.execute("DROP TABLE IF EXISTS nodes;")
    op.execute("DROP TABLE IF EXISTS schema_info;")
