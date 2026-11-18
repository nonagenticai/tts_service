"""Initial schema: tts_jobs and tts_usage base tables.

Revision ID: 001_initial
Revises:
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ========================================================================
    # tts_jobs: TTS generation job tracking (base columns; 002 adds caching,
    # routing, language detection, and cost tracking columns)
    # ========================================================================
    op.create_table(
        "tts_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), unique=True, nullable=False, index=True),
        sa.Column("batch_id", sa.String(36), nullable=True, index=True),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_length", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False, index=True),
        sa.Column("voice", sa.String(100), nullable=False),
        sa.Column("speed", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("output_format", sa.String(10), nullable=False, server_default="mp3"),
        sa.Column("model", sa.String(255), nullable=True),
        sa.Column("extra_params", JSONB(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending", index=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("audio_path", sa.String(500), nullable=True),
        sa.Column("audio_size_bytes", sa.Integer(), nullable=True),
        sa.Column("audio_duration_seconds", sa.Float(), nullable=True),
        sa.Column("caller_item_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_tts_jobs_status_created", "tts_jobs", ["status", "created_at"])

    # ========================================================================
    # tts_usage: Daily usage aggregation per tenant and provider (base columns;
    # 002 adds cost and cache tracking columns)
    # ========================================================================
    op.create_table(
        "tts_usage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("provider", sa.String(50), nullable=False, index=True),
        sa.Column("date", sa.Date(), nullable=False, index=True),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_characters", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_audio_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("tenant_id", "provider", "date", name="uq_usage_tenant_provider_date"),
    )


def downgrade() -> None:
    op.drop_table("tts_usage")
    op.drop_index("ix_tts_jobs_status_created", table_name="tts_jobs")
    op.drop_table("tts_jobs")
