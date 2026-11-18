"""Add enhanced features: caching, routing, language, cost tracking, quotas, alerts.

Revision ID: 002_enhanced_features
Revises: 001_initial
"""

import sqlalchemy as sa
from alembic import op

revision = "002_enhanced_features"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ========================================================================
    # TTSJob: new columns for caching, routing, language, cost
    # ========================================================================
    op.add_column(
        "tts_jobs", sa.Column("cached", sa.Boolean(), nullable=False, server_default="false")
    )
    op.add_column("tts_jobs", sa.Column("cache_key", sa.String(128), nullable=True))
    op.add_column("tts_jobs", sa.Column("routing_strategy", sa.String(20), nullable=True))
    op.add_column("tts_jobs", sa.Column("selected_provider", sa.String(50), nullable=True))
    op.add_column("tts_jobs", sa.Column("detected_language", sa.String(10), nullable=True))
    op.add_column("tts_jobs", sa.Column("estimated_cost_usd", sa.Float(), nullable=True))
    op.add_column("tts_jobs", sa.Column("latency_ms", sa.Float(), nullable=True))
    op.add_column("tts_jobs", sa.Column("concatenated_audio_path", sa.String(500), nullable=True))

    # ========================================================================
    # TTSUsage: new columns for cost and cache tracking
    # ========================================================================
    op.add_column(
        "tts_usage", sa.Column("estimated_cost_usd", sa.Float(), nullable=False, server_default="0")
    )
    op.add_column(
        "tts_usage", sa.Column("avg_latency_ms", sa.Float(), nullable=False, server_default="0")
    )
    op.add_column(
        "tts_usage", sa.Column("cache_hits", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "tts_usage", sa.Column("cache_misses", sa.Integer(), nullable=False, server_default="0")
    )

    # ========================================================================
    # TenantQuota: per-tenant quota/budget configuration
    # ========================================================================
    op.create_table(
        "tenant_quotas",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("monthly_budget_usd", sa.Float(), nullable=True),
        sa.Column("monthly_character_limit", sa.Integer(), nullable=True),
        sa.Column("monthly_request_limit", sa.Integer(), nullable=True),
        sa.Column("daily_request_limit", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # ========================================================================
    # UsageAlert: usage alert records
    # ========================================================================
    op.create_table(
        "usage_alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("threshold_pct", sa.Integer(), nullable=False),
        sa.Column("current_value", sa.Float(), nullable=False),
        sa.Column("limit_value", sa.Float(), nullable=False),
        sa.Column("period", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("usage_alerts")
    op.drop_table("tenant_quotas")

    op.drop_column("tts_usage", "cache_misses")
    op.drop_column("tts_usage", "cache_hits")
    op.drop_column("tts_usage", "avg_latency_ms")
    op.drop_column("tts_usage", "estimated_cost_usd")

    op.drop_column("tts_jobs", "concatenated_audio_path")
    op.drop_column("tts_jobs", "latency_ms")
    op.drop_column("tts_jobs", "estimated_cost_usd")
    op.drop_column("tts_jobs", "detected_language")
    op.drop_column("tts_jobs", "selected_provider")
    op.drop_column("tts_jobs", "routing_strategy")
    op.drop_column("tts_jobs", "cache_key")
    op.drop_column("tts_jobs", "cached")
