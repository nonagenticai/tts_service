"""Migrate UsageAlert.id from integer to UUID string.

Revision ID: 004_usage_alert_uuid_id
Revises: 003_datetime_timezone
"""

import sqlalchemy as sa
from alembic import op

revision = "004_usage_alert_uuid_id"
down_revision = "003_datetime_timezone"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old integer primary key and recreate as String(36) UUID
    op.drop_column("usage_alerts", "id")
    op.add_column(
        "usage_alerts",
        sa.Column("id", sa.String(36), nullable=False),
    )
    op.create_primary_key("pk_usage_alerts", "usage_alerts", ["id"])


def downgrade() -> None:
    op.drop_constraint("pk_usage_alerts", "usage_alerts", type_="primary")
    op.drop_column("usage_alerts", "id")
    op.add_column(
        "usage_alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    )
    op.create_primary_key("pk_usage_alerts", "usage_alerts", ["id"])
