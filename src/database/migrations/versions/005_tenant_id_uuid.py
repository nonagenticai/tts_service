"""Make ``tenant_id`` a real postgres ``uuid`` on every tenant-scoped table.

Revision ID: 005_tenant_id_uuid
Revises: 004_usage_alert_uuid_id

WHY
---
``w2v_common`` (>= 0.67) parses every inbound tenant as a ``uuid.UUID`` and
RAISES on anything else — the product-owner ruling that a ``tenant_id`` may
never hold a sentinel or a legacy slug. The storage layer had not caught up:
all four tenant-scoped tables kept ``character varying(255)``, which will
happily persist ``n8n-dev``, a truncated value, or a ``Bearer …`` header that
leaked into the wrong field, and only fails much later (or never). Postgres
``uuid`` makes the invariant structural.

The sibling ``tts_gateway`` database already stores ``tts_jobs.tenant_id`` as a
real ``uuid``, so this is the established target shape in this lane.

TWO DELIBERATE SHAPE CHOICES
----------------------------
1. **Explicit literal table names** (``TABLES`` below), never a walk of
   ``Base.metadata``. A metadata-driven migration re-reads the ORM at
   *upgrade* time, so once the models declare ``uuid`` the migration's own
   generated predicates flip type underneath it and ``alembic upgrade head``
   from ZERO dies with ``operator does not exist: character varying = uuid``.
   That is invisible in a long-lived production database (which never replays
   revision 001) and fatal in CI (which always does). A migration must
   describe the schema as it was ON THIS REVISION and nothing else.

2. **A census that RAISES before the cast.** ``USING tenant_id::uuid`` would
   abort on a non-uuid anyway, but with postgres's own error, which names the
   offending literal without saying which table, how many rows, or how many
   distinct bad values there are. Worse, the cast SILENTLY MERGES two tenants
   that differ only in letter case (``AAAA…`` and ``aaaa…`` fold to the same
   ``uuid``) — data loss that no error would ever report. The census reports
   both before a single byte is rewritten.
"""

import logging

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "005_tenant_id_uuid"
down_revision = "004_usage_alert_uuid_id"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

# ── The tables this revision rewrites, spelled out ───────────────────────────
# ⛔ Do NOT replace this with `Base.metadata` / `tenant_table_names(metadata)`.
# See "TWO DELIBERATE SHAPE CHOICES" in the module docstring: a metadata walk
# reads the CURRENT models, which already declare uuid, and breaks a from-zero
# upgrade. These names are literals on purpose.
TABLES: list[tuple[str, str]] = [
    ("tts_jobs", "tenant_id"),
    ("tts_usage", "tenant_id"),
    ("tenant_quotas", "tenant_id"),
    ("usage_alerts", "tenant_id"),
]

# The varchar width to restore on downgrade — the shape at revision 004.
LEGACY_TYPE = sa.String(255)

# Postgres POSIX regex (used with `!~`) for the canonical dashed uuid text form.
# Deliberately anchored and length-exact: `uuid.UUID()` in Python also accepts
# braced / urn: / undashed spellings, but a `varchar` column that holds one of
# those is still a value no reader can compare against a canonical uuid, so it
# must surface in the census rather than be silently normalised by the cast.
UUID_TEXT_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"


class TenantCensusError(RuntimeError):
    """A tenant column cannot be safely cast to ``uuid``. Nothing was altered."""


def row_count(conn: sa.Connection, table: str) -> int:
    """Total rows in ``table`` — logged either side of the cast."""
    # S608: `table` is never caller-supplied — it comes from the hardcoded
    # `TABLES` literal above, and postgres has no bind-parameter form for an
    # identifier. Same for the two queries below.
    sql = sa.text(f'SELECT count(*) FROM "{table}"')  # noqa: S608  # nosec B608
    return int(conn.execute(sql).scalar_one())


def non_uuid_values(conn: sa.Connection, table: str, column: str) -> list[tuple[str, int]]:
    """Distinct values in ``table.column`` that are not canonical uuid text.

    Returns ``[(value, occurrences), …]`` so the raised message can NAME the
    offending value and say how many rows carry it — the operator needs both to
    decide between a backfill and a delete.
    """
    rows = conn.execute(
        sa.text(
            f'SELECT "{column}"::text AS value, count(*) AS n '  # noqa: S608  # nosec B608
            f'FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND "{column}"::text !~ :pattern '
            f'GROUP BY "{column}"::text '
            "ORDER BY n DESC, value ASC"
        ),
        {"pattern": UUID_TEXT_RE},
    ).all()
    return [(str(r.value), int(r.n)) for r in rows]


def case_only_collisions(conn: sa.Connection, table: str, column: str) -> list[tuple[str, int]]:
    """Values that differ ONLY in letter case and would MERGE under ``::uuid``.

    Returns ``[(folded_value, distinct_spellings), …]``.

    ⛔ The predicate is ``count(DISTINCT "col"::text) > 1`` per folded group.
    It is NOT ``count(*) <> count(DISTINCT lower("col"))`` — that form counts
    ROWS, so every healthy table with many rows per tenant (which is every
    table here: a tenant has thousands of jobs) trips it and the migration
    refuses a database that is perfectly fine. The failure mode being guarded
    is two DIFFERENT SPELLINGS of one tenant, not repetition of one spelling.
    """
    rows = conn.execute(
        sa.text(
            f'SELECT lower("{column}"::text) AS folded, '  # noqa: S608  # nosec B608
            f'count(DISTINCT "{column}"::text) AS n_distinct '
            f'FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL '
            f'GROUP BY lower("{column}"::text) '
            f'HAVING count(DISTINCT "{column}"::text) > 1 '
            "ORDER BY folded ASC"
        )
    ).all()
    return [(str(r.folded), int(r.n_distinct)) for r in rows]


def census(conn: sa.Connection, table: str, column: str) -> int:
    """Refuse to cast ``table.column`` unless every value survives it losslessly.

    Raises :class:`TenantCensusError` on a non-uuid value or a case-only
    collision. Returns the table's row count so the caller can log it.
    """
    total = row_count(conn, table)

    bad = non_uuid_values(conn, table, column)
    if bad:
        detail = ", ".join(f"{value!r} x{n}" for value, n in bad[:20])
        more = "" if len(bad) <= 20 else f" (+{len(bad) - 20} more distinct values)"
        raise TenantCensusError(
            f"{table}.{column}: {len(bad)} distinct non-uuid value(s) across "
            f"{sum(n for _, n in bad)} of {total} row(s) — {detail}{more}. "
            "Backfill or delete these rows before re-running the migration; "
            "nothing has been altered."
        )

    collisions = case_only_collisions(conn, table, column)
    if collisions:
        detail = ", ".join(f"{folded!r} has {n} spellings" for folded, n in collisions[:20])
        more = "" if len(collisions) <= 20 else f" (+{len(collisions) - 20} more)"
        raise TenantCensusError(
            f"{table}.{column}: {len(collisions)} tenant value(s) differ only in letter case "
            f"and would MERGE into one row-set under ``::uuid`` — {detail}{more}. "
            "This is silent cross-tenant data loss, not a formatting nit. "
            "Canonicalise to lowercase deliberately (deciding which spelling owns the rows) "
            "before re-running; nothing has been altered."
        )

    return total


def upgrade() -> None:
    conn = op.get_bind()
    for table, column in TABLES:
        before = census(conn, table, column)
        logger.info("%s.%s: %d row(s) before cast to uuid", table, column, before)

        # ALTER COLUMN TYPE rebuilds every dependent index and unique
        # constraint automatically, and leaves NOT NULL / DEFAULT untouched —
        # so `tenant_quotas.tenant_id`'s UNIQUE and each table's btree index
        # survive without an explicit drop/create dance.
        op.alter_column(
            table,
            column,
            existing_type=LEGACY_TYPE,
            type_=PG_UUID(as_uuid=False),
            existing_nullable=False,
            postgresql_using=f'"{column}"::uuid',
        )

        after = row_count(conn, table)
        logger.info("%s.%s: %d row(s) after cast to uuid", table, column, after)
        if after != before:
            raise TenantCensusError(
                f"{table}.{column}: row count changed across the cast "
                f"({before} -> {after}). Aborting the transaction."
            )


def downgrade() -> None:
    conn = op.get_bind()
    for table, column in TABLES:
        before = row_count(conn, table)
        logger.info("%s.%s: %d row(s) before revert to varchar(255)", table, column, before)

        op.alter_column(
            table,
            column,
            existing_type=PG_UUID(as_uuid=False),
            type_=LEGACY_TYPE,
            existing_nullable=False,
            postgresql_using=f'"{column}"::text',
        )

        after = row_count(conn, table)
        logger.info("%s.%s: %d row(s) after revert to varchar(255)", table, column, after)
