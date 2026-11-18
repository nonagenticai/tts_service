"""The ``005_tenant_id_uuid`` census guard, exercised against real postgres.

⛔⛔ READ THIS BEFORE ANY BULK EDIT OF TENANT LITERALS IN THIS REPO ⛔⛔
The string literals in this module are DELIBERATELY DIRTY. ``"tenant-1"``,
``"n8n-dev"`` and the UPPERCASE uuid spellings are the INPUTS the census exists
to catch — they are the test data, not stale fixtures that "forgot" to move to
``TENANT_A``. A mechanical sweep that rewrites them to canonical uuids leaves
every assertion here passing while testing nothing at all, because a census
that has no dirty row to find can never go red. Every dirty literal below is
marked ``# DIRTY-ON-PURPOSE``. If a sweep touched one, revert it.

Why a real database and not a mock: the guard IS its SQL. ``!~``, ``lower()``
and ``count(DISTINCT …)`` are postgres semantics, and a fake that returns
canned rows would keep passing under exactly the rewrite this module exists to
prevent (``count(*)`` in place of ``count(DISTINCT …)``).
"""

from __future__ import annotations

import ast
import importlib.util
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

from tests.conftest import _TEST_DATABASE_URL

# ---------------------------------------------------------------------------
# Load the migration module by PATH.
#
# It is not importable as `src.database.migrations.versions.005_tenant_id_uuid`
# (the leading digits make that an invalid identifier), and alembic itself
# loads revisions this way. Importing the real module — rather than copying its
# SQL into the test — is what makes the mutation test meaningful: editing the
# migration is what turns these red.
# ---------------------------------------------------------------------------
_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "database"
    / "migrations"
    / "versions"
    / "005_tenant_id_uuid.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_rev_005_tenant_id_uuid", _MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rev005 = _load_migration()

PROBE_TABLE = "census_probe"

# A single legitimate tenant, canonical lowercase-dashed. The CONTROL case
# below writes MANY rows carrying exactly this value.
LEGIT_TENANT = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def census_conn():
    """A sync postgres connection with a disposable ``census_probe`` table.

    ``tenant_id`` is ``varchar(255)`` — the pre-005 shape — so the fixture can
    seed values a ``uuid`` column would refuse, which is the entire point.
    Everything happens inside a transaction that is rolled back.
    """
    engine = sa.create_engine(_TEST_DATABASE_URL)
    with engine.connect() as conn:
        trans = conn.begin()
        conn.execute(sa.text(f'DROP TABLE IF EXISTS "{PROBE_TABLE}"'))
        conn.execute(
            sa.text(
                f'CREATE TABLE "{PROBE_TABLE}" ('
                "  id serial PRIMARY KEY,"
                "  tenant_id varchar(255) NOT NULL"
                ")"
            )
        )
        yield conn
        trans.rollback()
    engine.dispose()


def _seed(conn: sa.Connection, values: list[str]) -> None:
    conn.execute(
        sa.text(f'INSERT INTO "{PROBE_TABLE}" (tenant_id) VALUES (:v)'),  # noqa: S608
        [{"v": v} for v in values],
    )


# ---------------------------------------------------------------------------
# CONTROL — the case that catches the `count(*)` mistake
# ---------------------------------------------------------------------------


def test_control_many_rows_sharing_one_legitimate_uuid_migrates_green(
    census_conn: sa.Connection,
) -> None:
    """A HEALTHY table: 250 rows, all one real tenant. The census MUST pass.

    This is the load-bearing test in this module. The obvious-looking spelling
    of the collision guard —

        count(*) <> count(DISTINCT lower(tenant_id))

    — counts ROWS, so it reports a "collision" for any tenant with more than
    one row. That is EVERY tenant in this service: a tenant has thousands of
    jobs. Under that spelling the migration refuses a perfectly healthy
    database, and because the tables happen to be empty in CI today, nothing
    would ever have caught it there. 250 rows of one tenant is the smallest
    fixture that does.

    The correct guard groups by the folded value and counts DISTINCT SPELLINGS
    within each group: one spelling, 250 times, is not a collision.
    """
    _seed(census_conn, [LEGIT_TENANT] * 250)

    assert rev005.case_only_collisions(census_conn, PROBE_TABLE, "tenant_id") == []
    assert rev005.non_uuid_values(census_conn, PROBE_TABLE, "tenant_id") == []

    total = rev005.census(census_conn, PROBE_TABLE, "tenant_id")
    assert total == 250

    # And the cast the census cleared actually succeeds.
    census_conn.execute(
        sa.text(
            f'ALTER TABLE "{PROBE_TABLE}" ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid'
        )
    )
    assert rev005.row_count(census_conn, PROBE_TABLE) == 250


def test_control_many_rows_many_distinct_tenants_migrates_green(
    census_conn: sa.Connection,
) -> None:
    """Several real tenants, each with many rows — still not a collision."""
    tenants = [str(uuid.uuid7()) for _ in range(4)]
    _seed(census_conn, [t for t in tenants for _ in range(30)])

    assert rev005.census(census_conn, PROBE_TABLE, "tenant_id") == 120


# ---------------------------------------------------------------------------
# RED — case-only collision
# ---------------------------------------------------------------------------


def test_case_only_collision_is_refused(census_conn: sa.Connection) -> None:
    """Two spellings of ONE uuid that differ only in case MUST be refused.

    ``::uuid`` folds them into a single value, so two row-sets that the
    application treated as different tenants silently merge. No postgres error
    is raised for that — it is a successful cast that loses data — which is why
    the census has to look for it explicitly.
    """
    upper = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"  # DIRTY-ON-PURPOSE
    lower = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"  # DIRTY-ON-PURPOSE
    # Healthy rows alongside, so the failure cannot be an artefact of the table
    # being trivially small.
    _seed(census_conn, [upper] * 3 + [lower] * 7 + [LEGIT_TENANT] * 40)

    collisions = rev005.case_only_collisions(census_conn, PROBE_TABLE, "tenant_id")
    assert collisions == [(lower, 2)]

    with pytest.raises(rev005.TenantCensusError) as exc:
        rev005.census(census_conn, PROBE_TABLE, "tenant_id")
    assert "differ only in letter case" in str(exc.value)
    assert lower in str(exc.value)


# ---------------------------------------------------------------------------
# RED — non-uuid value
# ---------------------------------------------------------------------------


def test_non_uuid_value_is_refused_and_named_with_its_count(
    census_conn: sa.Connection,
) -> None:
    """A legacy slug MUST be refused, and the message must NAME it and count it.

    An operator seeing this has to choose between backfilling and deleting, and
    cannot do either without knowing which value and how many rows.
    """
    _seed(
        census_conn,
        ["tenant-1"] * 5  # DIRTY-ON-PURPOSE
        + ["n8n-dev"] * 2  # DIRTY-ON-PURPOSE
        + [LEGIT_TENANT] * 10,
    )

    bad = rev005.non_uuid_values(census_conn, PROBE_TABLE, "tenant_id")
    assert bad == [("tenant-1", 5), ("n8n-dev", 2)]  # DIRTY-ON-PURPOSE

    with pytest.raises(rev005.TenantCensusError) as exc:
        rev005.census(census_conn, PROBE_TABLE, "tenant_id")
    message = str(exc.value)
    assert "non-uuid" in message
    assert "'tenant-1' x5" in message  # DIRTY-ON-PURPOSE
    assert "'n8n-dev' x2" in message  # DIRTY-ON-PURPOSE
    assert "of 17 row(s)" in message


@pytest.mark.parametrize(
    "sneaky",
    [
        "{11111111-1111-4111-8111-111111111112}",  # DIRTY-ON-PURPOSE braced
        "urn:uuid:11111111-1111-4111-8111-111111111113",  # DIRTY-ON-PURPOSE urn
        "11111111111141118111111111111114",  # DIRTY-ON-PURPOSE undashed
        "11111111-1111-4111-8111-11111111111",  # DIRTY-ON-PURPOSE truncated
    ],
)
def test_non_canonical_uuid_spellings_are_refused(census_conn: sa.Connection, sneaky: str) -> None:
    """A value ``uuid.UUID()`` would accept but postgres would REWRITE is dirty too.

    The braced / ``urn:`` / undashed spellings survive ``::uuid`` — postgres
    normalises them — so the cast would not error. But every reader that
    compared them as TEXT against the canonical form was already silently
    missing, and normalising them as a side effect of a type change hides that
    the data was wrong. Surface it instead; the operator canonicalises
    deliberately.
    """
    _seed(census_conn, [sneaky] + [LEGIT_TENANT] * 5)

    with pytest.raises(rev005.TenantCensusError, match="non-uuid"):
        rev005.census(census_conn, PROBE_TABLE, "tenant_id")


def test_census_reports_the_row_count_it_cleared(census_conn: sa.Connection) -> None:
    """The return value is the pre-cast row count the migration logs."""
    _seed(census_conn, [str(uuid.uuid7()) for _ in range(7)])
    assert rev005.census(census_conn, PROBE_TABLE, "tenant_id") == 7
    assert rev005.census(census_conn, PROBE_TABLE, "tenant_id") == rev005.row_count(
        census_conn, PROBE_TABLE
    )


def test_empty_table_passes_the_census() -> None:
    """The four real tables are empty today — the census must not trip on that.

    Documented explicitly because "it passed" on an empty table proves nothing
    about the guard, which is exactly why the CONTROL case above exists.
    """
    engine = sa.create_engine(_TEST_DATABASE_URL)
    with engine.connect() as conn:
        trans = conn.begin()
        conn.execute(sa.text(f'DROP TABLE IF EXISTS "{PROBE_TABLE}"'))
        conn.execute(
            sa.text(f'CREATE TABLE "{PROBE_TABLE}" (id serial PRIMARY KEY, tenant_id varchar(255))')
        )
        assert rev005.census(conn, PROBE_TABLE, "tenant_id") == 0
        trans.rollback()
    engine.dispose()


def test_migration_drives_ddl_from_literal_table_names_not_metadata() -> None:
    """``TABLES`` must stay hardcoded literals.

    A metadata walk re-reads the ORM at upgrade time. Once the models declare
    ``uuid`` (they now do), a metadata-driven revision generates predicates in
    the NEW type while replaying against the OLD schema and dies with
    ``operator does not exist: character varying = uuid`` — invisible in a
    long-lived production database, fatal in CI, which always upgrades from
    zero.
    """
    assert rev005.TABLES == [
        ("tts_jobs", "tenant_id"),
        ("tts_usage", "tenant_id"),
        ("tenant_quotas", "tenant_id"),
        ("usage_alerts", "tenant_id"),
    ]
    # Checked with `ast`, not `in source`: the module's comments talk ABOUT
    # `Base.metadata` at length to explain why it is banned, and a substring
    # check would fire on the explanation. Only real code counts.
    tree = ast.parse(_MIGRATION_PATH.read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("src.") for name in imported), (
        f"revision 005 must not import application code (imports: {sorted(imported)})"
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "metadata" for node in ast.walk(tree)
    ), "revision 005 must not walk a SQLAlchemy MetaData"


# ---------------------------------------------------------------------------
# The ORM side: uuid column, str Python value
# ---------------------------------------------------------------------------


def test_every_tenant_column_is_a_postgres_uuid_with_as_uuid_false() -> None:
    """All four tenant columns are ``uuid``, and all four hand back ``str``.

    ``as_uuid=False`` is the load-bearing half. With ``as_uuid=True`` every read
    returns a ``uuid.UUID`` object, which:

      * compares FALSE against the request-scoped ``str`` the routes carry
        (``TTSJob.tenant_id == tenant_id`` in Python, ``job.tenant_id ==
        tenant_id`` in the WS/notification paths), and
      * is not JSON-serialisable, so it dies inside the Redis notification
        payload rather than at the call site.

    That combination is what has frozen outbox paths elsewhere in the fleet.
    This test exists so a future "tidy-up" to ``as_uuid=True`` goes red here
    instead of in production.
    """
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID

    from src.database.models import Base

    tenant_columns = {
        table.name: table.columns["tenant_id"]
        for table in Base.metadata.sorted_tables
        if "tenant_id" in table.columns
    }
    assert set(tenant_columns) == {"tts_jobs", "tts_usage", "tenant_quotas", "usage_alerts"}

    for name, column in tenant_columns.items():
        assert isinstance(column.type, PG_UUID), f"{name}.tenant_id is {column.type!r}, not uuid"
        assert column.type.as_uuid is False, f"{name}.tenant_id must use as_uuid=False"
        assert column.nullable is False, f"{name}.tenant_id must stay NOT NULL"


async def test_a_tenant_round_trips_out_of_postgres_as_a_plain_str(db_session) -> None:
    """Write through the ORM, read back, and get ``str`` — not ``uuid.UUID``.

    The type-declaration test above can be satisfied by a mock; this one cannot.
    It is the assertion that actually proves the ``str`` contract survives a
    real psycopg round trip.
    """
    from src.database.models import TTSJob

    tenant = str(uuid.uuid7())
    db_session.add(
        TTSJob(
            job_id=str(uuid.uuid7()),
            tenant_id=tenant,
            text="hello",
            text_length=5,
            provider="stub",
            voice="af_heart",
        )
    )
    await db_session.commit()

    stored = (
        await db_session.execute(sa.select(TTSJob).where(TTSJob.tenant_id == tenant))
    ).scalar_one()
    assert isinstance(stored.tenant_id, str), type(stored.tenant_id)
    assert stored.tenant_id == tenant


async def test_a_non_uuid_tenant_cannot_be_persisted_at_all(db_session) -> None:
    """The column itself refuses a legacy slug. This is the point of the change.

    Before this revision the same insert succeeded silently and the slug sat in
    the table forever, unnoticed until something tried to compare it with a real
    tenant.
    """
    from src.database.models import TTSJob

    db_session.add(
        TTSJob(
            job_id=str(uuid.uuid7()),
            tenant_id="n8n-dev",  # DIRTY-ON-PURPOSE
            text="hello",
            text_length=5,
            provider="stub",
            voice="af_heart",
        )
    )
    with pytest.raises(Exception):  # noqa: B017 -- driver/SQLAlchemy wrap it differently
        await db_session.commit()
