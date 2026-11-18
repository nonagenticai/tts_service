"""Database connection module -- delegates to w2v_common."""

from w2v_common.database.connection import (
    DATABASE_URL as DATABASE_URL,
)
from w2v_common.database.connection import (
    DB_HOST as DB_HOST,
)
from w2v_common.database.connection import (
    DB_NAME as DB_NAME,
)
from w2v_common.database.connection import (
    DB_PORT as DB_PORT,
)
from w2v_common.database.connection import (
    _ensure_database_exists as _ensure_database_exists,
)
from w2v_common.database.connection import (
    close_engine as close_engine,
)
from w2v_common.database.connection import (
    create_engine_with_retry as create_engine_with_retry,
)
from w2v_common.database.connection import (
    get_engine as get_engine,
)
from w2v_common.database.connection import (
    health_check as health_check,
)

__all__ = [
    "DATABASE_URL",
    "DB_HOST",
    "DB_NAME",
    "DB_PORT",
    "_ensure_database_exists",
    "close_engine",
    "create_engine_with_retry",
    "get_engine",
    "health_check",
]
