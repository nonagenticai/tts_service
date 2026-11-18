"""Unit tests for database connection and session modules."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.connection import DB_HOST, DB_NAME, DB_PORT


class TestDatabaseConnectionConfig:
    def test_db_name_parsed(self):
        assert isinstance(DB_NAME, str)
        assert len(DB_NAME) > 0
        # Should match the DB name from the test DATABASE_URL
        assert "tts" in DB_NAME.lower()

    def test_db_host_parsed(self):
        assert isinstance(DB_HOST, str)
        assert len(DB_HOST) > 0

    def test_db_port_parsed(self):
        assert isinstance(DB_PORT, int)
        assert DB_PORT > 0


class TestDatabaseHealthCheck:
    @pytest.mark.asyncio
    @patch("w2v_common.database.connection.get_engine")
    async def test_health_check_success(self, mock_get_engine):
        mock_conn = AsyncMock()
        mock_engine = AsyncMock()
        mock_engine.begin = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_get_engine.return_value = mock_engine

        from src.database.connection import health_check

        result = await health_check()
        assert result is True

    @pytest.mark.asyncio
    @patch("w2v_common.database.connection.get_engine")
    async def test_health_check_failure(self, mock_get_engine):
        mock_get_engine.side_effect = ConnectionError("connection failed")

        from src.database.connection import health_check

        result = await health_check()
        assert result is False


class TestCloseEngine:
    @pytest.mark.asyncio
    @patch("w2v_common.database.connection._engine", None)
    async def test_close_engine_when_none(self):
        from src.database.connection import close_engine

        # Should not raise
        await close_engine()

    @pytest.mark.asyncio
    async def test_close_engine_disposes(self):
        import w2v_common.database.connection as conn_module

        mock_engine = AsyncMock()
        original = conn_module._engine
        conn_module._engine = mock_engine
        try:
            await conn_module.close_engine()
            mock_engine.dispose.assert_called_once()
            assert conn_module._engine is None
        finally:
            conn_module._engine = original


class TestSessionModule:
    @pytest.mark.asyncio
    async def test_close_session_maker(self):
        import w2v_common.database.session as session_module

        original = session_module._async_session_maker
        session_module._async_session_maker = MagicMock()
        try:
            await session_module.close_session_maker()
            assert session_module._async_session_maker is None
        finally:
            session_module._async_session_maker = original
