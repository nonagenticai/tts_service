"""Unit tests for database session module."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGetSessionMaker:
    @pytest.mark.asyncio
    @patch("w2v_common.database.session.get_engine")
    async def test_creates_session_maker(self, mock_get_engine):
        import w2v_common.database.session as mod

        original = mod._async_session_maker
        mod._async_session_maker = None

        try:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            sm = await mod.get_session_maker()
            assert sm is not None
            # Should cache the session maker
            sm2 = await mod.get_session_maker()
            assert sm is sm2
        finally:
            mod._async_session_maker = original

    @pytest.mark.asyncio
    async def test_close_session_maker_resets(self):
        import w2v_common.database.session as mod

        original = mod._async_session_maker
        mod._async_session_maker = MagicMock()

        try:
            await mod.close_session_maker()
            assert mod._async_session_maker is None
        finally:
            mod._async_session_maker = original


class TestGetDb:
    @pytest.mark.asyncio
    @patch("w2v_common.database.session.get_session_maker")
    async def test_get_db_yields_session(self, mock_get_sm):
        from src.database.session import get_db

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_sm = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_sm.return_value = mock_context
        mock_get_sm.return_value = mock_sm

        gen = get_db()
        session = await gen.__anext__()
        assert session is mock_session

        # Complete the generator
        with contextlib.suppress(StopAsyncIteration):
            await gen.__anext__()

    @pytest.mark.asyncio
    @patch("w2v_common.database.session.get_session_maker")
    async def test_get_db_rollback_on_error(self, mock_get_sm):
        from src.database.session import get_db

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock(side_effect=Exception("commit failed"))
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_sm = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_sm.return_value = mock_context
        mock_get_sm.return_value = mock_sm

        gen = get_db()
        session = await gen.__anext__()
        assert session is mock_session


class TestGetDbSession:
    @pytest.mark.asyncio
    @patch("w2v_common.database.session.get_session_maker")
    async def test_get_db_session_context_manager(self, mock_get_sm):
        from src.database.session import get_db_session

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        mock_sm = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_sm.return_value = mock_context
        mock_get_sm.return_value = mock_sm

        async with get_db_session() as session:
            assert session is mock_session
