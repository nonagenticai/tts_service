"""Real-assertion tests for retry utility functions.

These tests pin the exponential-backoff sequence, the ``max_delay`` cap, and
the jitter range by asserting on the actual delays passed to sleep. They are
designed to fail when someone changes the backoff formula in
``src/utils/retry.py`` — in particular, the ``2**attempt`` exponent at line 31.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.utils.retry import async_retry_with_backoff, retry_with_backoff


@pytest.fixture
def freeze_jitter() -> Iterator[None]:
    """Freeze ``random.random`` to ``0.5`` so jitter multiplier is exactly ``1.0``.

    The formula in ``retry.py:32`` is ``delay * (0.5 + random.random())``; with
    the random value pinned at ``0.5`` the multiplier becomes ``1.0`` and the
    final delay equals the pre-jitter exponential delay exactly, which lets
    tests assert integer delays.
    """
    with patch("src.utils.retry.random.random", return_value=0.5):
        yield


class TestRetryWithBackoff:
    @patch("src.utils.retry.time.sleep")
    def test_success_first_try_no_sleep(self, mock_sleep: MagicMock) -> None:
        fn: MagicMock = MagicMock(return_value="result")
        result = retry_with_backoff(fn, max_retries=3)
        assert result == "result"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @pytest.mark.usefixtures("freeze_jitter")
    @patch("src.utils.retry.time.sleep")
    def test_success_after_one_retry_uses_base_delay(self, mock_sleep: MagicMock) -> None:
        """With jitter frozen, the first retry sleeps for exactly ``base_delay``."""
        fn: MagicMock = MagicMock(side_effect=[ValueError("fail"), "result"])
        result = retry_with_backoff(fn, max_retries=3, base_delay=1.0)
        assert result == "result"
        assert fn.call_count == 2
        assert mock_sleep.call_args_list == [call(1.0)]

    @pytest.mark.usefixtures("freeze_jitter")
    @patch("src.utils.retry.time.sleep")
    def test_exponential_backoff_sequence(self, mock_sleep: MagicMock) -> None:
        """Regression guard: delays follow ``base_delay * 2**attempt`` before jitter.

        Tripwire for edits to ``retry.py:31`` — e.g. changing ``2**attempt`` to
        ``2**(attempt-1)`` or ``2**(attempt+1)`` causes this test to fail.
        """
        fn: MagicMock = MagicMock(
            side_effect=[ValueError("e0"), ValueError("e1"), ValueError("e2"), "ok"]
        )
        result = retry_with_backoff(fn, max_retries=3, base_delay=1.0, max_delay=30.0)
        assert result == "ok"
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == [1.0, 2.0, 4.0]

    @pytest.mark.usefixtures("freeze_jitter")
    @patch("src.utils.retry.time.sleep")
    def test_max_delay_caps_exponential_growth(self, mock_sleep: MagicMock) -> None:
        """Raw ``2**attempt`` sequence 10→20→40 must be capped at ``max_delay=15``."""
        fn: MagicMock = MagicMock(
            side_effect=[ValueError("a"), ValueError("b"), ValueError("c"), "ok"]
        )
        result = retry_with_backoff(fn, max_retries=3, base_delay=10.0, max_delay=15.0)
        assert result == "ok"
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == [10.0, 15.0, 15.0]

    @patch("src.utils.retry.time.sleep")
    def test_jitter_stays_in_half_to_one_and_a_half_range(self, mock_sleep: MagicMock) -> None:
        """Without freezing ``random``, delays must fall in ``[0.5*base, 1.5*base)``.

        Catches regressions that remove jitter entirely (all delays equal) or
        widen the range (e.g. ``random() * 2``).
        """
        observed: list[float] = []
        for _ in range(200):
            mock_sleep.reset_mock()
            fn: MagicMock = MagicMock(side_effect=[ValueError("fail"), "result"])
            retry_with_backoff(fn, max_retries=3, base_delay=1.0)
            observed.append(float(mock_sleep.call_args_list[0].args[0]))
        assert all(0.5 <= d < 1.5 for d in observed), (
            f"jitter out of range: min={min(observed)}, max={max(observed)}"
        )
        # Distribution must actually vary — catches "jitter removed, always returns 1.0".
        assert max(observed) - min(observed) > 0.3

    @pytest.mark.usefixtures("freeze_jitter")
    @patch("src.utils.retry.time.sleep")
    def test_exhausts_retries_raises_last_exception(self, mock_sleep: MagicMock) -> None:
        fn: MagicMock = MagicMock(side_effect=ValueError("always fails"))
        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(fn, max_retries=2)
        # 1 initial + 2 retries = 3 calls; sleeps only between retries, not after last fail.
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.utils.retry.time.sleep")
    def test_non_retryable_exception_propagates_without_sleeping(
        self, mock_sleep: MagicMock
    ) -> None:
        fn: MagicMock = MagicMock(side_effect=TypeError("not retryable"))
        with pytest.raises(TypeError):
            retry_with_backoff(fn, max_retries=3, retryable_exceptions=(ValueError,))
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("src.utils.retry.time.sleep")
    def test_zero_retries_never_sleeps(self, mock_sleep: MagicMock) -> None:
        fn: MagicMock = MagicMock(side_effect=ValueError("fail"))
        with pytest.raises(ValueError):
            retry_with_backoff(fn, max_retries=0)
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    def test_negative_max_retries_raises_runtime_error(self) -> None:
        """Edge case: ``range(max_retries + 1)`` is empty, loop never runs.

        Exercises the dead-code guard at ``retry.py:41-42``.
        """
        fn: MagicMock = MagicMock()
        with pytest.raises(RuntimeError, match="retry loop exited without an exception"):
            retry_with_backoff(fn, max_retries=-1)
        fn.assert_not_called()


class TestAsyncRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_success_first_try_no_sleep(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(return_value="result")
            result = await async_retry_with_backoff(fn, max_retries=3)
            assert result == "result"
            fn.assert_called_once()
            mock_sleep.assert_not_called()

    @pytest.mark.usefixtures("freeze_jitter")
    @pytest.mark.asyncio
    async def test_success_after_one_retry_uses_base_delay(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(side_effect=[ValueError("fail"), "result"])
            result = await async_retry_with_backoff(fn, max_retries=3, base_delay=1.0)
            assert result == "result"
            assert fn.call_count == 2
            assert mock_sleep.call_args_list == [call(1.0)]

    @pytest.mark.usefixtures("freeze_jitter")
    @pytest.mark.asyncio
    async def test_exponential_backoff_sequence(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(
                side_effect=[ValueError("e0"), ValueError("e1"), ValueError("e2"), "ok"]
            )
            result = await async_retry_with_backoff(
                fn, max_retries=3, base_delay=1.0, max_delay=30.0
            )
            assert result == "ok"
            sleeps = [c.args[0] for c in mock_sleep.call_args_list]
            assert sleeps == [1.0, 2.0, 4.0]

    @pytest.mark.usefixtures("freeze_jitter")
    @pytest.mark.asyncio
    async def test_max_delay_caps_exponential_growth(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(
                side_effect=[ValueError("a"), ValueError("b"), ValueError("c"), "ok"]
            )
            result = await async_retry_with_backoff(
                fn, max_retries=3, base_delay=10.0, max_delay=15.0
            )
            assert result == "ok"
            sleeps = [c.args[0] for c in mock_sleep.call_args_list]
            assert sleeps == [10.0, 15.0, 15.0]

    @pytest.mark.asyncio
    async def test_jitter_stays_in_half_to_one_and_a_half_range(self) -> None:
        observed: list[float] = []
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            for _ in range(200):
                mock_sleep.reset_mock()
                fn: AsyncMock = AsyncMock(side_effect=[ValueError("fail"), "result"])
                await async_retry_with_backoff(fn, max_retries=3, base_delay=1.0)
                observed.append(float(mock_sleep.call_args_list[0].args[0]))
        assert all(0.5 <= d < 1.5 for d in observed), (
            f"jitter out of range: min={min(observed)}, max={max(observed)}"
        )
        assert max(observed) - min(observed) > 0.3

    @pytest.mark.usefixtures("freeze_jitter")
    @pytest.mark.asyncio
    async def test_exhausts_retries_raises_last_exception(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(side_effect=ValueError("always fails"))
            with pytest.raises(ValueError, match="always fails"):
                await async_retry_with_backoff(fn, max_retries=2)
            assert fn.call_count == 3
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates_without_sleeping(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(side_effect=TypeError("not retryable"))
            with pytest.raises(TypeError):
                await async_retry_with_backoff(
                    fn, max_retries=3, retryable_exceptions=(ValueError,)
                )
            fn.assert_called_once()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_retries_never_sleeps(self) -> None:
        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fn: AsyncMock = AsyncMock(side_effect=ValueError("fail"))
            with pytest.raises(ValueError):
                await async_retry_with_backoff(fn, max_retries=0)
            fn.assert_called_once()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_negative_max_retries_raises_runtime_error(self) -> None:
        fn: AsyncMock = AsyncMock()
        with pytest.raises(RuntimeError, match="retry loop exited without an exception"):
            await async_retry_with_backoff(fn, max_retries=-1)
        fn.assert_not_called()
