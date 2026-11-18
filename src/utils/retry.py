"""Retry helpers with exponential backoff for sync and async callers."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def retry_with_backoff[T](
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call *fn* with exponential backoff on retryable exceptions."""
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            delay = delay * (0.5 + random.random())  # noqa: S311 -- jitter for retry, not crypto
            logger.warning(
                "Retry %d/%d after %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)
    if last_exc is None:
        raise RuntimeError("retry loop exited without an exception (max_retries < 0?)")
    raise last_exc


async def async_retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Async version of retry_with_backoff."""
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            delay = delay * (0.5 + random.random())  # noqa: S311 -- jitter for retry, not crypto
            logger.warning(
                "Async retry %d/%d after %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    if last_exc is None:
        raise RuntimeError("retry loop exited without an exception (max_retries < 0?)")
    raise last_exc
