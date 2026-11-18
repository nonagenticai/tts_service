"""Redis client for TTS service caching and pub/sub -- extends w2v_common.

Adds binary storage (get_bytes/set_bytes) and pub/sub (publish/get_pubsub)
on top of the base w2v_common RedisClient.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError
from w2v_common.cache.redis_client import RedisClient as _BaseRedisClient

logger = logging.getLogger(__name__)

# ``RedisError`` is NOT optional in the handlers below and must never be
# "simplified" away. redis-py's exception tree is entirely independent of the
# builtins::
#
#     redis.exceptions.ConnectionError.__mro__
#         -> (ConnectionError, RedisError, Exception, BaseException, object)
#     issubclass(redis.exceptions.ConnectionError, builtins.ConnectionError) -> False
#     issubclass(redis.exceptions.TimeoutError,    builtins.TimeoutError)    -> False
#     issubclass(redis.exceptions.RedisError,      OSError)                  -> False
#
# So ``except (OSError, ConnectionError, TimeoutError)`` catches only the
# builtins and lets EVERY real Redis failure escape — which turns this
# fail-soft wrapper (documented to return None/False/0) into a hard failure
# that propagates into its callers. The builtins stay in the tuples because a
# socket-level error can still surface as one.


class RedisClient(_BaseRedisClient):
    """Extended Redis client with binary storage and pub/sub support."""

    def __init__(
        self,
        redis_url: str | None = None,
        max_connections: int | None = None,
        socket_timeout: int | None = None,
        socket_connect_timeout: int | None = None,
        decode_responses: bool | None = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=decode_responses,
        )
        # Separate pool for binary operations (no decode_responses)
        self._binary_pool: ConnectionPool | None = None
        self._binary_client: redis.Redis | None = None
        # Separate pool for PubSub (no socket timeout -- subscribers idle indefinitely)
        self._pubsub_pool: ConnectionPool | None = None
        self._pubsub_client: redis.Redis | None = None

    async def connect(self) -> None:
        """Initialize base + binary + pubsub connection pools."""
        await super().connect()

        # Binary pool (no decode_responses) for raw bytes
        if self._binary_pool is None:
            self._binary_pool = ConnectionPool.from_url(
                self._redis_url,
                max_connections=max(3, self._max_connections // 3),
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=False,
            )
            self._binary_client = redis.Redis(connection_pool=self._binary_pool)

        # PubSub pool -- no socket timeout so subscribers can idle waiting for messages
        if self._pubsub_pool is None:
            self._pubsub_pool = ConnectionPool.from_url(
                self._redis_url,
                max_connections=2,
                socket_timeout=None,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=self._decode_responses,
            )
            self._pubsub_client = redis.Redis(connection_pool=self._pubsub_pool)

    async def disconnect(self) -> None:
        """Close all Redis connection pools and release resources."""
        await super().disconnect()
        if self._binary_client:
            await self._binary_client.aclose()
            self._binary_client = None
        if self._binary_pool:
            await self._binary_pool.aclose()
            self._binary_pool = None
        if self._pubsub_client:
            await self._pubsub_client.aclose()
            self._pubsub_client = None
        if self._pubsub_pool:
            await self._pubsub_pool.aclose()
            self._pubsub_pool = None

    def _ensure_binary_connected(self) -> redis.Redis:
        if self._binary_client is None:
            raise RuntimeError("RedisClient not connected. Call connect() first.")
        return self._binary_client

    # ========================================================================
    # Binary Operations (for audio cache data)
    # ========================================================================

    async def get_bytes(self, key: str) -> bytes | None:
        """Get raw binary data from Redis."""
        client = self._ensure_binary_connected()
        try:
            value = await client.get(key)
            return value if isinstance(value, bytes) else None
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error getting binary key '%s': %s", key, e)
            return None

    async def set_bytes(self, key: str, value: bytes, ttl: int | None = None) -> bool:
        """Store raw binary data in Redis."""
        client = self._ensure_binary_connected()
        try:
            if ttl is not None:
                result = await client.setex(key, ttl, value)
            else:
                result = await client.set(key, value)
            return bool(result)
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error setting binary key '%s': %s", key, e)
            return False

    # ========================================================================
    # Pub/Sub Operations (for WebSocket notifications)
    # ========================================================================

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a Redis Pub/Sub channel."""
        client = self._ensure_connected()
        try:
            result = await client.publish(channel, message)
            return int(result)
        except (RedisError, OSError, ConnectionError, TimeoutError) as e:
            logger.error("Error publishing to channel '%s': %s", channel, e)
            return 0

    def get_pubsub(self) -> redis.client.PubSub:
        """Get a Pub/Sub object for subscribing to channels.

        Uses a dedicated connection pool with no socket timeout,
        since PubSub subscribers idle indefinitely waiting for messages.
        """
        if self._pubsub_client is None:
            raise RuntimeError("RedisClient not connected. Call connect() first.")
        return self._pubsub_client.pubsub()


_redis_client: RedisClient | None = None


@lru_cache
def get_redis_client() -> RedisClient:
    """Return the process-wide TTS Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def init_redis() -> RedisClient:
    """Initialize and connect the process-wide TTS Redis client singleton."""
    client = get_redis_client()
    await client.connect()
    return client


async def close_redis() -> None:
    """Disconnect and clear the process-wide TTS Redis client singleton."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.disconnect()
        _redis_client = None
        get_redis_client.cache_clear()
