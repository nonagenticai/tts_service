"""Unit tests for Redis client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from src.cache.redis_client import RedisClient


class TestRedisClientInit:
    def test_init_defaults(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        assert client._redis_url == "redis://localhost:6379/0"
        assert client._pool is None
        assert client._client is None
        assert client._binary_pool is None
        assert client._binary_client is None


class TestRedisClientSerialization:
    def test_serialize_dict(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        result = client._serialize({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_serialize_string(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        result = client._serialize("hello")
        assert json.loads(result) == "hello"

    def test_serialize_pydantic_model(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"field": "value"}
        result = client._serialize(mock_model)
        assert json.loads(result) == {"field": "value"}

    def test_deserialize_valid_json(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        result = client._deserialize('{"key": "value"}')
        assert result == {"key": "value"}

    def test_deserialize_none(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        result = client._deserialize(None)
        assert result is None

    def test_deserialize_invalid_json(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        result = client._deserialize("not valid json{{{")
        assert result is None


class TestRedisClientEnsureConnected:
    def test_ensure_connected_raises_when_not_connected(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()

    def test_ensure_binary_connected_raises_when_not_connected(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_binary_connected()

    def test_ensure_connected_returns_client(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = MagicMock()
        client._client = mock_redis
        result = client._ensure_connected()
        assert result is mock_redis

    def test_ensure_binary_connected_returns_client(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = MagicMock()
        client._binary_client = mock_redis
        result = client._ensure_binary_connected()
        assert result is mock_redis


class TestRedisClientJsonOps:
    @pytest.mark.asyncio
    async def test_get_returns_deserialized_value(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value='{"key": "value"}')
        client._client = mock_redis

        result = await client.get("test_key")
        assert result == {"key": "value"}
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        client._client = mock_redis

        result = await client.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=RedisError("connection error"))
        client._client = mock_redis

        result = await client.get("error_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_without_ttl(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        client._client = mock_redis

        result = await client.set("key", {"data": 42})
        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_ttl(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(return_value=True)
        client._client = mock_redis

        result = await client.set("key", {"data": 42}, ttl=300)
        assert result is True
        mock_redis.setex.assert_called_once_with("key", 300, json.dumps({"data": 42}))

    @pytest.mark.asyncio
    async def test_set_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=RedisError("write error"))
        client._client = mock_redis

        result = await client.set("key", "val")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_success(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)
        client._client = mock_redis

        result = await client.delete("key")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(side_effect=RedisError("delete error"))
        client._client = mock_redis

        result = await client.delete("key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.exists = AsyncMock(return_value=1)
        client._client = mock_redis

        result = await client.exists("key")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.exists = AsyncMock(return_value=0)
        client._client = mock_redis

        result = await client.exists("key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.exists = AsyncMock(side_effect=RedisError("exists error"))
        client._client = mock_redis

        result = await client.exists("key")
        assert result is False


class TestRedisClientPing:
    @pytest.mark.asyncio
    async def test_ping_success_async(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = MagicMock()
        mock_ping = AsyncMock(return_value=True)
        mock_redis.ping = MagicMock(return_value=mock_ping())
        client._client = mock_redis

        # ping returns an awaitable
        result = await client.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = MagicMock()
        mock_redis.ping = MagicMock(side_effect=RedisError("ping error"))
        client._client = mock_redis

        result = await client.ping()
        assert result is False


class TestRedisClientBinaryOps:
    @pytest.mark.asyncio
    async def test_get_bytes_returns_bytes(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b"binary data")
        client._binary_client = mock_redis

        result = await client.get_bytes("key")
        assert result == b"binary data"

    @pytest.mark.asyncio
    async def test_get_bytes_returns_none_for_non_bytes(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="not bytes")
        client._binary_client = mock_redis

        result = await client.get_bytes("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_bytes_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("binary error"))
        client._binary_client = mock_redis

        result = await client.get_bytes("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_bytes_without_ttl(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        client._binary_client = mock_redis

        result = await client.set_bytes("key", b"data")
        assert result is True
        mock_redis.set.assert_called_once_with("key", b"data")

    @pytest.mark.asyncio
    async def test_set_bytes_with_ttl(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(return_value=True)
        client._binary_client = mock_redis

        result = await client.set_bytes("key", b"data", ttl=300)
        assert result is True
        mock_redis.setex.assert_called_once_with("key", 300, b"data")

    @pytest.mark.asyncio
    async def test_set_bytes_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=ConnectionError("write error"))
        client._binary_client = mock_redis

        result = await client.set_bytes("key", b"data")
        assert result is False


class TestRedisClientPubSub:
    @pytest.mark.asyncio
    async def test_publish_success(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        client._client = mock_redis

        result = await client.publish("channel", "message")
        assert result == 1

    @pytest.mark.asyncio
    async def test_publish_handles_error(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(side_effect=ConnectionError("pub error"))
        client._client = mock_redis

        result = await client.publish("channel", "message")
        assert result == 0

    def test_get_pubsub(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        mock_redis = MagicMock()
        mock_pubsub = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub
        client._pubsub_client = mock_redis

        result = client.get_pubsub()
        assert result is mock_pubsub


class TestRedisClientContextManager:
    @pytest.mark.asyncio
    async def test_aenter_calls_connect(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            result = await client.__aenter__()
            assert result is client
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_calls_disconnect(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        with patch.object(client, "disconnect", new_callable=AsyncMock) as mock_disconnect:
            await client.__aexit__(None, None, None)
            mock_disconnect.assert_called_once()


class TestRedisClientDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        client._client = AsyncMock()
        client._pool = AsyncMock()
        client._binary_client = AsyncMock()
        client._binary_pool = AsyncMock()

        await client.disconnect()

        assert client._client is None
        assert client._pool is None
        assert client._binary_client is None
        assert client._binary_pool is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        client = RedisClient(redis_url="redis://localhost:6379/0")
        # Should not raise
        await client.disconnect()
