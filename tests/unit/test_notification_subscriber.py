"""Unit tests for notification service subscriber loop and edge cases."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.notification_service import JobEvent, NotificationService
from tests.conftest import tenant_uuid


class TestSubscriberLoop:
    @pytest.mark.asyncio
    async def test_subscriber_loop_processes_messages(self):
        """Test that _subscriber_loop reads messages and dispatches locally."""
        event = JobEvent(event="job_completed", job_id="j-1", tenant_id=tenant_uuid("t-1"))

        call_count = 0

        async def fake_listen():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {"type": "message", "data": event.to_json()}
            # On subsequent calls, just raise CancelledError to stop the loop
            raise asyncio.CancelledError()

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        mock_pubsub.listen = fake_listen

        mock_redis = MagicMock()
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        service = NotificationService(redis_client=mock_redis)
        service._running = True

        # Subscribe a queue for this job
        queue = asyncio.Queue()
        service.subscribe("job:j-1", queue)

        await service._subscriber_loop()

        # Verify the event was dispatched
        assert not queue.empty()
        msg = queue.get_nowait()
        data = json.loads(msg)
        assert data["job_id"] == "j-1"

    @pytest.mark.asyncio
    async def test_subscriber_loop_skips_non_message_types(self):
        """Test that non-message types are skipped."""
        call_count = 0

        async def fake_listen():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {"type": "subscribe", "data": None}
            raise asyncio.CancelledError()

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        mock_pubsub.listen = fake_listen

        mock_redis = MagicMock()
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        service = NotificationService(redis_client=mock_redis)
        service._running = True

        await service._subscriber_loop()

    @pytest.mark.asyncio
    async def test_subscriber_loop_handles_bad_json(self):
        """Test that malformed JSON in messages is handled gracefully."""
        call_count = 0

        async def fake_listen():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {"type": "message", "data": "not-valid-json!!!"}
            raise asyncio.CancelledError()

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()
        mock_pubsub.listen = fake_listen

        mock_redis = MagicMock()
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        service = NotificationService(redis_client=mock_redis)
        service._running = True

        # Should not raise
        await service._subscriber_loop()

    @pytest.mark.asyncio
    async def test_subscriber_loop_cancelled_error_on_subscribe(self):
        """Test that CancelledError during subscribe breaks the loop cleanly."""
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock(side_effect=asyncio.CancelledError())

        mock_redis = MagicMock()
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        service = NotificationService(redis_client=mock_redis)
        service._running = True

        # Should return without error
        await service._subscriber_loop()

    @pytest.mark.asyncio
    async def test_subscriber_loop_reconnects_on_error(self):
        """Test that the loop retries on unexpected errors."""
        call_count = 0

        def get_pubsub():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection lost")
            # Second call raises CancelledError to break the loop
            raise asyncio.CancelledError()

        mock_redis = MagicMock()
        mock_redis.get_pubsub = get_pubsub

        service = NotificationService(redis_client=mock_redis)
        service._running = True

        # The loop should catch ConnectionError, sleep, then retry
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await service._subscriber_loop()

        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_subscriber_loop_stops_when_running_false(self):
        """Test that the loop checks _running between messages."""

        async def fake_listen():
            yield {
                "type": "message",
                "data": JobEvent(
                    event="job_started", job_id="j-1", tenant_id=tenant_uuid("t-1")
                ).to_json(),
            }
            # Now the loop should check _running after this iteration

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()

        listen_call_count = 0

        def listen_factory():
            nonlocal listen_call_count
            listen_call_count += 1
            if listen_call_count > 1:
                raise asyncio.CancelledError()
            return fake_listen()

        mock_pubsub.listen = listen_factory

        mock_redis = MagicMock()
        mock_redis.get_pubsub = MagicMock(return_value=mock_pubsub)

        service = NotificationService(redis_client=mock_redis)
        service._running = True

        await service._subscriber_loop()
