"""Unit tests for the notification service."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.notification_service import PUBSUB_CHANNEL, JobEvent, NotificationService
from tests.conftest import tenant_uuid


class TestJobEvent:
    def test_creation(self):
        event = JobEvent(
            event="job_completed",
            job_id="abc-123",
            tenant_id=tenant_uuid("tenant-1"),
        )
        assert event.event == "job_completed"
        assert event.job_id == "abc-123"
        assert event.tenant_id == tenant_uuid("tenant-1")
        assert event.timestamp is not None

    def test_to_json(self):
        event = JobEvent(
            event="job_started",
            job_id="abc-123",
            tenant_id=tenant_uuid("tenant-1"),
        )
        json_str = event.to_json()
        data = json.loads(json_str)
        assert data["event"] == "job_started"
        assert data["job_id"] == "abc-123"

    def test_from_json(self):
        original = JobEvent(
            event="job_failed",
            job_id="abc-123",
            tenant_id=tenant_uuid("tenant-1"),
            batch_id="batch-1",
            payload={"error": "boom"},
        )
        json_str = original.to_json()
        restored = JobEvent.from_json(json_str)
        assert restored.event == "job_failed"
        assert restored.job_id == "abc-123"
        assert restored.batch_id == "batch-1"
        assert restored.payload == {"error": "boom"}

    def test_roundtrip(self):
        event = JobEvent(
            event="batch_progress",
            job_id="j-1",
            tenant_id=tenant_uuid("t-1"),
            batch_id="b-1",
        )
        restored = JobEvent.from_json(event.to_json())
        assert restored.event == event.event
        assert restored.job_id == event.job_id
        assert restored.tenant_id == event.tenant_id
        assert restored.batch_id == event.batch_id


class TestNotificationService:
    def test_init_with_redis(self):
        redis = MagicMock()
        service = NotificationService(redis_client=redis)
        assert service._redis is redis

    def test_init_without_redis(self):
        service = NotificationService()
        assert service._redis is None

    @pytest.mark.asyncio
    async def test_start_with_redis(self):
        redis = MagicMock()
        service = NotificationService(redis_client=redis)
        await service.start()
        assert service._running is True
        assert service._subscriber_task is not None
        # Clean up
        await service.stop()

    @pytest.mark.asyncio
    async def test_start_without_redis(self):
        service = NotificationService()
        await service.start()
        assert service._subscriber_task is None

    @pytest.mark.asyncio
    async def test_stop(self):
        service = NotificationService()
        service._running = True

        # Create a real asyncio task that just sleeps forever (will be cancelled)
        async def dummy_loop():
            while True:
                await asyncio.sleep(100)

        task = asyncio.create_task(dummy_loop())
        service._subscriber_task = task
        await service.stop()
        assert service._running is False
        assert service._subscriber_task is None

    def test_subscribe(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe("job:abc", queue)
        assert queue in service._subscriptions["job:abc"]

    def test_unsubscribe(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe("job:abc", queue)
        service.unsubscribe("job:abc", queue)
        assert "job:abc" not in service._subscriptions

    def test_unsubscribe_nonexistent_key(self):
        service = NotificationService()
        queue = asyncio.Queue()
        # Should not raise
        service.unsubscribe("nonexistent", queue)

    def test_unsubscribe_all(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe("job:abc", queue)
        service.subscribe(f"tenant:{tenant_uuid('t1')}", queue)
        service.unsubscribe_all(queue)
        assert "job:abc" not in service._subscriptions
        assert f"tenant:{tenant_uuid('t1')}" not in service._subscriptions

    @pytest.mark.asyncio
    async def test_publish_event_dispatches_locally(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe("job:abc-123", queue)

        event = JobEvent(event="job_completed", job_id="abc-123", tenant_id=tenant_uuid("t1"))
        await service.publish_event(event)

        assert not queue.empty()
        msg = queue.get_nowait()
        data = json.loads(msg)
        assert data["event"] == "job_completed"

    @pytest.mark.asyncio
    async def test_publish_event_publishes_to_redis(self):
        mock_redis = AsyncMock()
        service = NotificationService(redis_client=mock_redis)

        event = JobEvent(event="job_started", job_id="j1", tenant_id=tenant_uuid("t1"))
        await service.publish_event(event)

        mock_redis.publish.assert_called_once_with(PUBSUB_CHANNEL, event.to_json())

    @pytest.mark.asyncio
    async def test_publish_event_redis_error_handled(self):
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(side_effect=ConnectionError("redis error"))
        service = NotificationService(redis_client=mock_redis)

        event = JobEvent(event="job_started", job_id="j1", tenant_id=tenant_uuid("t1"))
        # Should not raise
        await service.publish_event(event)

    def test_dispatch_local_to_job_subscriber(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe("job:j1", queue)

        event = JobEvent(event="job_completed", job_id="j1", tenant_id=tenant_uuid("t1"))
        service._dispatch_local(event)

        assert not queue.empty()

    def test_dispatch_local_to_tenant_subscriber(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe(f"tenant:{tenant_uuid('t1')}", queue)

        event = JobEvent(event="job_completed", job_id="j1", tenant_id=tenant_uuid("t1"))
        service._dispatch_local(event)

        assert not queue.empty()

    def test_dispatch_local_to_batch_subscriber(self):
        service = NotificationService()
        queue = asyncio.Queue()
        service.subscribe("batch:b1", queue)

        event = JobEvent(
            event="batch_progress", job_id="j1", tenant_id=tenant_uuid("t1"), batch_id="b1"
        )
        service._dispatch_local(event)

        assert not queue.empty()

    def test_dispatch_local_full_queue(self):
        service = NotificationService()
        queue = asyncio.Queue(maxsize=1)
        queue.put_nowait("full")
        service.subscribe("job:j1", queue)

        event = JobEvent(event="job_completed", job_id="j1", tenant_id=tenant_uuid("t1"))
        # Should not raise even though queue is full
        service._dispatch_local(event)
