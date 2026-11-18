"""Real-time job notification service using Redis Pub/Sub."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from redis.exceptions import RedisError

if TYPE_CHECKING:
    from src.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)

PUBSUB_CHANNEL = "tts:job_events"


@dataclass
class JobEvent:
    """Job lifecycle event."""

    event: str  # job_created, job_started, job_completed, job_failed, batch_progress
    job_id: str
    tenant_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    batch_id: str | None = None
    payload: dict | None = None

    def to_json(self) -> str:
        """Serialize the event to a JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> JobEvent:
        """Deserialize a JSON string into a JobEvent instance."""
        return cls(**json.loads(data))


class NotificationService:
    """Manages job event publishing and local subscriber dispatch."""

    def __init__(self, redis_client: RedisClient | None = None) -> None:
        self._redis = redis_client
        self._subscriptions: dict[str, set[asyncio.Queue]] = {}
        self._subscriber_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the Redis subscriber background task."""
        if self._redis is not None:
            self._running = True
            self._subscriber_task = asyncio.create_task(self._subscriber_loop())
            logger.info("NotificationService started with Redis Pub/Sub")
        else:
            logger.info("NotificationService started (local-only, no Redis)")

    async def stop(self) -> None:
        """Stop subscriber and clean up."""
        self._running = False
        if self._subscriber_task:
            self._subscriber_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._subscriber_task
            self._subscriber_task = None
        logger.info("NotificationService stopped")

    async def publish_event(self, event: JobEvent) -> None:
        """Publish event to Redis and dispatch to local subscribers."""
        # Dispatch locally
        self._dispatch_local(event)

        # Publish to Redis for multi-instance support
        if self._redis is not None:
            try:
                await self._redis.publish(PUBSUB_CHANNEL, event.to_json())
            # ``RedisError`` is required: redis-py's ConnectionError/TimeoutError
            # do NOT subclass the builtins of the same name, so without it every
            # real Redis outage escaped this handler and killed the caller.
            except (RedisError, OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                logger.warning("Failed to publish event to Redis: %s", exc)

    def subscribe(self, key: str, queue: asyncio.Queue) -> None:
        """Register a WebSocket's queue for a subscription key."""
        if key not in self._subscriptions:
            self._subscriptions[key] = set()
        self._subscriptions[key].add(queue)

    def unsubscribe(self, key: str, queue: asyncio.Queue) -> None:
        """Remove a WebSocket's queue from a subscription key."""
        if key in self._subscriptions:
            self._subscriptions[key].discard(queue)
            if not self._subscriptions[key]:
                del self._subscriptions[key]

    def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        """Remove a queue from all subscriptions."""
        empty_keys: list[str] = []
        for key, queues in self._subscriptions.items():
            queues.discard(queue)
            if not queues:
                empty_keys.append(key)
        for key in empty_keys:
            del self._subscriptions[key]

    def _dispatch_local(self, event: JobEvent) -> None:
        """Dispatch event to matching local subscriptions."""
        keys_to_check = [
            f"job:{event.job_id}",
            f"tenant:{event.tenant_id}",
        ]
        if event.batch_id:
            keys_to_check.append(f"batch:{event.batch_id}")

        for key in keys_to_check:
            queues = self._subscriptions.get(key)
            if queues:
                msg = event.to_json()
                for queue in list(queues):
                    try:
                        queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        logger.warning(
                            "Queue full for subscription %s, dropping event job_id=%s",
                            key,
                            event.job_id,
                        )

    async def _consume_pubsub_messages(self, pubsub: Any) -> None:
        """Iterate pubsub messages until stop is requested, dispatching events locally."""
        async for message in pubsub.listen():
            if not self._running:
                break
            if message["type"] != "message":
                continue
            try:
                event = JobEvent.from_json(message["data"])
                self._dispatch_local(event)
            except (
                RedisError,
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                ValueError,
            ) as exc:
                logger.warning("Failed to process pub/sub message: %s", exc)

    async def _close_pubsub_quietly(self, pubsub: Any, *, where: str) -> None:
        """Close a pubsub connection, swallowing/logging connection-style errors."""
        try:
            await pubsub.aclose()
        except (RedisError, OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
            logger.warning("Failed to close pubsub during %s: %s", where, exc)

    async def _subscriber_loop(self) -> None:
        """Background: read from Redis Pub/Sub and dispatch to local subscribers."""
        if self._redis is None:
            return
        pubsub: Any = None
        while self._running:
            try:
                pubsub = self._redis.get_pubsub()
                await pubsub.subscribe(PUBSUB_CHANNEL)
                await self._consume_pubsub_messages(pubsub)
                await pubsub.unsubscribe(PUBSUB_CHANNEL)
                await pubsub.aclose()
                pubsub = None
            except asyncio.CancelledError:
                break
            # ``RedisError`` is the error this reconnect arm exists to survive:
            # ``pubsub.subscribe()`` / ``listen()`` raise
            # ``redis.exceptions.ConnectionError``, which is NOT the builtin
            # ``ConnectionError``. Without it the task died on the first Redis
            # blip, leaked the pubsub connection, and left notifications dead
            # for the life of the process while /health stayed green.
            except (RedisError, OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                logger.error("Subscriber loop error, reconnecting in 5s: %s", exc)
                if pubsub is not None:
                    await self._close_pubsub_quietly(pubsub, where="reconnect")
                    pubsub = None
                await asyncio.sleep(5)
        # Final cleanup
        if pubsub is not None:
            await self._close_pubsub_quietly(pubsub, where="shutdown")
