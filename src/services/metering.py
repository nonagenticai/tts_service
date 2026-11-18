"""Cost metering — tts_service routing identity over the shared cost-bus.

Stream resolution, the XADD field shape, and the Redis connection lifecycle
all live in :mod:`w2v_common.metering` (the single source of truth shared by
every w2v service). This module only owns the tts_service-specific routing
identity (``cost_source`` / ``service_key``) and the "a tenant is REQUIRED"
contract; it re-exports the shared helpers so existing importers keep working.

The cost-bus stream name now resolves centrally via the ``METERING_STREAM``
env var (falling back to the canonical ``billing:costs``) — see
``w2v_common.metering.resolve_stream``. There is no longer a per-service
``stream`` config field.
"""

from __future__ import annotations

import uuid

from w2v_common.metering import (
    Funding,
    close_metering_redis,
    cost_event_key,
    init_metering_redis,
    schedule_cost_event,
)

COST_SOURCE = "w2v-tts"
SERVICE_KEY = "w2v-tts"  # pragma: allowlist secret -- routing slug, not a credential

#: WHO FUNDS a synthesis, per provider (billing ``funding`` taxonomy). Self-hosted
#: kokoro/kitten run on platform hardware; openai/elevenlabs are external vendors
#: billed on the PLATFORM key (tts_service has no tenant-key path, so there is no
#: ``tenant_byo`` arm). ``stub`` is the in-process test provider (no vendor).
_PROVIDER_FUNDING: dict[str, Funding] = {
    "kokoro": "local_gpu",
    "kitten": "local_gpu",
    "stub": "local_gpu",
    "openai": "platform_external",
    "elevenlabs": "platform_external",
}


class UnknownFundingProviderError(ValueError):
    """A provider with no declared funding reached the cost bus.

    Fail loud: a guessed funding is stored verbatim by billing, so a new provider
    must be added to ``_PROVIDER_FUNDING`` deliberately.
    """


def funding_for_provider(provider: str) -> Funding:
    """The billing ``funding`` for a synthesis served by ``provider``."""
    try:
        return _PROVIDER_FUNDING[provider]
    except KeyError:
        raise UnknownFundingProviderError(
            f"tts provider {provider!r} has no declared funding (add it to _PROVIDER_FUNDING)"
        ) from None


def synthesis_event_key(call_kind: str, call_id: str | None) -> str | None:
    """Deterministic ``event_key`` for ONE synthesis, or ``None`` without a durable id.

    ``call_id`` is the async job id (``call_kind="job"``) or the inbound
    ``X-Request-Id`` (``"sync"`` / ``"stream"``). A re-emit of the same call
    (process restart, job redelivery) yields the same key, so billing dedupes it.
    With no id at all the cost bus mints a uuid7 and counts it on
    ``w2v_cost_event_key_source_total{source="minted"}`` -- visible, not silent.
    """
    if not call_id:
        return None
    return cost_event_key(SERVICE_KEY, call_kind, call_id)


# ── PRODUCT-OWNER RULING: a tenant_id may NEVER hold a sentinel ──────────────
# Do NOT re-add ``DEFAULT_TENANT_ID``, ``UNATTRIBUTED_TENANT_ID``, a
# ``DEFAULT_TENANT_ID`` env read, or ``config.auth.default_tenant_id`` here.
# Every message carries a REAL registered tenant forwarded from the requestor;
# an absent tenant is an ERROR (HTTP 401 at the route layer), never a
# substituted default.
#
# The nil UUID was the most dangerous spelling of the substitution: it is a
# VALID uuid, so it passes every uuid cast and dirty-row check, and billing then
# buckets it as unattributed and NEVER CHARGES it (billing-service
# ``_parsers.py`` + ``cost_flusher.py``). Real usage, silently unbilled.


class MissingTenantError(ValueError):
    """No real tenant was forwarded on a call that MUST be tenant-scoped.

    FAIL LOUD, never substitute. Raised by :func:`require_tenant` and mapped to
    HTTP 401 by the route helpers in :mod:`src.api.tts` / :mod:`src.api.usage`.
    """

    def __init__(self, field: str = "tenant_id") -> None:
        super().__init__(
            f"{field} is required and must be a real registered tenant forwarded by the "
            "caller — no sentinel, no default, no substitution. Nothing was recorded."
        )


class InvalidTenantIdError(ValueError):
    """A tenant WAS forwarded, but it is not a UUID.

    Distinct from :class:`MissingTenantError` on purpose: something DID
    authenticate, so 401 would be a lie — the value is simply unstorable in the
    ``uuid`` column (revision ``005_tenant_id_uuid``). Mapped to HTTP 400 by
    the route helpers in :mod:`src.api.tts` / :mod:`src.api.usage`, matching
    ``video_compiler_service``, which reached the uuid column first.

    The offending value is deliberately NOT echoed: ``X-Tenant-Id`` carries
    ``Bearer <license_token>`` on this service's own auth handshake, so a
    mis-plumbed caller can put a live credential in this field. Only the length
    is reported — enough to tell a slug from a truncated uuid from a token.
    """

    def __init__(self, field: str = "tenant_id", *, length: int) -> None:
        super().__init__(
            f"{field} must be a UUID — tts_service stores it as a postgres ``uuid`` column, "
            f"so a legacy slug (`n8n-dev`, `tenant-1`) or a truncated value cannot be "
            f"persisted. Got a non-UUID value of length {length}. Nothing was recorded."
        )


def require_tenant(tenant_id: object, field: str = "tenant_id") -> str:
    """Return the forwarded tenant as a CANONICAL uuid ``str``, or raise.

    ``tenant_id`` is a postgres ``uuid`` column on all four tenant-scoped
    tables (revision ``005_tenant_id_uuid``), so a non-uuid value is no longer
    "unusual but storable" — it is unstorable. This function is the boundary
    where that becomes a clean 4xx instead of a psycopg ``DataError`` surfacing
    as a 500 halfway through a job write.

    Two distinct failures, deliberately NOT collapsed into one:

    * **absent / blank / nil-uuid** → :class:`MissingTenantError` → HTTP 401.
      Authentication resolved no identity. The nil UUID is rejected BY VALUE in
      every spelling it can arrive in (dashed, undashed, braced, ``urn:``,
      upper/lower case) because it is a *valid* uuid that passes every cast and
      dirty-row check, and billing then buckets it as unattributed and NEVER
      CHARGES it.
    * **present but not a uuid** (a legacy slug like ``n8n-dev``, a truncated
      value) → :class:`InvalidTenantIdError` → HTTP 400. A bad argument, not a
      missing identity.

    The return value is ``str(uuid.UUID(...))`` — lowercase, dashed. That
    canonicalisation is load-bearing, not cosmetic: postgres renders a ``uuid``
    column back in exactly that form, so an inbound ``ABCDEF12-…`` must be
    folded here or every ``TTSJob.tenant_id == tenant_id`` comparison and every
    ``f"tenant:{tenant_id}"`` Redis channel key would silently miss.
    """
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise MissingTenantError(field)
    text = tenant_id.strip()
    try:
        parsed = uuid.UUID(text)
    except ValueError as exc:
        raise InvalidTenantIdError(field, length=len(text)) from exc
    if parsed.int == 0:
        raise MissingTenantError(field)
    return str(parsed)


__all__ = [
    "COST_SOURCE",
    "SERVICE_KEY",
    "InvalidTenantIdError",
    "MissingTenantError",
    "UnknownFundingProviderError",
    "close_metering_redis",
    "funding_for_provider",
    "init_metering_redis",
    "require_tenant",
    "schedule_cost_event",
    "synthesis_event_key",
]
