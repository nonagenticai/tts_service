# TTS Service

Multi-provider text-to-speech gateway that routes synthesis to local and cloud engines behind one REST surface. Port 6194.

|  |  |
|---|---|
| **Stack** | Python 3.14 · FastAPI · Pydantic v2 · SQLAlchemy (async) |
| **Database** | PostgreSQL 18 via psycopg 3 |
| **Deploy** | macOS **launchd** unit on an Apple Silicon host — **no k8s pod, no production Docker image** |
| **Mesh name** | `tts.test:9202` (config default port is `6194`) |

## What TTS Service Does

TTS Service accepts a block of text plus a provider or voice hint and returns spoken audio. A caller sends `POST /api/tts/generate` with the text and, optionally, a provider name, voice, output format, and whether it wants the call to block for the audio bytes or return a job ID for asynchronous polling. The service resolves a provider, synthesizes the audio, optionally post-processes it (WAV to MP3 conversion, concatenation), stores it in MinIO or on local disk, and returns either the raw audio, a streamed sequence of chunks, or a job record the caller can poll and download later.

Under the hood it is a **multi-provider gateway**. Five providers implement a common `TTSProvider` interface: **Kokoro-82M** and **KittenTTS** run locally and free (they need `torch` plus `espeak-ng`), **OpenAI TTS** and **ElevenLabs** are cloud engines reached over async `httpx` with per-provider API keys, and a **stub** provider emits silent 24 kHz PCM with no dependencies so CI and offline pipeline runs can exercise the whole path without a model or a network call. A routing layer picks a provider per request using a cost/quality/balanced strategy with fallback, or `TTS_PROVIDER=<name>` pins a single provider and disables the rest. Per-synthesis cost events are emitted to a Redis metering stream, and Redis also backs generation caching and WebSocket job notifications — all of which degrade gracefully if Redis is absent.

The service is stateful. It persists job lifecycle and usage-aggregation rows to PostgreSQL (`tts_jobs` and `tts_usage`) through psycopg 3 and async SQLAlchemy, and on boot it creates/verifies its tables via `Base.metadata.create_all` and resets any jobs left in `processing` from a previous crash. Requests are tenant-isolated by license token; the tenant ID is read off `request.state` and scopes every job and usage record. Async and batch jobs run behind concurrency semaphores, with FFmpeg concatenating multi-part batch output into a single file.

Two operational facts matter before you change anything here. First, this service deploys as a **macOS launchd unit**, not a Kubernetes Deployment — launchd runs `uvicorn` directly against an on-device venv, there is no deployed container, and dependency or schema changes reach the host only on the next on-device redeploy, never via Flux. Second, although this is the richer, better-tooled gateway, **it is not currently the live production narrate path**: the orchestrator's narrate stages call `tts.test:9202` `POST /api/tts/synthesize` (base64-WAV JSON), a route this repo does not expose, served instead by the separate `tts_backend` kokoro-onnx unit — a fix here does not reach that live path (see `CLAUDE.md`).

## Features

- **Multi-provider routing** — resolves each request across Kokoro, KittenTTS, OpenAI, and ElevenLabs by cost/quality/balanced strategy with fallback so callers are not locked to one engine and a failed provider retries against another
- **Local + cloud engines** — Kokoro-82M and KittenTTS run on-device for free while OpenAI TTS and ElevenLabs cover higher-quality cloud voices, so quality and cost can be traded off per request
- **Stub provider for CI** — a silent 24 kHz PCM provider with zero external deps lets the full pipeline (batch concat, downstream video services) run in CI and offline with no models, keys, or network
- **Single-provider override** — `TTS_PROVIDER=<name>` registers only the named provider and force-disables the rest, giving deterministic offline/CI smoke runs
- **Sync and async generation** — `/api/tts/generate` returns audio bytes inline or a job ID for polling, so short prompts stay simple while long jobs don't block the caller
- **Streaming output** — Server-Sent Events (`/stream`) and chunked binary (`/stream/raw`) deliver audio progressively for low-latency playback
- **Batch synthesis with concatenation** — `/generate/batch` runs many segments asynchronously and optionally FFmpeg-concatenates them into one downloadable file
- **Tenant-scoped usage metering** — records per-tenant character counts and cost, exposes analytics/dashboard/export endpoints, and streams cost events to `metering_service`
- **Cost-aware quotas** — optional monthly budget and character-limit enforcement with Redis-cached quota lookups and usage alerts
- **WebSocket notifications** — `/api/tts/ws` pushes real-time job and batch completion events per tenant
- **Presigned audio delivery** — completed audio lands in MinIO (or local disk) and is served by presigned URL so large files never stream through the app pod
- **Graceful Redis degradation** — caching, notifications, and metering silently disable when Redis is unavailable; the service still boots and synthesizes

## Authentication

Auth is **enabled by default** (`AUTH_ENABLED=true`). The `AuthMiddleware` (`src/auth/middleware.py`) reads a license token from the `X-Tenant-Id: Bearer <license_token>` header — **not** `Authorization`, which is reserved for Keycloak JWTs from the SPA — introspects it against the licenses service, and caches the result for ~5 minutes (`AUTH_CACHE_TTL=300`). The validated tenant ID is placed on `request.state` and scopes every job and usage row. Auth is bypassed only when `AUTH_ENABLED=false` (development).

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, cached ~5 min (`cache_ttl=300`, `cache_maxsize=1000`) |
| 2 | Trusted network | Optional in-cluster synthetic identity, gated by `AUTH_TRUST_INCLUSTER_IDENTITY` (off — the license-token path is the load-bearing one) |

Public paths (no token required): `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics`, plus anything under `/docs`, `/redoc`, and `/health/`.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6194
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### TTS `/api/tts`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Synthesize speech — returns audio bytes (sync) or a job ID (async) |
| POST | `/generate/batch` | Synthesize many segments asynchronously, optional FFmpeg concatenation |
| POST | `/stream` | Stream audio as Server-Sent Events |
| POST | `/stream/raw` | Stream audio as chunked binary |
| GET | `/providers` | List registered providers |
| GET | `/providers/{provider}/voices` | List available voices for a provider |
| GET | `/jobs/{job_id}` | Get job status |
| GET | `/jobs/{job_id}/audio` | Download completed job audio |
| GET | `/jobs/{job_id}/audio/url` | Get a presigned URL for the job audio |
| GET | `/batch/{batch_id}/audio` | Download concatenated batch audio |

### Usage `/api/tts/usage`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Per-tenant usage summary |
| GET | `/costs` | Cost breakdown by provider |
| GET | `/dashboard` | Aggregated usage dashboard |
| GET | `/export` | Export usage records |
| GET | `/quota` | Get tenant quota |
| PUT | `/quota` | Update tenant quota |
| GET | `/alerts` | List usage alerts |
| POST | `/alerts/{alert_id}/acknowledge` | Acknowledge a usage alert |
| GET | `/rates` | Get per-provider cost rates |

### WebSocket `/api/tts`

| Path | Description |
|------|-------------|
| `/ws` | Real-time job and batch completion notifications |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/healthz` | Liveness alias |
| GET | `/readyz` | Readiness alias |

## Configuration

All configuration via environment variables. Defaults come from `src/config.py` and the shared `w2v_common` settings base.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6194` | Listen port (`8000` is rejected at validation) |
| `API_DEBUG` | `false` | Enable /docs and /redoc debug behavior |
| `API_RELOAD` | `false` | Auto-reload on code changes |
| `AUTH_ENABLED` | `true` | Toggle authentication middleware |
| `AUTH_CACHE_TTL` | `300` | License introspection cache TTL (seconds) |
| `AUTH_INTROSPECT_TIMEOUT` | `2.0` | License introspection timeout (seconds) |
| `DATABASE_URL` | — (required) | PostgreSQL connection URL (psycopg 3) |
| `DATABASE_POOL_SIZE` | `10` | Connection pool size |
| `REDIS_URL` | — (required) | Redis URL for cache/notifications (optional at boot) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON_LOGS` | `false` | Structured JSON logging |
| `TTS_PROVIDER` | — | Single-provider override; registers only this provider |
| `KOKORO_ENABLED` | `true` | Enable Kokoro provider (local) |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Default Kokoro voice |
| `KITTEN_ENABLED` | `true` | Enable KittenTTS provider (local) |
| `KITTEN_MODEL_ID` | `KittenML/kitten-tts-mini-0.8` | KittenTTS model ID |
| `KITTEN_DEFAULT_VOICE` | `Bella` | Default KittenTTS voice |
| `OPENAI_TTS_ENABLED` | `true` | Enable OpenAI TTS provider (cloud) |
| `OPENAI_TTS_API_KEY` | `""` | OpenAI API key |
| `OPENAI_TTS_MODEL` | `tts-1-hd` | OpenAI TTS model |
| `OPENAI_TTS_DEFAULT_VOICE` | `alloy` | Default OpenAI voice |
| `ELEVENLABS_ENABLED` | `true` | Enable ElevenLabs provider (cloud) |
| `ELEVENLABS_API_KEY` | `""` | ElevenLabs API key |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | ElevenLabs model ID |
| `ELEVENLABS_DEFAULT_VOICE` | `rachel` | Default ElevenLabs voice |
| `STUB_ENABLED` | `false` | Register the silent stub provider alongside others |
| `ROUTING_ENABLED` | `true` | Enable auto-routing across providers |
| `ROUTING_DEFAULT_STRATEGY` | `balanced` | Routing strategy (cost/quality/balanced) |
| `STORAGE_BACKEND` | `local` | Audio storage backend (`local` or `minio`) |
| `STORAGE_LOCAL_PATH` | `/app/data/audio` | Local audio storage path |
| `STORAGE_MINIO_BUCKET` | `tts-audio` | MinIO bucket for audio |
| `CACHE_DEFAULT_TTL` | `86400` | Generation cache TTL (24h) |
| `METERING_REDIS_URL` | `""` | Separate Redis for cost-event stream (empty = disabled) |
| `QUOTA_ENFORCEMENT_ENABLED` | `false` | Enable monthly budget/character-limit enforcement |
| `WS_MAX_CONNECTIONS_PER_TENANT` | `10` | Max WebSocket connections per tenant |
| `WS_HEARTBEAT_INTERVAL` | `30` | WebSocket server ping interval (seconds) |
| `CONCURRENCY_MAX_CONCURRENT_GENERATIONS` | `10` | Global generation semaphore |

## Architecture

```
src/
├── asgi.py                          # FastAPI app, lifespan, health/root, middleware, OpenAPI
├── config.py                        # Pydantic settings (port 6194, provider sub-configs)
├── logging_config.py                # Structured logging setup
├── api/
│   ├── tts.py                       # /api/tts — generate, stream, batch, providers, jobs
│   ├── usage.py                     # /api/tts/usage — analytics, cost, quota, alerts, rates
│   ├── ws.py                        # /api/tts/ws — WebSocket job notifications
│   ├── health.py                    # /health, /healthz, /readyz probes
│   └── schemas.py                   # Request/response DTOs
├── auth/
│   ├── middleware.py                # License-token auth (X-Tenant-Id: Bearer)
│   └── models.py                    # Auth/identity models
├── providers/
│   ├── base.py                      # TTSProvider ABC (generate/stream/list_voices)
│   ├── registry.py                  # Provider registration + single-provider override
│   ├── kokoro_provider.py           # Kokoro-82M local provider
│   ├── kitten_provider.py           # KittenTTS local provider
│   ├── openai_provider.py           # OpenAI TTS cloud provider
│   ├── elevenlabs_provider.py       # ElevenLabs cloud provider
│   └── stub_provider.py             # Silent 24 kHz PCM provider for CI/offline
├── services/
│   ├── tts_service.py               # Core synthesis orchestration
│   ├── router.py                    # Provider routing + cost estimation
│   ├── job_manager.py               # Async/batch job lifecycle
│   ├── storage_service.py           # MinIO/local audio storage
│   ├── notification_service.py      # Redis pub/sub job events
│   ├── connection_manager.py        # WebSocket connection pool
│   ├── metering.py                  # Cost-event Redis stream emitter
│   ├── quota_service.py             # Budget/character quota enforcement
│   ├── language_detector.py         # Language detection (lingua)
│   ├── ssml_processor.py            # SSML parsing/normalization
│   └── protocols.py                 # Service protocol/type definitions
├── database/
│   ├── connection.py                # Async engine (psycopg 3)
│   ├── session.py                   # Session factory
│   ├── models.py                    # TTSJob / TTSUsage ORM models
│   └── migrations/                  # Alembic env + versions/ (001–004)
├── cache/
│   └── redis_client.py              # Redis client init/teardown
├── middleware/
│   └── request_id.py                # Request-ID ContextVar middleware
└── utils/
    ├── audio.py                     # Audio format helpers
    ├── post_processor.py            # WAV→MP3 / FFmpeg concat
    └── retry.py                     # Retry helpers
```

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest tests/integration/ -v        # Integration tests
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Coverage threshold: 80% (`fail_under = 80` in pyproject.toml). Vendor-dependent provider modules (`kitten_provider.py`, `kokoro_provider.py`) are excluded from coverage since their native libs aren't installed in every environment. Run the full local gate — which mirrors CI step-for-step — before pushing:

```bash
./scripts/ci-local.sh
# uv sync → ruff check → ruff format --check → ty check → pytest+cov →
# detect-secrets → pip-audit → bandit
```

## CI/CD

This is a **macOS launchd service**, not a k8s deployment. `.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners.

CI runs lint + ruff format + `ty` + pytest with coverage + detect-secrets + pip-audit + bandit, exercising the same on-device dependency surface production uses (kokoro / kittentts / espeak-ng / ffmpeg / torch wheels). It does **not** build or push a Docker image and does **not** Flux-deploy — updates reach the running unit only on the next **on-device redeploy** (host deploy script + launchd plist), not via cluster rollout. `torch` CVE-2025-3000 has no upstream fix and is owner-ignored in the workflow's `pip_audit_ignores` (the launchd archetype audits the live venv with bare pip-audit, so the ignore lives in `ci.yaml`).

## Docker

**Local dev only.** The `Dockerfile*` / `docker-compose*` files exist for local containerized smoke testing and are **not** the deployment path — production runs under launchd against the on-device venv.

```bash
docker compose up -d
curl http://localhost:6194/health
```

## Dependencies

- **licenses service** — validates license tokens via introspection (auth)
- **metering_service** — consumes the per-synthesis cost-event Redis stream
- **PostgreSQL 18** — job/usage persistence via psycopg 3 + async SQLAlchemy
- **Redis** — generation cache, WebSocket notifications, quota cache, cost-event stream (optional at boot)
- **MinIO** — audio object storage with presigned URL delivery
- **OpenAI TTS API** — cloud provider (API key)
- **ElevenLabs API** — cloud provider (API key)
- **kokoro** + **torch** — local Kokoro-82M synthesis (optional extra; needs espeak-ng)
- **kittentts** — local KittenTTS synthesis (optional extra)
- **ffmpeg** / **espeak-ng** — native deps for concat/format conversion and Kokoro phonemes (on-device)
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2** / **pydantic-settings** — validation and settings
- **SQLAlchemy** (async) + **psycopg 3** — database layer
- **Alembic** — schema migrations
- **httpx[http2]** — async HTTP client for cloud providers
- **miniopy-async** — async MinIO client
- **lingua-language-detector** — language detection
- **cachetools** — in-process caching
- **defusedxml** — safe SSML/XML parsing
- **structlog** — structured logging
- **w2v-common** (>=0.25.0) — shared config, auth middleware, ASGI helpers, error envelope

## Known Issues

- **Port 6194 (config) vs 9202 (mesh)** — `src/config.py` defaults `API_PORT=6194`, but the live launchd unit is reached at `tts.test:9202`. The gap is real (config default vs the unit's actual listen address), not a docs lag — see `CLAUDE.md`.
- **Not the live production narrate path** — the orchestrator's narrate stages call `tts.test:9202` `POST /api/tts/synthesize` (base64-WAV JSON), which this repo does **not** expose (it serves `/api/tts/generate`). That route is served by the separate `tts_backend` kokoro-onnx unit (`nonagenticai/tts_backend`, no GitHub CI). A fix here does **not** reach the live path — edit `tts_backend` for that.
- **launchd delivery, not Flux** — dependency and schema changes reach the host only on the next on-device redeploy. There is no initContainer / `alembic upgrade head` step; boot uses `Base.metadata.create_all` and resets orphaned `processing` jobs.
- **`torch` CVE-2025-3000** — no upstream fix; owner-approved and ignored in `ci.yaml` `pip_audit_ignores`.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. It runs with no licence switch set; unset means off.
