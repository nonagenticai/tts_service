# Operation ID Plan: tts_service

Canonical registry of `operation_id` values for every route in `tts_service`.
These IDs feed the org-wide n8n custom-node generator and the `ultrareview`
cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (nlp_service,
  worker_management, kag, analysis_service, video_compiler_service,
  video_analyzer_service),
- **globally unique** — disambiguated against the rest of the fleet's verbs.

This service hosts a single TTS domain (generation, streaming, batching, jobs,
providers/voices) plus usage analytics and a health surface. Generic verbs
(`generate`, `stream`, `list`, `download`, `get`, `update`, `acknowledge`,
`export`) are prefixed with `tts_` / `speech_` where collision is plausible
across the fleet.

## Files to edit

- `src/asgi.py` (root endpoint)
- `src/api/health.py`
- `src/api/tts.py`
- `src/api/usage.py`
- `src/api/ws.py` (WebSocket — name only; not in OpenAPI schema)

## Mapping Tables

### `src/asgi.py` (tag: `health`)

| Method | Function | operation_id   | Rationale                                  |
|--------|----------|----------------|--------------------------------------------|
| GET    | `root`   | `service_root` | Service banner; matches the fleet standard. |

### `src/api/health.py` (tag: `health`)

| Method | Function       | operation_id   | Rationale                                              |
|--------|----------------|----------------|--------------------------------------------------------|
| GET    | `health_check` | `health_check` | Primary health probe; fleet standard verb.             |
| GET    | `health_live`  | `health_live`  | Liveness sub-probe; one-to-one with path.              |
| GET    | `health_ready` | `health_ready` | Readiness sub-probe; one-to-one with path.             |

### `src/api/tts.py` (tag: `tts`)

| Method | Function                  | operation_id                  | Rationale                                                                                              |
|--------|---------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------|
| POST   | `generate`                | `generate_speech`             | Domain-suffix `speech` disambiguates from `generate_*` in nlp_service / image_gen_backend / etc.       |
| POST   | `stream_generate`         | `stream_speech_sse`           | SSE variant; transport suffix disambiguates from raw streaming below.                                  |
| POST   | `stream_generate_raw`     | `stream_speech_raw`           | Chunked-binary variant; pairs with `stream_speech_sse`.                                                |
| POST   | `generate_batch`          | `generate_speech_batch`       | Batched variant of `generate_speech`; `_batch` suffix.                                                 |
| GET    | `download_batch_audio`    | `download_speech_batch_audio` | Generic `download_*` would collide; prefix with `speech_batch`.                                        |
| GET    | `list_providers`          | `list_tts_providers`          | `list_providers` is too generic at fleet scope (worker_management, inference-router also list providers). |
| GET    | `list_voices`             | `list_tts_voices`             | Same — domain prefix.                                                                                  |
| GET    | `get_job_status`          | `get_tts_job_status`          | Domain-prefix to disambiguate from `get_job_status` in analysis_service / worker_management / etc.     |
| GET    | `download_audio`          | `download_tts_audio`          | Generic `download_audio` would collide; domain prefix.                                                 |
| GET    | `get_audio_presigned_url` | `get_tts_audio_url`           | Domain prefix + verb_noun shape; URL kind (presigned vs direct) is body-not-id concern.                |

### `src/api/usage.py` (tag: `usage`)

| Method | Function             | operation_id                  | Rationale                                                                                 |
|--------|----------------------|-------------------------------|-------------------------------------------------------------------------------------------|
| GET    | `get_usage`          | `get_tts_usage_summary`       | Generic `get_usage` would collide with worker_management / nlp_service usage routes.      |
| GET    | `get_cost_breakdown` | `get_tts_cost_breakdown`      | Same — domain prefix.                                                                     |
| GET    | `get_dashboard`      | `get_tts_dashboard`           | `get_dashboard` is fleet-generic.                                                         |
| GET    | `export_usage`       | `export_tts_usage`            | Generic `export_*` collides with analytics/observability exports across services.         |
| GET    | `get_quota`          | `get_tts_quota`               | Many services expose quota; domain-prefix.                                                |
| PUT    | `update_quota`       | `update_tts_quota`            | Same — domain prefix.                                                                     |
| GET    | `get_alerts`         | `list_tts_usage_alerts`       | Reads as `list_*` per fleet convention for collection endpoints; domain prefix.           |
| POST   | `acknowledge_alert`  | `acknowledge_tts_usage_alert` | Generic `acknowledge_alert` will collide with future alerting endpoints elsewhere.        |
| GET    | `get_cost_rates`     | `get_tts_cost_rates`          | Domain prefix.                                                                            |

### `src/api/ws.py` (tag: `tts-ws`)

WebSocket routes do not produce `operationId` entries in the OpenAPI HTTP
schema (FastAPI excludes them). The route is still named for client codegen
and the n8n generator, via the `name=` kwarg on `@router.websocket`.

| Method | Function                       | name                     | Rationale                                       |
|--------|--------------------------------|--------------------------|-------------------------------------------------|
| WS     | `websocket_job_notifications`  | `tts_job_notifications`  | Domain-prefixed name; matches fleet WS naming.  |

## Verification

```bash
cd tts_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the union of every right-hand column above
(excluding the WebSocket row, which never appears in the OpenAPI HTTP schema).
Drift (rename, missed decorator, fallback) fails the test loudly.

## Disambiguation rationale

Concrete renames where the function name was generic enough to collide
across the fleet or within this service:

- `generate` → `generate_speech`; `generate_batch` → `generate_speech_batch`.
  Multiple services in the fleet have `generate_*` endpoints; domain noun
  pins this to the TTS surface.
- `stream_generate` / `stream_generate_raw` → `stream_speech_sse` /
  `stream_speech_raw`. Transport suffix keeps the two variants apart.
- `list_providers` / `list_voices` → `list_tts_providers` / `list_tts_voices`.
  Provider/voice listings exist in several services; domain prefix.
- `get_job_status` → `get_tts_job_status`; `download_audio` →
  `download_tts_audio`; `download_batch_audio` → `download_speech_batch_audio`.
  Generic job/audio verbs collide; domain prefix.
- `get_audio_presigned_url` → `get_tts_audio_url`. Shortened to the verb_noun
  shape; "presigned vs direct" is a backend implementation detail.
- All `/api/tts/usage/*` ops → `*_tts_*` prefix. Multiple services expose
  usage/cost/quota/alerts surfaces; the TTS prefix keeps them disjoint.
- `get_alerts` → `list_tts_usage_alerts`. Reads as `list_*` (collection
  endpoint) per fleet convention.

## Future routes

When adding a route in this service, **always** set `operation_id="…"`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
