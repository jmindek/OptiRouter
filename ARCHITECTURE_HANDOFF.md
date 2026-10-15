# Architecture & Execution Handoff

## Project Overview

Discount Model Router — routes AI inference requests to OpenRouter models that have active discounts (≥25% by default). Uses a decision model (`typesafe/jev-1.13` via OpenRouter Decisions API, but pluggable) to pick the best discounted candidate for each request guided by user-configurable feature weights.

**Target API**: OpenAI-compatible (`/v1/chat/completions`). Compatible with OpenRouter.

---

## Architecture

```
Client → FastAPI (port 8080) → Router Logic → OpenRouter API
                              │
                              ├── Load discounted pool from cache
                              ├── Derive request requirements (image / tools / context)
                              ├── Filter candidates (hard requirements only)
                              ├── Ask decision model to pick (Decisions API)
                              ├── Forward request (with rate limiting)
                              └── Return response (stream or non-stream)
                              │
                              └── Inventory refresh (cron, hourly at :00)
                                  ├── GET /api/v1/models (all models, metadata)
                                  └── GET /api/v1/models/{id}/endpoints (discount info, per-endpoint)
```

### Key Insight: Discount Data Source

The `/models` endpoint does NOT include discount info. Discount is only in `/models/{id}/endpoints`, so the whole catalogue is swept (one call per model) to find discounted ones. Measured live: all 462 models in ~4s at concurrency 10, with no rate limiting.

### Key Insight: Discounts Are Per-Endpoint

A model may have 40+ providers serving it on OpenRouter, each with its own pricing and discount. `endpoints[0]` may show 0% while another endpoint has 50%+, so the router scans all endpoints and picks a trustworthy one (uptime ≥ `ROUTER_MIN_ENDPOINT_UPTIME`, request count ≥ `ROUTER_MIN_ENDPOINT_REQUESTS`) with the largest discount.

### Key Insight: The Decision Model Is Not a Chat Model

`typesafe/jev-1.13` does not generate text. `/v1/chat/completions` against it returns 400. It is called through the OpenRouter Decisions API with `state` plus typed `questions`, and returns typed `answers` with a probability distribution and a calibrated `confidence`. Note `confidence` is NOT the top probability — it summarises the whole distribution. See README for the full contract.

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.py` | 388 | FastAPI app, lifespan, routing endpoint, forward/stream logic, healthz |
| `src/router.py` | 400 | Request requirements, candidate filter, decision-model call, response parsing |
| `src/inventory.py` | 291 | Discount sweep, non-chat filtering, cache save/load, client injection |
| `src/config.py` | 122 | Pydantic settings, env var mapping |
| `src/models.py` | 56 | Pydantic models: `ChatCompletionRequest`, `ModelInfo`, `ChatMessage` |
| `src/handoff.py` | 36 | Handoff when a chosen model falls off the discount list |
| `src/rate_limiter.py` | 48 | Reactive rate limiting (429 handling, backoff tracking) |
| `src/telemetry.py` | 84 | OpenTelemetry metrics and traces, exporter wiring |
| `src/langfuse_exporter.py` | 174 | SpanProcessor translating spans to Langfuse's native ingestion format |
| `tests/test_langfuse_exporter.py` | 241 | Langfuse export: translation, auth, never-raise |
| `tests/test_router.py` | 390 | Router: requirements, filtering, decision parsing |
| `tests/test_inventory_sweep.py` | 183 | Sweep integration, non-chat filtering, cache fallback |
| `tests/test_regression_p0.py` | 326 | Regression tests for P0 fixes |
| `tests/test_model_handoff.py` | 65 | Handoff modes |
| `tests/test_inventory.py` | 37 | Cache save/load |
| `tests/test_rate_limit_policy.py` | 54 | Shared 429 policy (both transports) |
| `tests/test_rate_limiter.py` | 28 | Rate limiter |
| `tests/test_models.py` | 23 | Model validation |
| `tests/conftest.py` | 19 | Test path setup |
| `.env.example` | 79 | Environment variable template |
| `scripts/probe_decisions.py` | — | Live probe of the Decisions API (manual verification) |
| `Dockerfile` | 21 | Container build |
| `docker-compose.yml` | 10 | Docker compose (router only) |
| `pyproject.toml` | 29 | Project config, dependencies |

No Playwright: the discount sweep replaced the browser scraper (`src/scraper.py` was deleted). `playwright` is no longer a dependency.

---

## Key Components

### 1. Inventory (`src/inventory.py`)
- Sweeps `/models` then every `/models/{id}/endpoints` concurrently (`ENDPOINT_CONCURRENCY = 10`) to find discounted models.
- `_is_chat_capable()` drops models that cannot serve a chat completion: no text output (audio/video/image/embedding) or no context window (placeholder rows such as `docs/quickstart`).
- `_meta_from_api()` captures capability hints: provider, name, description, input/output modalities, supported parameters, and `artificial_analysis` benchmark scores (`intelligence_index`, `coding_index`, `agentic_index`).
- Filters by `min_discount` threshold (default 25%).
- Accepts optional `client` parameter for dependency injection in tests.
- Falls back to the cached inventory when the sweep fails or finds nothing.
- Saves to `data/models.json`. Refreshed hourly by default (see `ROUTER_REFRESH_CRON`).

### 2. Router (`src/router.py`)
- `_request_requirements()`: derives hard requirements from the transcript — image parts, tool calls, and an estimated context floor (`_estimate_tokens`, measured on the untruncated text).
- `_filter_candidates()`: keeps only models satisfying those requirements; falls back to the full pool rather than returning nothing.
- `_build_criteria()`: builds the `choice` question criteria — one option per candidate, all eligible models are included (no cap).
- `_build_state()`: state sent to the decision model contains only weights and preference — no request transcript, no computed values, no candidate ranking.
- `classify_request()`: posts `{model, state, questions}` to `POST /api/alpha/decisions` and returns `RoutingDecision(model_id, confidence, reason)`.
- `_parse_decision()`: validates the typed answer; a missing or malformed answer is an error, never approval. Abstains below `ROUTER_MIN_CONFIDENCE`.

### 3. Main App (`src/main.py`)
- FastAPI app with `/v1/chat/completions`.
- `forward_to_openrouter()`: proxies to OpenRouter with reactive rate limiting, preserving all request fields (temperature, tools, logprobs, ...) and passing the non-stream response through unchanged.
- On a `low_confidence` decision, uses the best available discounted model instead of the static fallback.
- `/healthz` reports cache age and model count.
- Fallback: static model or 503 when no models are available.

### 4. Handoff (`src/handoff.py`)
- `model_in_inventory()` / `handle_model_handoff()`: when the chosen model is no longer discounted, `stop` (400), `next_best` (switch), or `keep_current` (proceed). Fires on the request after the pool refresh drops the model.

### 5. Rate Limiter (`src/rate_limiter.py`)
- Reactive: only activates on 429 responses.
- Backoff tracking per-model.
- Clears backoff state on successful request.

### 6. Config (`src/config.py`)
- Pydantic Settings with env var aliases.
- All vars prefixed with `ROUTER_` (except `OPENROUTER_API_KEY`).
- `min_discount` (default 0.25), `min_confidence` (default 0.30), `min_context_length` (default 131072), `min_endpoint_uptime` (default 95.0), `min_endpoint_requests` (default 100), `weights` (default: discount=0.4, latency=0.2, uptime=0.2, coding_index=0.2, intelligence_index=0.0, agentic_index=0.0), `preference` (cost/quality/none).

---

## Current State

### Tests
- **88 tests, all passing.** `uv run pytest`.
- Coverage: router (requirements, filtering, decision parsing), inventory sweep (non-chat filtering, cache fallback), cache save/load, model validation, rate limiter, handoff modes, P0 regressions, Langfuse exporter.
- `uv run pyright src/` reports 0 errors.

### Git History & Status
- **Active Branch**: `feature/router`
- **Recent work**:
  - Replaced a guessed decision-model integration with the live-verified OpenRouter Decisions API contract (decision model, not chat).
  - Removed the keyword task-tagger; the transcript is not sent to the decision model.
  - Added request-derived filtering (image/tools/context) — no shortlist, no ranking bias.
  - Removed the shortlist cap (`MAX_CANDIDATES`) and rank-sum shortlist (`_rank_candidates`).
  - Removed transcript bias (`_render_state`, `_truncate_middle`) — the decision model sees only weights and preference, not the user prompt.
  - Captured `artificial_analysis` benchmark scores into the candidate descriptions.
  - Replaced the Playwright scraper with a browserless `/endpoints` sweep; refresh is now hourly.
  - Forward all request fields and pass the non-stream response through unchanged.
  - `/healthz` reports cache age and model count.
  - Added per-endpoint discount scanning (discounts are per-endpoint, not per-model).
  - Added `ROUTER_MIN_CONTEXT_LENGTH` (default 128K) and `ROUTER_WEIGHTS` for configurable feature weighting.

---

## Known Issues & Backlog

### Resolved (P0)
1. **[FIXED] Closed HTTP Client in Inventory Refresh** — Scoped `httpx.AsyncClient` context around both `/models` and `/endpoints` loop.
2. **[FIXED] Broken Streaming Lifecycle** — `event_generator` now manages connection closing in a `finally` block upon completion.
3. **[FIXED] Base URL `/v1` Duplication & API Key Assignment** — Added `forward_to_openrouter()` helper to strip trailing slashes/v1 duplicates and forward to OpenRouter.

### Resolved (later iterations)
4. **Inventory refresh concurrency** — `asyncio.Semaphore(10)`; whole catalogue in ~4s.
5. **Non-streaming response wrapping** — response is passed through unchanged, preserving tool calls, logprobs, and the full OpenAI schema.
6. **Category keyword map** — removed. The decision model picks from the real candidate pool.
7. **Inventory health check** — `/healthz` reports cache age and model count.
8. **Request field passthrough** — temperature, max_tokens, tools, logprobs and the rest are forwarded via `model_dump(exclude_none=True)`.
9. **Playwright scraper** — deleted; the `/endpoints` sweep replaces it.
10. **Stream/non-stream duplication** — the 429 retry/backoff policy was duplicated across both transports; it now lives in `_apply_rate_limit_policy()` with a shared test.
11. **Shortlist cap and rank-sum bias** — removed `MAX_CANDIDATES` and `_rank_candidates`; all eligible candidates are sent to the decision model.
12. **Transcript bias** — removed `_render_state` and `_truncate_middle`; the decision model sees only weights and preference, not the user prompt.
13. **RateLimiter lock scope** — lock is released before `asyncio.sleep`, so a throttled model never blocks concurrent requests for unthrottled models.
14. **Dockerfile dependency list** — Dockerfile installs directly from `pyproject.toml`.

### Pending Issues & Open Questions

1. **`forward_to_openrouter` Stream/Non-Stream Duplication**
   - Stream and non-stream logic can be further consolidated.

2. **Validate-on-forward (deferred)** — the pool is up to 1h stale, so an in-flight model's discount can change unnoticed within that window. A per-model `/endpoints` check with a short TTL would catch it for ~1–2s/day. Deferred by choice; handoff covers the next-request case.

---

## Active Requirements for Next Iteration

### Medium Priority
1. **Decision model pluggability** — the decision model is now pluggable via config; local open-weight models are the next step.

---

## Run Commands

```bash
# Local development
uv run uvicorn src.main:app --reload --port 8080

# Tests
uv run pytest

# Type checking
uv run pyright

# Docker
docker compose up --build
```
