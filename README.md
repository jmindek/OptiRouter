# Discount Model Router

Routes AI requests only to discounted OpenRouter models. Auto-switches to the next best discounted model when the current one falls off the discount list.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your API keys

docker compose up --build
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENROUTER_API_KEY` | | OpenRouter API key for forwarding requests |
| `ROUTER_MODEL` | `typesafe/jev-1.13` | System One decision model (called via OpenRouter Decisions API) |
| `ROUTER_API_KEY` | | API key for the routing model |
| `ROUTER_API_BASE` | `https://openrouter.ai/api/v1` | API base URL for the routing model |
| `ROUTER_INVENTORY_URL` | `https://openrouter.ai/api/v1` | OpenRouter API URL for fetching discounted models |
| `ROUTER_MODE` | `auto` | Routing mode: cloud, local, auto |
| `ROUTER_OMLX_BASE` | `http://127.0.0.1:8000` | Local oMLX fallback base URL |
| `ROUTER_MIN_DISCOUNT` | `0.50` | Minimum discount threshold (0.0–1.0) |
| `ROUTER_MIN_CONFIDENCE` | `0.30` | Minimum decision-model confidence to accept a pick; below it the router uses the best available model |
| `ROUTER_PREFERENCE` | `cost` | Tie-breaker handed to the decision model: `cost`, `quality`, or `none` |
| `ROUTER_FALLBACK_MODE` | `static` | `static` = auto-fallback to `ROUTER_STATIC_FALLBACK`; `halt` = return 503 |
| `ROUTER_STATIC_FALLBACK` | `openrouter/deepseek/deepseek-v3.1` | Fallback model when routing fails |
| `ROUTER_MODEL_HANDOFF` | `next_best` | Behavior when selected model falls off discount list: `stop`, `next_best`, `keep_current` |
| `ROUTER_CACHE_FILE` | `data/models.json` | Path to cached discounted models |
| `ROUTER_REFRESH_CRON` | `0 * * * *` | Cron schedule for inventory refresh (default: hourly) |
| `LANGFUSE_HOST` | | Langfuse base URL (e.g. `http://localhost:3000`). Set with the two keys below to send traces to Langfuse; otherwise traces print to console |
| `ROUTER_TRACE_INPUT` | `truncated` | What a trace records as input: `none`, `decision`, `truncated`, or `full` |
| `ROUTER_TRACE_INPUT_CHARS` | `500` | Character cap when `ROUTER_TRACE_INPUT=truncated` |
| `ROUTER_TRACE_ROUTING_DETAIL` | `true` | Record the routing decision (model, confidence, reason) as trace output |
| `LANGFUSE_PUBLIC_KEY` | | Langfuse public key (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | | Langfuse secret key (`sk-lf-...`) |

## Model Handoff

When the selected model falls off the discount list, the router handles it based on `ROUTER_MODEL_HANDOFF`:

- **`stop`** (default: `next_best`) — Returns 400 with a message telling the user the model is no longer discounted and they need to pick a new one.
- **`next_best`** — Automatically switches to the best available discounted model.
- **`keep_current`** — Continues using the model even though it's no longer discounted.

## Telemetry

The router emits OpenTelemetry traces and metrics. **Traces go to Langfuse when
it is configured, otherwise to the console.** Metrics always go to the console.

### Langfuse

Set all three of `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, and
`LANGFUSE_SECRET_KEY` and traces are POSTed to Langfuse's ingestion API.

This uses a **native Langfuse exporter**, not OTLP. Langfuse v2 has no OTLP
endpoint: `POST /api/public/otel` returns 404 and nothing listens on gRPC port
4317, so OTLP cannot be delivered to it. Instead, spans are translated to
Langfuse's own batch format (`trace-create` + `span-create` events) and sent to
`{LANGFUSE_HOST}/api/public/ingestion` with basic auth. See
`src/langfuse_exporter.py`.

Delivery failures are logged and never break a request.

#### What gets recorded

Prompts are user content, so trace content is configurable:

| `ROUTER_TRACE_INPUT` | trace `input` |
|---|---|
| `none` | nothing |
| `decision` | nothing (routing decision only) |
| `truncated` **(default)** | first `ROUTER_TRACE_INPUT_CHARS` characters of the prompt |
| `full` | the whole prompt |

Trace `output` carries the routing decision (`model=... reason=... confidence=...`)
when `ROUTER_TRACE_ROUTING_DETAIL` is true.

Background spans such as `inventory_refresh` have no prompt and record no
input or output — that is correct, not a gap.

### Metrics

| Name | Type | Description |
|------|------|-------------|
| `router.request.duration` | Histogram (ms) | Request latency |
| `router.model.selections` | Counter | Model selections, labeled by model ID |
| `router.rate_limits` | Counter | Rate limit (429) events, labeled by model |
| `router.inventory.refreshes` | Counter | Inventory refresh attempts |
| `router.inventory.errors` | Counter | Inventory refresh failures |
| `router.handoffs` | Counter | Model handoff events, labeled by reason |
| `router.errors.total` | Counter | Total errors, labeled by type |

### Traces

- `chat_completions` — Main request flow, labeled with selected model
- `inventory_refresh` — Inventory refresh job

## Architecture

```
Client → /v1/chat/completions → classify_request() → select discounted model
                                              ↓
                                    decisions API (Jev) picks candidate
                                      ↙          ↘
                                 confident    low confidence
                                    ↓              ↓
                              forward()      best available model
```

- **Inventory**: Fetches discounted models from OpenRouter `/models` and `/models/{id}/endpoints` endpoints concurrently (semaphore 10). Cached to `data/models.json`. Refreshed on cron schedule.
- **Router**: Sends the request transcript verbatim to the System One decision model (`typesafe/jev-1.13` via `POST /api/alpha/decisions`). Candidates are filtered to models that can serve the request (image input, tools, context), then capped to a small blended pool. Accepts the pick only when confidence ≥ `ROUTER_MIN_CONFIDENCE`.
- **Candidate selection**: `_rank_candidates()` blends benchmark rank and discount rank so the capped pool spans cheap-weak to strong-expensive. Ranking by discount alone gives five cheap near-twins; by score alone, five expensive near-twins; either way the decision model has no tradeoff to reason about.
- **Request requirements**: `_request_requirements()` derives hard constraints from the transcript — image parts, tool calls, and an estimated context floor — and `_filter_candidates()` drops models that cannot satisfy them. Oversized transcripts are truncated from the middle before being sent; the estimate uses the untruncated text.
- **Rate Limiter**: Reactive — tracks 429 responses and backs off per-model.
- **Forwarding**: Proxies to OpenRouter with streaming support and retry logic.

## Development

```bash
uv sync
uv run pytest -v
uv run pyright src/
```

### Tuning notes

Live testing against the real discounted pool informed the defaults:

- **`MAX_CANDIDATES=5`.** Pool size is the strongest lever on confidence.
  Measured, no tie-breaker, same coding task:

  | candidates | confidence | tokens in |
  |---:|---:|---:|
  | 3 | 0.06 | 678 |
  | **5** | **0.53** | **893** |
  | 8 | 0.35 | 1261 |
  | 21 | 0.10 | 2724 |

  With 21 near-identical text models the decision model spreads probability
  thinly and lands near 0.10. With 5 it is three times more confident and
  uses a third of the tokens. Below 5 it is unsure the right option is even
  present — the curve is an inverted U with the peak at 5.
- **Blended pool selection.** Keeping both cheap and strong candidates
  visible lifts confidence further: ~0.80 for a coding task, against ~0.52
  when the pool is picked by discount alone and ~0.38 by score alone. A pool
  whose members genuinely differ gives the decision model something to weigh.
- **`ROUTER_MIN_CONFIDENCE=0.30`.** Confidence for model selection sits
  roughly 0.30–0.55 with a good pool. A 0.55 bar abstained often; 0.30 leaves
  room for the crossing cases while still rejecting the near-coin-flip
  answers a bad pool produces.
- **`ROUTER_PREFERENCE=cost`.** Of the tie-breakers tested, only cost raised
  confidence (0.30 → 0.35 on a large pool). Open-weight and closed-weight
  preferences *lowered* it, and a reasoning preference lowered it and was
  ignored. A detailed judging rubric lowered it too. A clean question with one
  real discriminator beats a long list of constraints.
- **Verbatim transcript, no keyword tag.** An earlier version derived a task
  label (`coding`, `writing`, ...) from keywords and sent that instead of the
  prompt. It removed the signal the decision needs. Sending the real
  transcript routes image requests to image-capable models, which a keyword
  tag cannot do.

Capability hints matter: each candidate is described with its name, provider
summary, modalities, features (tool calling / reasoning / structured output),
context length, benchmark scores (intelligence / coding / agentic), discount,
and effective input price. Non-chat models (audio/video/embedding/placeholder
rows) are dropped from the inventory before routing, because the scraper sees
discount badges on them too.

### Verifying the decision model

The router calls the OpenRouter Decisions API (`POST /api/alpha/decisions`) with
`{model, state, questions}` and reads a typed `answers` block. The contract is
not OpenAI chat completions — a decision model is not a chat model.

`scripts/probe_decisions.py` sends one real request and prints the exact
request and response, so you can confirm the shape (and your key/model) after
any provider change:

```bash
uv run python scripts/probe_decisions.py
```

Two things worth knowing from the live response:

- `confidence` is not the top probability. For `answers.target`, `probabilities`
  is the spread over all options and `confidence` summarises that whole
  distribution. The router gates on `confidence`, not on the largest probability.
- The response `model` reports the resolved version (e.g.
  `typesafe/jev-1.13-20260917`), not the alias you sent. Use it for attribution.

