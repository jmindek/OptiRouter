# OptiRouter

[![Ko-fi](https://img.shields.io/badge/Ko--fi-FF5E5B?style=flat&logo=kofi&logoColor=white)](https://ko-fi.com/jerrymindek)

Weighted feature router for optimal inference — maximize whatever you value most.

I built this after noticing how many excellent models are discounted on OpenRouter — but they rotate so frequently that sitting on one model and expecting to pay that low price forever isn't realistic. OptiRouter scores each candidate against configurable weights (discount, cost, latency, throughput, uptime, coding index, etc.) and picks the best fit for every request. Provider-agnostic by design; OpenRouter is step 1, with support for any provider coming soon.

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
| `ROUTER_MODEL` | `typesafe/jev-1.13` | Decision model (called via OpenRouter Decisions API, pluggable) |
| `ROUTER_API_KEY` | | API key for the decision model |
| `ROUTER_API_BASE` | `https://openrouter.ai/api/v1` | API base URL for the decision model |
| `ROUTER_INVENTORY_URL` | `https://openrouter.ai/api/v1` | OpenRouter API URL for fetching discounted models |
| `ROUTER_MODE` | `auto` | Routing mode: cloud, local, auto |
| `ROUTER_OMLX_BASE` | `http://127.0.0.1:8000` | Local oMLX fallback base URL |
| `ROUTER_MIN_DISCOUNT` | `0.25` | Minimum discount threshold (0.0–1.0) |
| `ROUTER_MIN_CONFIDENCE` | `0.30` | Minimum decision-model confidence to accept a pick; below it the router picks by `ROUTER_LOW_CONFIDENCE_CHOICE` |
| `ROUTER_LOW_CONFIDENCE_CHOICE` | `discount` | Feature used to pick the best candidate when confidence < `ROUTER_MIN_CONFIDENCE`: `discount`, `cost`, `latency`, `throughput`, `uptime`, `context_window`, `reasoning`, `composite_score`, `coding_index`, `intelligence_index`, `agentic_index` |
| `ROUTER_PREFERENCE` | `cost` | Tie-breaker handed to the decision model: `cost`, `quality`, or `none` |
| `ROUTER_FALLBACK_MODE` | `static` | `static` = auto-fallback to `ROUTER_STATIC_FALLBACK`; `halt` = return 503 |
| `ROUTER_STATIC_FALLBACK` | `openrouter/deepseek/deepseek-v3.1` | Fallback model when routing fails |
| `ROUTER_MODEL_HANDOFF` | `next_best` | Behavior when selected model falls off discount list: `stop`, `next_best`, `keep_current` |
| `ROUTER_CACHE_FILE` | `data/models.json` | Path to cached discounted models |
| `ROUTER_REFRESH_CRON` | `0 * * * *` | Cron schedule for inventory refresh (default: hourly) |
| `ROUTER_MIN_CONTEXT_LENGTH` | `131072` | Minimum context length in tokens; models with shorter context are excluded |
| `ROUTER_MIN_ENDPOINT_UPTIME` | `95.0` | Minimum 1d endpoint uptime (%) to consider endpoint stats trustworthy |
| `ROUTER_MIN_ENDPOINT_REQUESTS` | `100` | Minimum 30m endpoint request count to consider endpoint latency trustworthy |
| `ROUTER_WEIGHTS` | `{"discount": 0.3, "cost": 0.2, "latency": 0.15, "throughput": 0.15, "uptime": 0.1, "coding_index": 0.1, ...}` | Feature weights for the decision model. JSON string, dict, or comma-separated pairs (`"discount=0.3,cost=0.2"`). Features missing from a candidate are treated as 0. |
| `LANGFUSE_HOST` | | Langfuse base URL (e.g. `http://localhost:3000`). Set with the two keys below to send traces to Langfuse; otherwise traces print to console |
| `ROUTER_TRACE_INPUT` | `truncated` | What a trace records as input: `none`, `decision`, `truncated`, or `full` |
| `ROUTER_TRACE_INPUT_CHARS` | `500` | Character cap when `ROUTER_TRACE_INPUT=truncated` |
| `ROUTER_TRACE_ROUTING_DETAIL` | `true` | Record the routing decision (model, confidence, reason) as trace output |
| `LANGFUSE_PUBLIC_KEY` | | Langfuse public key (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | | Langfuse secret key (`sk-lf-...`) |

## Model Handoff

When the selected model falls off the discount list, the router handles it based on `ROUTER_MODEL_HANDOFF`:

- **`stop`** — Returns 400 with a message telling the user the model is no longer discounted.
- **`next_best`** (default) — Automatically switches to the best available discounted model.
- **`keep_current`** — Continues using the model even though it's no longer discounted.

## How It Works

1. **Inventory refresh** (hourly cron): sweeps OpenRouter `/models` and every `/models/{id}/endpoints` concurrently to find discounted models. Discounts are per-endpoint; the router picks the endpoint with the largest discount among trustworthy providers (uptime ≥95%, request count ≥100). Non-chat models (audio/video/embedding/placeholder rows) are dropped. Saved to `data/models.json`.

2. **Request arrives** at `/v1/chat/completions`.

3. **Hard requirements derived** from the transcript: image parts, tool calls, and an estimated context floor (`ROUTER_MIN_CONTEXT_LENGTH`, default 128K). Models that cannot satisfy any of these are excluded.

4. **Decision model called** via OpenRouter Decisions API (`POST /api/alpha/decisions`). The eligible pool (all models passing hard requirements) is sent as `criteria` — one entry per candidate with its facts (name, description, modalities, benchmark scores, discount, price, serving endpoint stats). The user's configured weights are included in the instructions so the decision model weighs trade-offs accordingly.

5. **Decision model picks** a candidate. If confidence ≥ `ROUTER_MIN_CONFIDENCE`, the request is forwarded. Otherwise the best candidate according to `ROUTER_LOW_CONFIDENCE_CHOICE` (default `discount`) is used.

6. **Forwarding**: proxied to OpenRouter with streaming support and rate limiting.

7. **Handoff**: if the selected model falls off the discount list before the next refresh, `ROUTER_MODEL_HANDOFF` controls whether to stop, switch to the next best, or keep using it.

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
                                    decisions API picks candidate
                                      ↙          ↘
                                 confident    low confidence
                                    ↓              ↓
                              forward()      best available model
```

- **Inventory**: Fetches discounted models from OpenRouter `/models` and `/models/{id}/endpoints` endpoints concurrently (semaphore 10). Discount data is per-endpoint — the router scans all endpoints per model and picks the best trustworthy one. Cached to `data/models.json`. Refreshed on cron schedule.
- **Router**: Derives hard requirements from the transcript (image, tools, context), filters the candidate pool, and sends the full eligible set to the decision model with user-configured feature weights. No shortlist, no ranking bias — the decision model sees all candidates and weighs them according to the configured weights.
- **Decision model**: Called via OpenRouter Decisions API (`POST /api/alpha/decisions`). The decision model is pluggable — Jev today, local open-weight models later. Returns a typed answer with a calibrated `confidence`.
- **Forwarding**: Proxies to OpenRouter with streaming support and rate limiting.
- **Rate Limiter**: Reactive — tracks 429 responses and backs off per-model.

## License

[MIT](LICENSE)

If this project is useful, consider [buying me a coffee](https://ko-fi.com/jerrymindek).


```bash
uv sync
uv run pytest -v
uv run pyright src/
```

### Tuning notes

Live testing against the real discounted pool informed the defaults.

- **`ROUTER_MIN_DISCOUNT=0.25`**. The threshold controls which models enter the pool. Lower values include more candidates (and more competition for the decision model). The default was 0.50 but was lowered to 0.25 after the inventory sweep was fixed to scan all endpoints — models with a discount on a non-first endpoint were previously missed.

- **`ROUTER_WEIGHTS`**. The user-configurable feature weights guide the decision model's selection. All weights are passed to the decision model as evaluation guidance; the decision model weighs them in context.
  Available features:
  - `discount` (% off)
  - `cost` / `price` (effective blended USD/Mtok based on 75% input, 25% output)
  - `latency` (p50 ms)
  - `throughput` (p50 tok/s)
  - `uptime` (%)
  - `context_window` (token context capacity)
  - `reasoning` (native reasoning/thinking model support)
  - `composite_score` / `model_tier` (benchmark average with heuristic model tier fallback)
  - `coding_index`, `intelligence_index`, `agentic_index` (Artificial Analysis scores)

- **`ROUTER_MIN_CONFIDENCE=0.30`.** Confidence for model selection sits
  roughly 0.30–0.55 with a good pool. A 0.55 bar abstained often; 0.30 leaves
  room for the crossing cases while still rejecting the near-coin-flip
  answers a bad pool produces.

- **`ROUTER_PREFERENCE=cost`.** Of the tie-breakers tested, only cost raised
  confidence (0.30 → 0.35 on a large pool). Open-weight and closed-weight
  preferences *lowered* it, and a reasoning preference lowered it and was
  ignored. A detailed judging rubric lowered it too. A clean question with one
  real discriminator beats a long list of constraints.

- **No transcript sent to the decision model**. The user prompt is not forwarded. Hard requirements (image/tools/context) are derived from it and applied as filters; the user's weights define the optimization objective. Sending the transcript was found to bias the decision model toward whatever the user happened to write rather than the features that matter.

- **No shortlist cap**. The decision model sees all eligible candidates. Capping the pool was found to collapse confidence when candidates are near-identical, and the rank-sum shortlist introduced bias by pre-selecting which models the decision model could choose from.

- **Verbatim benchmark scores** are included per candidate (intelligence, coding, agentic). Missing scores are treated as 0, not omitted, so the decision model can distinguish "no data" from "low score."

- **Serving endpoint facts** (endpoint name, latency p50, throughput, uptime, request count) are included per candidate so the decision model can weigh provider reliability and speed alongside discount and capability.


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

