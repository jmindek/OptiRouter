import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.config import settings
from src.handoff import handle_model_handoff, model_in_inventory
from src.inventory import fetch_discounted_models, load_cache, save_cache
from src.models import ChatCompletionRequest
from src.rate_limiter import rate_limiter
from src.router import RoutingDecision, classify_request, render_transcript
from src.telemetry import (
    errors_total,
    handoff_events,
    init_telemetry,
    inventory_errors,
    inventory_refreshes,
    model_selections,
    rate_limit_events,
    request_duration,
    tracer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_cron(expr: str) -> dict:
    """Parse cron expression to APScheduler kwargs."""
    parts = expr.strip().split()
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_telemetry()
    logger.info("Starting model router...")
    await refresh_inventory()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        refresh_inventory,
        "cron",
        **_parse_cron(settings.refresh_cron),
        id="inventory_refresh",
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    yield
    scheduler.shutdown()


async def refresh_inventory():
    """Refresh discount models from OpenRouter."""
    with tracer.start_as_current_span("inventory_refresh"):
        inventory_refreshes.add(1)
        try:
            models = await fetch_discounted_models()
            save_cache(models)
            logger.info(f"Inventory refreshed: {len(models)} models")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            inventory_errors.add(1)
            logger.error(f"Inventory refresh failed: {e}")


def _build_forward_body(messages: list, model: str, stream: bool, request: ChatCompletionRequest) -> dict:
    """Build the JSON body to forward to OpenRouter, preserving all original fields."""
    msg_list = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]
    body: dict = {"model": model, "messages": msg_list, "stream": stream}
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.tools is not None:
        body["tools"] = request.tools
    if request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    if request.logprobs is not None:
        body["logprobs"] = request.logprobs
    if request.top_logprobs is not None:
        body["top_logprobs"] = request.top_logprobs
    if request.stop is not None:
        body["stop"] = request.stop
    if request.presence_penalty is not None:
        body["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        body["frequency_penalty"] = request.frequency_penalty
    if request.response_format is not None:
        body["response_format"] = request.response_format
    if request.seed is not None:
        body["seed"] = request.seed
    if request.user is not None:
        body["user"] = request.user
    return body


def _record_trace_content(
    span, messages: list, model: str, decision: "RoutingDecision | None"
) -> None:
    """Attach trace input/output to a span, per the configured policy.

    Prompts are user content, so what gets recorded is user-configurable:
    `ROUTER_TRACE_INPUT` is `none`, `decision`, `truncated` (default) or `full`.
    The routing decision is router metadata and is controlled separately by
    `ROUTER_TRACE_ROUTING_DETAIL`.
    """
    policy = settings.trace_input

    if policy == "full":
        span.set_attribute("input", render_transcript(messages))
    elif policy == "truncated":
        span.set_attribute("input", render_transcript(messages)[: settings.trace_input_chars])
    elif policy == "decision":
        pass  # no prompt content, decision only

    if settings.trace_routing_detail and decision is not None:
        parts = [f"model={model}", f"reason={decision.reason}"]
        if decision.confidence:
            parts.append(f"confidence={decision.confidence:.2f}")
        span.set_attribute("output", " ".join(parts))


def _apply_rate_limit_policy(
    status_code: int, model: str, attempt: int, max_retries: int
) -> bool:
    """Shared 429 policy for both the streaming and non-streaming paths.

    Records the rate limit and decides whether to retry. Both transports call
    this so the backoff policy cannot drift between them.

    Returns True when the caller should retry (after the backoff sleep),
    False when the response is not rate limited and the caller should proceed.
    Raises HTTPException(429) when attempts are exhausted.
    """
    if status_code != 429:
        return False

    rate_limiter.record_429(model)
    rate_limit_events.add(1, {"model": model})
    if attempt < max_retries - 1:
        return True
    raise HTTPException(status_code=429, detail="Rate limited")


async def forward_to_openrouter(
    messages: list, model: str, stream: bool = False, request: ChatCompletionRequest | None = None
) -> dict | StreamingResponse:
    """Forward request to OpenRouter with reactive rate limiting.

    Preserves all original request fields (temperature, tools, logprobs, etc.)
    and passes through OpenRouter response directly for non-streaming.
    """
    if request is None:
        request = ChatCompletionRequest(messages=messages, stream=stream)

    api_base = settings.inventory_url.removesuffix("/v1").rstrip("/")
    api_key = settings.openrouter_api_key
    body = _build_forward_body(messages, model, stream, request)

    max_retries = 3

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        # Streaming needs its own retry loop around connection setup. It cannot
        # reuse the loop below, because the response body is consumed later, in
        # the generator; the retry decision has to happen before any byte is
        # handed to the client. Once the connection is open the client and
        # stream are transferred to the generator, which closes them.
        client: httpx.AsyncClient | None = None
        stream_ctx = None
        resp = None
        try:
            for attempt in range(max_retries):
                await rate_limiter.wait_if_needed(model)

                client = httpx.AsyncClient(timeout=120.0)
                stream_ctx = client.stream(
                    "POST",
                    f"{api_base}/v1/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp = await stream_ctx.__aenter__()

                # A 429 at setup is retryable: nothing has been sent yet.
                if _apply_rate_limit_policy(
                    resp.status_code, model, attempt, max_retries
                ):
                    await stream_ctx.__aexit__(None, None, None)
                    await client.aclose()
                    client = stream_ctx = resp = None
                    await asyncio.sleep(2**attempt)
                    continue

                if resp.status_code >= 400:
                    # Not retryable, and it must not look like a success. Read
                    # the body for the log, then surface a real error so the
                    # caller sees a failure instead of an empty 200 stream.
                    status = resp.status_code
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    logger.error(
                        f"Upstream stream failed {status}: {detail[:500]}"
                    )
                    await stream_ctx.__aexit__(None, None, None)
                    await client.aclose()
                    client = stream_ctx = resp = None
                    raise HTTPException(
                        status_code=502,
                        detail=f"Upstream stream failed {status}",
                    )

                rate_limiter.clear(model)
                break
            else:
                raise HTTPException(status_code=503, detail="Max retries exceeded")

            # Ownership of client and stream_ctx moves into the generator so
            # the connection stays open for the whole response. Returning
            # while still inside the context would close it before FastAPI
            # iterates the body, truncating the stream.
            async def event_generator(
                gen_resp=resp, gen_ctx=stream_ctx, gen_client=client
            ):
                try:
                    # Pass SSE frames through verbatim. Re-joining them corrupts
                    # the framing the downstream OpenAI client relies on.
                    async for chunk in gen_resp.aiter_raw():
                        yield chunk
                finally:
                    await gen_ctx.__aexit__(None, None, None)
                    await gen_client.aclose()

            return StreamingResponse(
                event_generator(), media_type="text/event-stream"
            )
        except BaseException:
            # Setup failed before the generator existed to own these. httpx
            # tolerates __aexit__ after a failed __aenter__, so this also
            # covers a failure while opening the connection.
            if stream_ctx is not None:
                await stream_ctx.__aexit__(None, None, None)
            if client is not None:
                await client.aclose()
            raise

    for attempt in range(max_retries):
        await rate_limiter.wait_if_needed(model)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"{api_base}/v1/chat/completions",
                    headers=headers,
                    json=body,
                )
                if _apply_rate_limit_policy(
                    resp.status_code, model, attempt, max_retries
                ):
                    await asyncio.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                rate_limiter.clear(model)
                # Pass through OpenRouter response directly — preserves tool calls,
                # logprobs, and full OpenAI schema.
                return resp.json()

            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

    raise HTTPException(status_code=503, detail="Max retries exceeded")


app = FastAPI(title="Discount Model Router", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main routing endpoint."""
    start = time.time()
    model = "unknown"
    decision: RoutingDecision | None = None
    with tracer.start_as_current_span("chat_completions") as span:
        try:
            messages = request.messages

            models = load_cache()
            if not models:
                if settings.fallback_mode == "halt":
                    raise HTTPException(status_code=503, detail="No models available")
                model = settings.static_fallback
            else:
                decision = await classify_request(messages, models)
                if decision.model_id:
                    model = decision.model_id
                elif decision.reason == "low_confidence":
                    # Decision model answered but was not sure enough: use the
                    # best available discounted model rather than a fixed one.
                    model = models[0].id
                    logger.info(
                        f"Decision abstained (confidence {decision.confidence:.2f}), "
                        f"using best available: {model}"
                    )
                elif settings.fallback_mode == "halt":
                    raise HTTPException(status_code=503, detail="Routing failed")
                else:
                    model = settings.static_fallback

            # Check if selected model is still discounted
            if models and not model_in_inventory(model, models):
                model, error = handle_model_handoff(model, models)
                if error:
                    raise HTTPException(status_code=400, detail=error)
                handoff_events.add(1, {"reason": "model_removed_from_discount"})

            model_selections.add(1, {"model": model})
            span.set_attribute("router.model", model)
            _record_trace_content(span, messages, model, decision)

            result = await forward_to_openrouter(messages, model, request.stream, request)
            if isinstance(result, StreamingResponse):
                return result

            # Non-streaming: pass through OpenRouter response directly.
            # Preserves tool_calls, logprobs, and full OpenAI schema.
            return result
        except HTTPException:
            raise
        except Exception as e:
            errors_total.add(1, {"type": type(e).__name__})
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            request_duration.record(
                duration_ms, {"model": model if "model" in dir() else "unknown"}
            )


@app.get("/healthz")
async def health():
    """Health check with inventory freshness info."""
    cache_path = Path(settings.cache_file)
    cache_info = {}
    if cache_path.exists():
        stat = cache_path.stat()
        age_s = time.time() - stat.st_mtime
        cache_info = {
            "exists": True,
            "age_seconds": round(age_s, 0),
            "age_human": f"{age_s/3600:.1f}h" if age_s > 3600 else f"{age_s/60:.0f}m",
        }
    else:
        cache_info = {"exists": False}

    models = load_cache()
    return {
        "status": "ok",
        "cache": cache_info,
        "cached_models": len(models),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
