import asyncio
import json
import logging
from pathlib import Path

import httpx

from src.config import settings
from src.models import ModelInfo

logger = logging.getLogger(__name__)

ENDPOINT_CONCURRENCY = 10

# Models whose output is not text cannot serve chat completions. Discounts
# also appear on audio/video/embedding models, so drop them.
_TEXT_OUTPUT = {"text"}

# Suffixes for models that exist but are not callable through
# /chat/completions. OpenRouter rejects these with 404, e.g.
# "google/gemini-4-argon:batch cannot be used with the chat/completions
# endpoint". Batch variants look discounted and chat-capable otherwise, so
# they must be dropped explicitly.
_NON_SYNC_SUFFIXES = (":batch", ":free")


def _is_chat_capable(m: ModelInfo) -> bool:
    """True when the model can serve a chat completion.

    Rejects: no text output (audio, video, image, embeddings), models that
    exist but are not callable through /chat/completions (batch/free
    variants), and the docs/placeholder rows OpenRouter lists without any real
    modality.
    """
    if m.id.endswith(_NON_SYNC_SUFFIXES):
        return False
    if not m.output_modalities:
        return False
    if not _TEXT_OUTPUT.issubset(set(m.output_modalities)):
        return False
    # Junk rows (e.g. "docs/quickstart") carry no context window.
    if m.context_length <= 0:
        return False
    return True


def _meta_from_api(m_meta: dict) -> dict:
    """Pull capability hints out of one /models entry."""
    arch = m_meta.get("architecture", {}) or {}
    top_prov = m_meta.get("top_provider", {})
    supported = m_meta.get("supported_parameters", []) or []
    aa = (m_meta.get("benchmarks") or {}).get("artificial_analysis") or {}
    return {
        "provider": top_prov.get("name", "") if isinstance(top_prov, dict) else "",
        "name": m_meta.get("name", "") or "",
        "description": m_meta.get("description", "") or "",
        "input_modalities": list(arch.get("input_modalities", []) or []),
        "output_modalities": list(arch.get("output_modalities", []) or []),
        "supported_parameters": list(supported),
        "context_length": int(m_meta.get("context_length", 0) or 0),
        "intelligence_index": _as_float(aa.get("intelligence_index")),
        "coding_index": _as_float(aa.get("coding_index")),
        "agentic_index": _as_float(aa.get("agentic_index")),
    }


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def fetch_discounted_models(
    client: httpx.AsyncClient | None = None,
) -> list[ModelInfo]:
    """Fetch all models from OpenRouter and keep those with an active discount.

    Discounts are not in /models: they only appear in each model's
    /models/{id}/endpoints entry. The full catalogue is therefore swept
    concurrently (see ENDPOINT_CONCURRENCY). Measured: all 462 models in ~4s at
    concurrency 10, with no rate limiting.

    Falls back to the cached inventory when the sweep fails or finds nothing,
    so a provider hiccup never leaves the router with an empty pool.
    """
    base_url = settings.inventory_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}

    try:
        if client is not None:
            found = await _fetch_via_endpoints(client, base_url, headers)
        else:
            async with httpx.AsyncClient(timeout=30.0) as c:
                found = await _fetch_via_endpoints(c, base_url, headers)

        if found:
            save_cache(found)
            logger.info(
                f"Found and cached {len(found)} discounted models "
                f"(>= {settings.min_discount * 100:.0f}%)"
            )
            return found
        logger.warning("Discount sweep found no models; falling back to cache")
    except Exception as e:
        logger.warning(f"Discount sweep failed: {e}. Falling back to cache.")

    cached = load_cache()
    if cached:
        logger.info(f"Loaded {len(cached)} models from cache.")
        return cached
    return []


async def _fetch_via_endpoints(
    c: httpx.AsyncClient, base_url: str, headers: dict
) -> list[ModelInfo]:
    """Fetch models and query individual endpoints concurrently."""
    resp = await c.get(f"{base_url}/models", headers=headers)
    resp.raise_for_status()
    data = resp.json()

    models = data.get("data", [])
    models_by_id = {m["id"]: m for m in models if m.get("id")}
    model_ids = list(models_by_id)

    decision_model = settings.model
    if decision_model and decision_model not in model_ids:
        model_ids.append(decision_model)

    semaphore = asyncio.Semaphore(ENDPOINT_CONCURRENCY)
    tasks = [
        fetch_endpoint(c, base_url, headers, mid, semaphore, models_by_id.get(mid, {}))
        for mid in model_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    discounted = []
    for r in results:
        if isinstance(r, ModelInfo):
            discounted.append(r)

    before = len(discounted)
    discounted = [m for m in discounted if _is_chat_capable(m)]
    if before != len(discounted):
        logger.info(f"Dropped {before - len(discounted)} non-chat models")

    logger.info(
        f"Found {len(discounted)} discounted models (>= {settings.min_discount * 100:.0f}%)"
    )
    return discounted


# Endpoints below these floors are not trustworthy enough to record numbers
# from. uptime 1d under 95% means the provider is failing requests; a request
# count under 100 makes the latency figure noise rather than a measurement.
MIN_ENDPOINT_UPTIME = 95.0
MIN_ENDPOINT_REQUESTS = 100


def _endpoint_perf(ep: dict) -> dict:
    """Pull serving facts out of one endpoint entry."""
    wl = (ep.get("perf_last_30m_by_workload") or {}).get("text_generation") or {}
    return {
        "endpoint_name": ep.get("provider_name", "") or "",
        "latency_p50_ms": _as_float((ep.get("latency_last_30m") or {}).get("p50")),
        "throughput_p50": _as_float((ep.get("throughput_last_30m") or {}).get("p50")),
        "uptime_pct": _as_float(ep.get("uptime_last_1d")),
        "request_count": int(wl.get("request_count", 0) or 0),
    }


def _pick_endpoint(endpoints: list[dict]) -> dict | None:
    """Choose which endpoint's numbers to record for this model.

    Prefers endpoints with trustworthy perf data, and takes the largest
    discount among those. Falls back to the largest discount overall when
    none qualify, so a model is never dropped merely for lacking stats.

    Deliberately does not weigh latency against discount: that trade belongs
    to the decision model, which needs the facts to make it.
    """

    def discount_of(e: dict) -> float:
        return float((e.get("pricing") or {}).get("discount", 0) or 0)

    healthy = [
        e
        for e in endpoints
        if (_endpoint_perf(e)["latency_p50_ms"] is not None)
        and (_endpoint_perf(e)["uptime_pct"] or 0) >= MIN_ENDPOINT_UPTIME
        and _endpoint_perf(e)["request_count"] >= MIN_ENDPOINT_REQUESTS
    ]
    return max(healthy or endpoints, key=discount_of)


async def fetch_endpoint(
    c: httpx.AsyncClient,
    base_url: str,
    headers: dict,
    model_id: str,
    semaphore: asyncio.Semaphore,
    m_meta: dict | None = None,
) -> ModelInfo | None:
    """Fetch endpoints for one model, gated by semaphore."""
    m_meta = m_meta or {}
    async with semaphore:
        try:
            ep_resp = await c.get(
                f"{base_url}/models/{model_id}/endpoints",
                headers=headers,
                timeout=10.0,
            )
            if ep_resp.status_code != 200:
                return None

            ep_data = ep_resp.json()
            endpoints = ep_data.get("data", {}).get("endpoints", [])

            if not endpoints:
                return None

            # Discounts are per-endpoint, and OpenRouter does not order them
            # by discount. Reading only endpoints[0] silently drops models
            # whose discount sits on another endpoint (e.g. z-ai/glm-5.3 has
            # 50% off on endpoint 14 of 43 and 0% on endpoint 0).
            #
            # This is inventory, not routing: it chooses which endpoint's
            # numbers get recorded, not which model serves a request. So it
            # only refuses to record a broken endpoint's numbers; it does not
            # weigh latency against discount. That trade is the decision
            # model's job, and it can only make it if the facts are present.
            ep = _pick_endpoint(endpoints)
            if ep is None:
                return None
            pricing = ep.get("pricing", {})
            discount = pricing.get("discount", 0)

            if discount >= settings.min_discount:
                meta = _meta_from_api(m_meta)
                perf = _endpoint_perf(ep)
                return ModelInfo(
                    id=model_id,
                    discount_pct=discount * 100,
                    provider=ep.get("provider", "") or meta["provider"],
                    input_price=float(pricing.get("prompt", 0) or 0),
                    output_price=float(pricing.get("completion", 0) or 0),
                    context_length=int(ep.get("context_length", 0) or 0)
                    or meta["context_length"],
                    name=meta["name"],
                    description=meta["description"],
                    input_modalities=meta["input_modalities"],
                    output_modalities=meta["output_modalities"],
                    supported_parameters=meta["supported_parameters"],
                    intelligence_index=meta["intelligence_index"],
                    coding_index=meta["coding_index"],
                    agentic_index=meta["agentic_index"],
                    **perf,
                )
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch endpoints for {model_id}: {e}")
    return None


def save_cache(models: list[ModelInfo], path: str | None = None) -> None:
    """Save model cache to JSON file."""
    if path is None:
        path = settings.cache_file

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data = [m.model_dump() for m in models]
    cache_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved {len(models)} models to {path}")


def load_cache(path: str | None = None) -> list[ModelInfo]:
    """Load model cache from JSON file."""
    if path is None:
        path = settings.cache_file

    cache_path = Path(path)
    if not cache_path.exists():
        return []

    try:
        data = json.loads(cache_path.read_text())
        return [ModelInfo(**m) for m in data]
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load cache: {e}")
        return []
