"""Model selection via a System One decision model (typesafe/jev-1.13).

The decision model is NOT a chat model. It is called through the OpenRouter
Decisions API:

    POST https://openrouter.ai/api/alpha/decisions

Request carries ``state`` plus a map of typed ``questions``; the response
carries typed ``answers`` with probabilities and a calibrated confidence.
See: https://openrouter.ai/docs/api/api-reference/alphadecisions
"""

import logging
from dataclasses import dataclass

import httpx

from src.config import settings
from src.models import ModelInfo

logger = logging.getLogger(__name__)

DECISIONS_PATH = "/api/alpha/decisions"

# Keep the candidate pool small. Measured on the real discounted pool: with
# 21 near-identical text models confidence collapses to ~0.10, but with 5
# eligible candidates it rises to ~0.45-0.55 and the pick reflects capability
# Preference text handed to the decision model. Live testing showed that vague
# or redundant constraints lower confidence; a concrete cost tie-breaker raises
# it. Open/closed-weight and reasoning preferences were tested and removed.
_PREFERENCE_TEXT = {
    "cost": (
        "when candidates are otherwise comparable, prefer the cheaper one "
        "(lower effective price per million tokens)"
    ),
    "quality": "prefer the strongest candidate for the task",
    "none": "",
}





@dataclass
class RoutingDecision:
    """Outcome of a decision-model call."""

    model_id: str | None
    confidence: float
    reason: str  # "ok" | "no_models" | "no_api_key" | "http_error" | "parse_error"
    # | "unknown_choice" | "low_confidence"


def _format_weights_instruction(weights: dict[str, float]) -> str:
    """Format user evaluation weights into clear decision guidelines for Jev."""
    active = {k: v for k, v in weights.items() if v > 0.0}
    if not active:
        return ""
    sorted_weights = sorted(active.items(), key=lambda kv: -kv[1])
    weight_desc = ", ".join(f"{k}: {int(v * 100) if v < 1.0 else v}%" for k, v in sorted_weights)
    return (
        f"User evaluation weights: {weight_desc}. "
        "Weigh these factors when selecting the best candidate for this request."
    )


def _build_criteria(models: list[ModelInfo]) -> dict[str, str]:
    """Build the ``choice`` criteria: candidate id -> capability hint.

    Every filtered candidate is sent, unranked and uncapped: the decision
    model should see the full eligible pool with no shortlist bias. Hard
    requirements (image/tools/context) were already applied by
    _filter_candidates; beyond that, selection is entirely the decision
    model's call, guided by the user's weights.
    """
    return {m.id: _describe_model(m) for m in models}


@dataclass
class Requirements:
    """What the request itself demands of a model."""

    needs_image_input: bool = False
    needs_tools: bool = False
    min_context: int = 0


def _request_requirements(messages: list, estimated_tokens: int = 0) -> Requirements:
    """Derive hard requirements from the request transcript.

    These filter the candidate pool before the decision model sees it, so it
    only ranks models that can actually serve the request.
    """
    context_floor = max(estimated_tokens, settings.min_context_length)
    req = Requirements(min_context=context_floor)
    for m in messages:
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type")
                    if t in ("image_url", "input_image", "image"):
                        req.needs_image_input = True
        if getattr(m, "tool_calls", None) or (
            isinstance(m, dict) and m.get("tool_calls")
        ):
            req.needs_tools = True
        role = getattr(m, "role", None)
        if role is None and isinstance(m, dict):
            role = m.get("role")
        if role == "tool":
            req.needs_tools = True
    return req


def _filter_candidates(
    models: list[ModelInfo], req: Requirements
) -> list[ModelInfo]:
    """Keep only models that satisfy the request's hard requirements.

    Falls back to the unfiltered pool when filtering would leave nothing, so
    the router never refuses to route.
    """
    kept = []
    for m in models:
        if req.needs_image_input and "image" not in set(m.input_modalities):
            continue
        if req.needs_tools and "tools" not in set(m.supported_parameters):
            continue
        if req.min_context and m.context_length and m.context_length < req.min_context:
            continue
        kept.append(m)
    if not kept:
        logger.info("Candidate filter left nothing; using the full pool")
        return models
    return kept


def _describe_model(m: ModelInfo) -> str:
    """One line describing a candidate: identity, strengths, cost, modality."""
    name = m.name or m.id
    facts: list[str] = [name]

    # Strengths come from the provider's own description; trim to a sentence.
    if m.description:
        facts.append(_first_sentence(m.description))

    modalities = set(m.input_modalities)
    extras = sorted(modalities - {"text"})
    if extras:
        facts.append(f"accepts {'+'.join(extras)}")

    features: list[str] = []
    params = set(m.supported_parameters)
    if "tools" in params:
        features.append("tool calling")
    if "reasoning" in params or "include_reasoning" in params:
        features.append("reasoning")
    if "structured_outputs" in params:
        features.append("structured output")
    if features:
        facts.append(", ".join(features))

    if m.context_length:
        facts.append(f"{m.context_length} token context")

    # Benchmark scores are the real capability signal; without them every
    # candidate reads the same. Treat missing/null scores explicitly as 0.
    scores: list[str] = [
        f"intelligence {m.intelligence_index or 0.0:.0f}",
        f"coding {m.coding_index or 0.0:.0f}",
        f"agentic {m.agentic_index or 0.0:.0f}",
    ]
    facts.append(", ".join(scores))

    facts.append(f"{m.discount_pct:.0f}% off")
    # Effective price is the tie-breaker the cost preference acts on.
    facts.append(f"${m.input_price * 1e6:.2f}/Mtok in")

    # Serving facts describe the provider, not the model. The same model is
    # often served by many providers at different speeds and reliabilities,
    # so without these the decision cannot tell a fast endpoint from a broken
    # one, and price is the only comparable signal it has.
    served: list[str] = []
    if m.endpoint_name:
        served.append(f"served by {m.endpoint_name}")
    if m.latency_p50_ms is not None:
        served.append(f"{m.latency_p50_ms:.0f}ms p50")
    if m.throughput_p50 is not None:
        served.append(f"{m.throughput_p50:.0f} tok/s")
    if m.uptime_pct is not None:
        served.append(f"{m.uptime_pct:.0f}% uptime")
    if served:
        facts.append(", ".join(served))

    return "; ".join(facts)


def _first_sentence(text: str, limit: int = 200) -> str:
    """First sentence of a description, capped for prompt size."""
    text = " ".join(text.split())
    for end in (". ", "! ", "? "):
        idx = text.find(end)
        if idx != -1:
            text = text[: idx + 1]
            break
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _estimate_tokens(messages: list) -> int:
    """Rough token count for the full transcript (chars / 4).

    Measured on the untruncated text: this is the context floor used to filter
    candidates, so it must reflect what the caller actually asked for, not the
    capped state sent to the decision model.
    """
    return len(render_transcript(messages)) // 4


def render_transcript(messages: list) -> str:
    """Render the transcript with no truncation (used for size estimates)."""
    parts: list[str] = []
    for m in messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None) or "user"
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        parts.append(f"[{role}] {_render_content(content)}")
    return "\n\n".join(parts)


def _render_content(content) -> str:
    """Flatten a message content value (string or content-part list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                else:
                    chunks.append(f"<{part.get('type', 'part')}>")
            else:
                chunks.append(str(part))
        return " ".join(chunks)
    return str(content) if content is not None else ""


async def classify_request(messages: list, models: list[ModelInfo]) -> RoutingDecision:
    """Ask the decision model which discounted model should serve this request.

    The prompt transcript is not sent: hard requirements derived from it
    (image/tools/context) are applied by _filter_candidates, and the user's
    weights define the optimization objective. The decision model sees only
    the eligible candidates and their facts.

    Returns a :class:`RoutingDecision`. ``model_id`` is ``None`` unless the
    decision model picked a known candidate with confidence at or above
    ``ROUTER_MIN_CONFIDENCE``.
    """
    if not models:
        logger.warning("No models available for routing")
        return RoutingDecision(None, 0.0, "no_models")

    api_base, api_key = _get_router_api()
    if not api_key:
        logger.error("No router API key configured")
        return RoutingDecision(None, 0.0, "no_api_key")

    criteria = _build_criteria(models)
    if not criteria:
        return RoutingDecision(None, 0.0, "no_models")

    req = _request_requirements(messages, estimated_tokens=_estimate_tokens(messages))
    filtered = _filter_candidates(models, req)
    if len(filtered) != len(models):
        logger.info(
            f"Filtered candidates {len(models)} -> {len(filtered)} "
            f"(image={req.needs_image_input}, tools={req.needs_tools}, "
            f"min_context={req.min_context})"
        )
        criteria = _build_criteria(filtered)
    if not criteria:
        return RoutingDecision(None, 0.0, "no_models")

    preference = _PREFERENCE_TEXT.get(settings.preference, "")
    weights_guidance = _format_weights_instruction(settings.weights)

    instructions = (
        "Choose which candidate model should serve this request. The user's "
        "weights state what they are optimizing for; weigh the candidates' "
        "facts accordingly."
    )
    if weights_guidance:
        instructions += f" {weights_guidance}"
    if preference:
        instructions += f" Tie-breaker: {preference}."
    instructions += " Answer with one of the supplied options."

    payload = {
        "model": settings.model,
        "state": {
            "preference": preference or "none",
            "weights": settings.weights,
        },
        "questions": {
            "target": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }

    endpoint = f"{_decisions_base(api_base)}{DECISIONS_PATH}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as e:
        logger.error(f"Decision model request failed: {e}")
        return RoutingDecision(None, 0.0, "http_error")

    if resp.status_code != 200:
        logger.error(f"Decision model failed: {resp.status_code} {resp.text}")
        return RoutingDecision(None, 0.0, "http_error")

    try:
        data = resp.json()
    except ValueError as e:
        logger.error(f"Decision model returned invalid JSON: {e}")
        return RoutingDecision(None, 0.0, "parse_error")

    return _parse_decision(data)


def _parse_decision(data: dict) -> RoutingDecision:
    """Validate and interpret a Decisions API response.

    A missing or malformed answer is an error, never approval.
    """
    answer = data.get("answers", {}).get("target")
    if not isinstance(answer, dict):
        logger.warning(f"Decision model returned no 'target' answer: {data}")
        return RoutingDecision(None, 0.0, "parse_error")

    choice = answer.get("choice")
    if not isinstance(choice, str) or not choice:
        logger.warning(f"Decision model answer has no choice: {answer}")
        return RoutingDecision(None, 0.0, "parse_error")

    raw_confidence = answer.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < settings.min_confidence:
        logger.info(
            f"Decision confidence {confidence:.2f} below "
            f"{settings.min_confidence:.2f}; abstaining (chose {choice})"
        )
        return RoutingDecision(None, confidence, "low_confidence")

    # The resolved version is reported back in ``model``; log it for attribution.
    resolved = data.get("model")
    logger.info(f"Decision model selected: {choice} (confidence {confidence:.2f}, resolved {resolved})")
    return RoutingDecision(choice, confidence, "ok")


def _decisions_base(api_base: str) -> str:
    """Normalize the API base to the OpenRouter host root.

    ROUTER_API_BASE defaults to ``https://openrouter.ai/api/v1`` but the
    Decisions API lives at ``https://openrouter.ai/api/alpha/decisions``.
    """
    return api_base.rstrip("/").removesuffix("/api/v1")


def _get_router_api() -> tuple[str, str]:
    """Get decision-model API base and key based on mode."""
    if settings.mode == "local":
        return settings.omlx_base, ""
    elif settings.mode == "cloud":
        return settings.api_base, settings.api_key
    else:
        if settings.api_key:
            return settings.api_base, settings.api_key
        return settings.omlx_base, ""
