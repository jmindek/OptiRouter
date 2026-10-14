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
# (benchmark scores) rather than only price. Below 5 the model is unsure the
# right option is even present. See README for the pool-size curve.
MAX_CANDIDATES = 5

# Cap the request state. Jev rejects the whole request with
# "max_tokens_exceeded" when state + questions overflow its window, so the
# transcript is truncated from the middle: the start (system prompt, first
# ask) and the end (the actual last instruction) carry the intent.
MAX_STATE_CHARS = 24000  # ~6k tokens; leaves room for the candidate list

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


def _rank_candidates(models: list[ModelInfo]) -> list[ModelInfo]:
    """Order candidates so the capped pool spans the real tradeoff.

    Ranking by discount alone gives five cheap near-twins; ranking by score
    alone gives five expensive near-twins. Summed ranks keep cheap, mid and
    strong options in view, which is what lets the decision model compare
    price against capability instead of guessing among equals. Measured on the
    real pool: blended selection lifts confidence to ~0.80 (was ~0.52 by
    discount, ~0.38 by score).
    """
    by_score = sorted(
        models,
        key=lambda m: (
            -(m.coding_index or m.intelligence_index or 0.0),
            -m.discount_pct,
            m.input_price + m.output_price,
        ),
    )
    score_rank = {m.id: i for i, m in enumerate(by_score)}
    discount_rank = {
        m.id: i
        for i, m in enumerate(
            sorted(models, key=lambda m: (-m.discount_pct, m.input_price + m.output_price))
        )
    }
    return sorted(models, key=lambda m: score_rank[m.id] + discount_rank[m.id])


def _build_criteria(models: list[ModelInfo]) -> dict[str, str]:
    """Build the ``choice`` criteria: candidate id -> capability hint.

    The hint tells the decision model what the model is and what it is good
    for. Discount, price and benchmark scores are included so the decision has
    real signals to compare instead of near-identical blurbs.
    """
    criteria: dict[str, str] = {}
    for m in _rank_candidates(models)[:MAX_CANDIDATES]:
        criteria[m.id] = _describe_model(m)
    return criteria


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
    req = Requirements(min_context=estimated_tokens)
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
    # candidate reads the same.
    scores: list[str] = []
    if m.intelligence_index is not None:
        scores.append(f"intelligence {m.intelligence_index:.0f}")
    if m.coding_index is not None:
        scores.append(f"coding {m.coding_index:.0f}")
    if m.agentic_index is not None:
        scores.append(f"agentic {m.agentic_index:.0f}")
    if scores:
        facts.append(", ".join(scores))

    facts.append(f"{m.discount_pct:.0f}% off")
    # Effective price is the tie-breaker the cost preference acts on.
    facts.append(f"${m.input_price * 1e6:.2f}/Mtok in")

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


def _render_state(messages: list) -> str:
    """Render the request transcript for the decision model.

    The prompt carries its own context, so the router does not summarise or
    tag it. Everything the caller sent is passed through, in order, except
    that an oversized transcript is truncated from the middle.
    """
    return _truncate_middle(render_transcript(messages))


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


def _truncate_middle(text: str, limit: int = MAX_STATE_CHARS) -> str:
    """Trim a long transcript from the middle, keeping both ends."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return (
        f"{text[:head]}\n\n[... {omitted} characters omitted ...]\n\n{text[-tail:]}"
    )


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

    The full request transcript is sent verbatim: the prompt carries its own
    context, so no keyword tag or summary is derived from it.

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

    instructions = (
        "Read the request below, then choose which candidate model should "
        "serve it. Match the candidate's strengths to what the request "
        "actually needs."
    )
    if preference:
        instructions += f" Tie-breaker: {preference}."
    instructions += " Answer with one of the supplied options."

    payload = {
        "model": settings.model,
        "state": {
            "request": _render_state(messages),
            "preference": preference or "none",
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
