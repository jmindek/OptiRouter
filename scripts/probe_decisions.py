"""Live probe of the configured decision-model provider.

Sends one real request matching the router's current payload structure
(filtered candidates, user weights, serving facts, no transcript bias)
and prints the exact request and response.

Works with every provider (openrouter, ollama, vllm, systemone):

    uv run python scripts/probe_decisions.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.decision_providers import get_provider  # noqa: E402
from src.inventory import load_cache  # noqa: E402
from src.models import ChatMessage  # noqa: E402
from src.router import (  # noqa: E402
    _PREFERENCE_TEXT,
    _build_criteria,
    _estimate_tokens,
    _filter_candidates,
    _format_weights_instruction,
    _request_requirements,
)


def build_payload(messages: list[ChatMessage]) -> dict:
    models = load_cache()
    if not models:
        raise RuntimeError("No cached models; refresh inventory first")

    req = _request_requirements(messages, estimated_tokens=_estimate_tokens(messages))
    filtered = _filter_candidates(models, req)
    criteria = _build_criteria(filtered)

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
    return payload


async def main() -> None:
    provider = get_provider()
    sample_messages = [
        ChatMessage(
            role="user",
            content="Write a Python function that reverses a linked list. Use recursion.",
        )
    ]
    payload = build_payload(sample_messages)

    candidate_count = len(payload["questions"]["target"]["criteria"])
    print(f"=== PROVIDER: {provider.name} ===")
    print(f"Model: {payload['model']}")
    if provider.name == "systemone":
        print(f"Base:  {settings.systemone_base}")
    print(f"=== REQUEST ({candidate_count} candidates) ===")
    print(f"State: {json.dumps(payload['state'])}")
    print(f"Instructions: {payload['questions']['target']['instructions']}")
    print()

    try:
        data = await provider.call(payload)
    except Exception as e:
        print(f"=== FAILED: {type(e).__name__}: {e} ===")
        raise

    print("=== RESPONSE ===")
    print(json.dumps(data, indent=2))
    ans = data.get("answers", {}).get("target", {})
    print("\n=== SUMMARY ===")
    print(f"Chosen model: {ans.get('choice')}")
    print(f"Confidence:   {ans.get('confidence')}")
    print(f"Min conf:     {settings.min_confidence}")
    confidence = ans.get("confidence")
    if isinstance(confidence, (int, float)):
        verdict = "accepted" if confidence >= settings.min_confidence else "ABSTAINS"
        print(f"Router would: {verdict}")
    top3 = sorted(ans.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:3]
    print(f"Top 3 probs:  {top3}")


if __name__ == "__main__":
    asyncio.run(main())
