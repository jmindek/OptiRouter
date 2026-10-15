"""Live probe of the OpenRouter Decisions API.

Sends one real request matching the router's current payload structure
(filtered candidates, user weights, serving facts, no transcript bias)
and prints the exact request and response.

    uv run python scripts/probe_decisions.py
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.inventory import load_cache  # noqa: E402
from src.models import ChatMessage  # noqa: E402
from src.router import (  # noqa: E402
    DECISIONS_PATH,
    _PREFERENCE_TEXT,
    _build_criteria,
    _decisions_base,
    _estimate_tokens,
    _filter_candidates,
    _format_weights_instruction,
    _get_router_api,
    _request_requirements,
)


def build_payload(messages: list[ChatMessage]) -> tuple[dict, str, str]:
    api_base, api_key = _get_router_api()
    if not api_key:
        raise RuntimeError("No ROUTER_API_KEY or OPENROUTER_API_KEY found")

    models = load_cache()
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
    url = f"{_decisions_base(api_base)}{DECISIONS_PATH}"
    return payload, url, api_key


async def main() -> None:
    sample_messages = [
        ChatMessage(
            role="user",
            content="Write a Python function that reverses a linked list. Use recursion.",
        )
    ]
    payload, url, api_key = build_payload(sample_messages)

    candidate_count = len(payload["questions"]["target"]["criteria"])
    print(f"=== REQUEST ({candidate_count} candidates) ===")
    print(f"POST {url}")
    print(f"Model: {payload['model']}")
    print(f"State: {json.dumps(payload['state'])}")
    print(f"Instructions: {payload['questions']['target']['instructions']}")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    print(f"=== RESPONSE (status {resp.status_code}) ===")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
        ans = data.get("answers", {}).get("target", {})
        print("\n=== SUMMARY ===")
        print(f"Chosen model: {ans.get('choice')}")
        print(f"Confidence:   {ans.get('confidence')}")
        top3 = sorted(ans.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:3]
        print(f"Top 3 probs:  {top3}")
    except ValueError:
        print(resp.text)


if __name__ == "__main__":
    asyncio.run(main())
