"""Live probe of the OpenRouter Decisions API (typesafe/jev-1.13).

Sends one real request and prints the exact request/response so we can verify
our assumed contract. Uses the real discounted inventory from the cache.

    uv run python scripts/probe_decisions.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inventory import load_cache  # noqa: E402
from src.router import _build_criteria, _decisions_base  # noqa: E402

MODEL = os.getenv("ROUTER_MODEL", "typesafe/jev-1.13")
API_KEY = os.getenv("ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
API_BASE = os.getenv("ROUTER_API_BASE", "https://openrouter.ai/api/v1")
URL = f"{_decisions_base(API_BASE)}/api/alpha/decisions"

TASK = "Write a Python function that reverses a linked list. Use recursion."


def build_payload() -> dict:
    models = load_cache()
    criteria = _build_criteria(models)
    return {
        "model": MODEL,
        "state": {"message": TASK},
        "questions": {
            "target": {
                "type": "choice",
                "instructions": (
                    "Which model should serve this request? Pick the one that best "
                    "fits the task. Prefer higher discounts when models are comparable."
                ),
                "criteria": criteria,
            }
        },
    }


async def main() -> None:
    payload = build_payload()
    print(
        f"=== REQUEST ({len(payload['questions']['target']['criteria'])} candidates) ==="
    )
    print(f"POST {URL}")
    print(f"task: {TASK}")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    print(f"=== RESPONSE (status {resp.status_code}) ===")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)


if __name__ == "__main__":
    asyncio.run(main())
