"""Probe C: Jev makes DECISIONS using weights as priorities, not a math test.

Give Jev real features (e.g. "89% off", "coding index 69", "1308ms latency")
and clear weighting priorities (e.g. "Prioritize: 1. capability, 2. discount, 3. latency"),
and ask it to select the best model for the specific task.

    uv run python scripts/probe_decision_weights.py
"""

import asyncio
import os
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inventory import load_cache
from src.router import _decisions_base, _rank_candidates, _describe_model

MODEL = os.getenv("ROUTER_MODEL", "typesafe/jev-1.13")
API_KEY = os.getenv("ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
API_BASE = os.getenv("ROUTER_API_BASE", "https://openrouter.ai/api/v1")
URL = f"{_decisions_base(API_BASE)}/api/alpha/decisions"

# Distinct tasks to see if Jev actually routes differently based on task + weights
TASKS = [
    ("code", "Write a Python function that reverses a linked list using recursion."),
    ("email", "Draft a short, firm email declining a vendor's renewal proposal."),
    (
        "sql_debug",
        "Find the bug in this complex PostgreSQL query with 4 recursive CTEs.",
    ),
    ("chat", "Say hello and give a one-sentence inspiring quote."),
]

# Weighting schemes expressed as decision priorities
PRIORITIES = {
    "discount_first": "Priority order: 1. Highest discount (cost savings), 2. Low latency, 3. Capability.",
    "capability_first": "Priority order: 1. Benchmark capability (coding/intelligence), 2. Low latency, 3. Discount.",
    "balanced": "Balance capability and discount equally. Do not pick a weak model for complex tasks just to save cost, but take the discount when tasks are simple.",
}


async def run_probe():
    models = _rank_candidates(load_cache())[:5]
    criteria = {m.id: _describe_model(m) for m in models}

    print(f"Candidates ({len(models)}):")
    for m in models:
        lat = f"{m.latency_p50_ms:.0f}ms" if m.latency_p50_ms else "n/a"
        print(
            f"  {m.id[:38]:38s} | disc: {m.discount_pct:2.0f}% | intel: {m.intelligence_index} | code: {m.coding_index} | {lat}"
        )
    print("=" * 80)

    async with httpx.AsyncClient(timeout=45.0) as client:
        for p_name, priority_text in PRIORITIES.items():
            print(f"\n--- Weight Policy: {p_name} ---")
            print(f"Instruction: {priority_text}")
            for t_type, prompt in TASKS:
                instructions = (
                    "Select the single best candidate model to serve this request.\n"
                    f"User evaluation criteria: {priority_text}\n"
                    "Evaluate the candidate features against the task requirements and the user's priority order. "
                    "Answer with one of the supplied options."
                )
                payload = {
                    "model": MODEL,
                    "state": {"request": prompt},
                    "questions": {
                        "target": {
                            "type": "choice",
                            "instructions": instructions,
                            "criteria": criteria,
                        }
                    },
                }
                resp = await client.post(
                    URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code != 200:
                    print(f"  [{t_type:10s}] Error {resp.status_code}")
                    continue
                a = resp.json()["answers"]["target"]
                choice = a.get("choice", "?").split("/")[-1]
                conf = a.get("confidence", 0.0)
                print(f"  [{t_type:10s}] -> {choice:25s} (conf: {conf:.2f})")


if __name__ == "__main__":
    asyncio.run(run_probe())
