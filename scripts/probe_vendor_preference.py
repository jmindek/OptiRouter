"""Probe: does a preferred-vendor preference help or hurt decision confidence?

Runs the same task set against several preference variants and reports the
decision model's confidence for each. This is the instrument that showed the
open/closed-weight preference failed, so it is the right one to test a vendor
preference before wiring it into the router.

    uv run python scripts/probe_vendor_preference.py [--runs N]

Variants:
  baseline        current "cost" preference (control)
  cost+vendor     cost tie-breaker plus a preferred-vendor sentence
  vendor_only     preferred vendors named, cost not mentioned
  pool_restrict   pool filtered to preferred vendors, then normal cost

pool_restrict predicts what a real pool-constraint implementation would do,
without needing that code to exist yet.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inventory import load_cache  # noqa: E402
from src.router import _build_criteria, _decisions_base, _rank_candidates  # noqa: E402

MODEL = os.getenv("ROUTER_MODEL", "typesafe/jev-1.13")
API_KEY = os.getenv("ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
API_BASE = os.getenv("ROUTER_API_BASE", "https://openrouter.ai/api/v1")
URL = f"{_decisions_base(API_BASE)}/api/alpha/decisions"

VENDORS = ["google", "openai", "moonshotai", "deepseek", "z-ai", "zai"]

TASKS = [
    "Write a Python function that reverses a linked list. Use recursion.",
    "Summarize this earnings report and compare guidance to last quarter.",
    "Refactor this module to remove the duplicated validation logic.",
    "Explain in plain English what a database transaction isolation level does.",
    "Draft a short, firm email declining a vendor's renewal proposal.",
    "Find the bug in this regular expression that fails on nested brackets.",
]

PREFERENCES = {
    "baseline": (
        "when candidates are otherwise comparable, prefer the cheaper one "
        "(lower effective price per million tokens)"
    ),
    "cost+vendor": (
        "when candidates are otherwise comparable, prefer the cheaper one; "
        "when still comparable, prefer models from Google, OpenAI, Moonshot, "
        "Z.ai, or DeepSeek"
    ),
    "vendor_only": (
        "when candidates are otherwise comparable, prefer models from "
        "Google, OpenAI, Moonshot, Z.ai, or DeepSeek"
    ),
    "pool_restrict": (
        "when candidates are otherwise comparable, prefer the cheaper one "
        "(lower effective price per million tokens)"
    ),
}


def vendor_of(model_id: str) -> str:
    return model_id.split("/")[0]


def build_payload(task: str, pref_text: str, restrict: bool) -> dict:
    models = load_cache()
    if restrict:
        models = [m for m in models if vendor_of(m.id) in VENDORS]
        models = _rank_candidates(models)
    criteria = _build_criteria(models)
    instructions = (
        "Read the request below, then choose which candidate model should "
        "serve it. Match the candidate's strengths to what the request "
        "actually needs."
    )
    if pref_text:
        instructions += f" Tie-breaker: {pref_text}."
    instructions += " Answer with one of the supplied options."
    return {
        "model": MODEL,
        "state": {"request": task, "preference": pref_text or "none"},
        "questions": {
            "target": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }


def extract_confidence(data: dict) -> tuple[float, str | None] | None:
    """Pull (confidence, chosen) out of the Decisions API response."""
    try:
        answers = data["answers"]["target"]
    except (KeyError, TypeError):
        return None
    if isinstance(answers, dict):
        conf = answers.get("confidence", answers.get("score"))
        chosen = answers.get("choice", answers.get("answer"))
        if conf is not None:
            return float(conf), chosen
        # Distribution form: {choice: prob}
        probs = answers.get("probabilities") or answers
        num = {k: v for k, v in probs.items() if isinstance(v, (int, float))}
        if num:
            best = max(num, key=num.get)
            return float(num[best]), best
    return None


async def one(client: httpx.AsyncClient, task: str, pref_text: str, restrict: bool):
    payload = build_payload(task, pref_text, restrict)
    resp = await client.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    parsed = extract_confidence(resp.json())
    if parsed is None:
        return None, "unparsed response shape"
    return parsed, None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    if not API_KEY:
        print("No API key (ROUTER_API_KEY / OPENROUTER_API_KEY)")
        return

    models = load_cache()
    if not models:
        print("No cached inventory. Run the refresh first.")
        return

    present = sorted({vendor_of(m.id) for m in models})
    print(f"pool: {len(models)} models, vendors: {', '.join(present)}")
    for v in VENDORS:
        n = sum(1 for m in models if vendor_of(m.id) == v)
        print(f"  {v}: {n} in pool{'  <-- ABSENT' if n == 0 else ''}")
    print()

    variants = [
        (k, PREFERENCES[k], k == "pool_restrict")
        for k in PREFERENCES
        if not args.only or k == args.only
    ]

    results: dict[str, list[float]] = {k: [] for k, _, _ in variants}
    picks: dict[str, list[str | None]] = {k: [] for k, _, _ in variants}

    async with httpx.AsyncClient(timeout=45.0) as client:
        for name, pref_text, restrict in variants:
            for run in range(args.runs):
                for task in TASKS:
                    parsed, err = await one(client, task, pref_text, restrict)
                    if err:
                        print(f"  {name}: {err}")
                        continue
                    conf, chosen = parsed
                    results[name].append(conf)
                    picks[name].append(chosen)
            vals = results[name]
            if vals:
                print(
                    f"{name:14s} n={len(vals):2d}  "
                    f"mean={statistics.mean(vals):.3f}  "
                    f"median={statistics.median(vals):.3f}  "
                    f"min={min(vals):.3f}  max={max(vals):.3f}"
                )
            else:
                print(f"{name:14s} no results")

    print()
    base = results.get("baseline") or []
    if base:
        bmean = statistics.mean(base)
        print(f"baseline mean {bmean:.3f}")
        for name, vals in results.items():
            if name == "baseline" or not vals:
                continue
            delta = statistics.mean(vals) - bmean
            verdict = (
                "raises" if delta > 0.02 else "lowers" if delta < -0.02 else "flat"
            )
            print(f"  {name:14s} {delta:+.3f}  {verdict} confidence")

    print()
    print("=== chosen models by variant ===")
    for name, chosen in picks.items():
        counts: dict[str, int] = {}
        for c in chosen:
            counts[c] = counts.get(c, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        print(f"{name:14s} " + ", ".join(f"{k}({v})" for k, v in top))
    print()
    print(
        f"preferred vendors actually present: {present and [v for v in VENDORS if v in present]}"
    )


if __name__ == "__main__":
    asyncio.run(main())
