"""Sweep MAX_CANDIDATES against the live decision model.

The 5-candidate cap was tuned on a 16-model pool and thin criteria. The pool
is now larger and each candidate carries serving facts (latency, throughput,
uptime, endpoint name), so the cap needs re-measuring.

    uv run python scripts/probe_candidate_cap.py [--runs N] [--caps 3,5,8,12]

Reports, per cap: mean/median/min confidence and which models get chosen.
The useful cap is the smallest one that holds confidence up and still lets
the pick reflect capability rather than price.
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
from src.router import (  # noqa: E402
    _build_criteria,
    _decisions_base,
    _describe_model,
    _rank_candidates,
)

MODEL = os.getenv("ROUTER_MODEL", "typesafe/jev-1.13")
API_KEY = os.getenv("ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
API_BASE = os.getenv("ROUTER_API_BASE", "https://openrouter.ai/api/v1")
URL = f"{_decisions_base(API_BASE)}/api/alpha/decisions"

TASKS = [
    "Write a Python function that reverses a linked list. Use recursion.",
    "Summarize this earnings report and compare guidance to last quarter.",
    "Refactor this module to remove the duplicated validation logic.",
    "Explain in plain English what a database transaction isolation level does.",
    "Draft a short, firm email declining a vendor's renewal proposal.",
    "Find the bug in this regular expression that fails on nested brackets.",
    "Analyse this 200k-token log file and find the first anomaly.",
    "Look at this screenshot of the error and tell me what is wrong.",
]


async def ask(client: httpx.AsyncClient, task: str, cap: int):
    models = _rank_candidates(load_cache())[:cap]
    criteria = {m.id: _describe_model(m) for m in models}
    payload = {
        "model": MODEL,
        "state": {"request": task, "preference": "cost"},
        "questions": {
            "target": {
                "type": "choice",
                "instructions": (
                    "Read the request below, then choose which candidate model "
                    "should serve it. Match the candidate's strengths to what "
                    "the request actually needs. Tie-breaker: when candidates "
                    "are otherwise comparable, prefer the cheaper one "
                    "(lower effective price per million tokens). "
                    "Answer with one of the supplied options."
                ),
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
        return None, f"HTTP {resp.status_code}"
    try:
        a = resp.json()["answers"]["target"]
        return (float(a["confidence"]), a.get("choice")), None
    except (KeyError, TypeError, ValueError):
        return None, "unparsed"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--caps", default="3,5,8,12,20")
    args = ap.parse_args()

    if not API_KEY:
        print("no key")
        return

    pool = load_cache()
    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    caps = [c for c in caps if c <= len(pool)]
    print(f"pool: {len(pool)} models | caps: {caps} | runs: {args.runs}\n")

    results: dict[int, list[float]] = {c: [] for c in caps}
    picks: dict[int, list[str]] = {c: [] for c in caps}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for cap in caps:
            for _ in range(args.runs):
                for task in TASKS:
                    parsed, err = await ask(client, task, cap)
                    if err:
                        print(f"  cap {cap}: {err}")
                        continue
                    conf, chosen = parsed
                    results[cap].append(conf)
                    picks[cap].append(chosen or "?")
            v = results[cap]
            if v:
                print(
                    f"cap {cap:3d}  n={len(v):3d}  mean={statistics.mean(v):.3f}  "
                    f"median={statistics.median(v):.3f}  min={min(v):.3f}  "
                    f"max={max(v):.3f}  uniq_picks={len(set(picks[cap]))}"
                )

    print("\n=== chosen models per cap ===")
    for cap in caps:
        counts: dict[str, int] = {}
        for p in picks[cap]:
            counts[p] = counts.get(p, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print(
            f"cap {cap:3d}: "
            + ", ".join(f"{k.split('/')[-1]}({v})" for k, v in top[:5])
        )

    print("\n=== per-task confidence by cap ===")
    idx = 0
    for task in TASKS:
        line = f"{task[:38]:38s}"
        for cap in caps:
            vals = results[cap][idx * args.runs : (idx + 1) * args.runs]
            line += f"  {cap}:{statistics.mean(vals):.2f}" if vals else f"  {cap}: n/a"
        print(line)
        idx += 1


if __name__ == "__main__":
    asyncio.run(main())
