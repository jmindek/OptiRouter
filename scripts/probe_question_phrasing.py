"""Compare question phrasings for the routing objective.

The router asks Jev "which candidate is the best match?" with discount as a
tie-break. The project's goal is "the best discounted model for this task",
where discount is the objective and fitness is the constraint. Those are
different questions and can produce different answers.

This probe sends the same candidate sets under several phrasings and reports
confidence, choice diversity, and how strongly picks favor discount.

    uv run python scripts/probe_question_phrasing.py [--runs N]
"""

import argparse
import asyncio
import os
import statistics
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inventory import load_cache  # noqa: E402
from src.router import (  # noqa: E402
    _decisions_base,
    _describe_model,
    _rank_candidates,
)

MODEL = os.getenv("ROUTER_MODEL", "typesafe/jev-1.13")
API_KEY = os.getenv("ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
API_BASE = os.getenv("ROUTER_API_BASE", "https://openrouter.ai/api/v1")
URL = f"{_decisions_base(API_BASE)}/api/alpha/decisions"

CAP = 5
RUNS = 2

TASKS = [
    "Write a Python function that reverses a linked list. Use recursion.",
    "Summarize this earnings report and compare guidance to last quarter.",
    "Refactor this module to remove the duplicated validation logic.",
    "Explain in plain English what a database transaction isolation level does.",
    "Draft a short, firm email declining a vendor's renewal proposal.",
    "Find the bug in this regular expression that fails on nested brackets.",
]

PHRASINGS = {
    # Current production question.
    "A_match_then_cost": (
        "Read the request below, then choose which candidate model should "
        "serve it. Match the candidate's strengths to what the request "
        "actually needs. Tie-breaker: when candidates are otherwise "
        "comparable, prefer the cheaper one (lower effective price per "
        "million tokens). Answer with one of the supplied options."
    ),
    # Discount as the objective, fitness as a gate.
    "B_best_discount": (
        "Every candidate below can serve the request. Choose the one offering "
        "the best discount for the money. Prefer the deeper discount; treat "
        "latency, throughput and uptime as quality gates rather than goals, "
        "so rule out a candidate only if its serving stats are poor. "
        "Answer with one of the supplied options."
    ),
    # Explicit ordering: maximise discount subject to being able to serve.
    "C_max_discount_subject_to_fit": (
        "The request below must be served. From the candidates, choose the "
        "one with the largest discount that is still a good fit for the "
        "task. Maximise the discount; use the candidate's capabilities and "
        "serving stats only to disqualify poor fits, not to rank them. "
        "Answer with one of the supplied options."
    ),
    # Discount objective, but capability still allowed to rank.
    "D_balanced_discount_first": (
        "Choose the candidate that gives the best value: a deep discount on a "
        "model that genuinely suits this request. Weigh discount first, then "
        "capability, then serving speed. Answer with one of the supplied "
        "options."
    ),
}


async def ask(client, task, phrasing):
    models = _rank_candidates(load_cache())[:CAP]
    criteria = {m.id: _describe_model(m) for m in models}
    disc = {m.id: m.discount_pct for m in models}
    payload = {
        "model": MODEL,
        "state": {"request": task, "preference": "discount-first"},
        "questions": {
            "target": {
                "type": "choice",
                "instructions": phrasing,
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
        return (float(a["confidence"]), a.get("choice")), disc
    except (KeyError, TypeError, ValueError):
        return None, "unparsed"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    args = ap.parse_args()
    if not API_KEY:
        print("no key")
        return

    pool = load_cache()
    cand = _rank_candidates(pool)[:CAP]
    print(f"pool: {len(pool)} | candidates shown: {CAP} | runs: {args.runs}\n")
    for m in cand:
        print(f"  {m.id:42s} {m.discount_pct:3.0f}% off  ${m.input_price*1e6:.3f}/Mtok")
    print()

    results = {k: [] for k in PHRASINGS}
    picks = {k: [] for k in PHRASINGS}
    disc_of = {m.id: m.discount_pct for m in cand}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for name, phrasing in PHRASINGS.items():
            for _ in range(args.runs):
                for task in TASKS:
                    parsed, _ = await ask(client, task, phrasing)
                    if not isinstance(parsed, tuple):
                        print(f"  {name}: {_}")
                        continue
                    conf, chosen = parsed
                    results[name].append(conf)
                    picks[name].append(chosen or "?")
            v = results[name]
            if v:
                d = statistics.mean(disc_of.get(p, 0) for p in picks[name])
                print(
                    f"{name:32s} mean={statistics.mean(v):.3f}  "
                    f"med={statistics.median(v):.3f}  min={min(v):.3f}  "
                    f"uniq={len(set(picks[name])):2d}  mean_disc={d:4.1f}%"
                )

    print("\n=== picks per phrasing ===")
    for name in PHRASINGS:
        counts = {}
        for p in picks[name]:
            counts[p] = counts.get(p, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"{name}:")
        for k, v in top[:4]:
            print(f"    {k:42s} x{v}  ({disc_of.get(k, 0):.0f}% off)")

    print("\n=== per-task choice ===")
    for i, task in enumerate(TASKS):
        print(f"{task[:36]:36s}")
        for name in PHRASINGS:
            seg = picks[name][i * args.runs : (i + 1) * args.runs]
            cs = statistics.mean(results[name][i * args.runs : (i + 1) * args.runs])
            print(f"    {name:32s} {seg[0].split('/')[-1]:24s} conf={cs:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
