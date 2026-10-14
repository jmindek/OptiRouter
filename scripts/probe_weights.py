"""Probe: do numeric criteria + user weights make Jev pick according to the weights?

Sends candidates as numeric feature vectors (normalised in Python, option (b))
plus an explicit weighted-sum instruction, then checks whether the chosen
candidate is the one the weights actually favour.

This is the untested configuration. The earlier vendor probe added prose to
the instruction and left criteria as text, and confidence dropped. Here the
features are numbers and the arithmetic is unambiguous.

    uv run python scripts/probe_weights.py [--runs N]

Reports per weight profile: chosen model, whether it matches the arithmetic
argmax computed in Python, and the decision confidence.
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
from src.router import _decisions_base, _rank_candidates  # noqa: E402

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
    "Draft a short, firm email declining a vendor's renewal proposal.",
]

# Per-lab scores, so "lab" can carry a weight like any other feature.
LAB_SCORE = {
    "google": 1.0,
    "openai": 1.0,
    "anthropic": 1.0,
    "moonshotai": 0.8,
    "z-ai": 0.7,
    "deepseek": 0.7,
    "mistralai": 0.7,
    "qwen": 0.6,
    "upstage": 0.5,
    "inclusionai": 0.5,
    "minimax": 0.5,
    "xiaomi": 0.4,
    "nvidia": 0.4,
    "meta-llama": 0.4,
    "inception": 0.3,
    "meituan": 0.3,
    "poolside": 0.3,
}

PROFILES = {
    "P1_discount_heavy": {
        "discount": 0.50,
        "intelligence": 0.20,
        "latency": 0.10,
        "coding": 0.10,
        "uptime": 0.10,
    },
    "P2_capability_heavy": {
        "intelligence": 0.40,
        "discount": 0.20,
        "coding": 0.20,
        "latency": 0.10,
        "uptime": 0.10,
    },
    "P3_latency_heavy": {
        "latency": 0.40,
        "discount": 0.25,
        "intelligence": 0.20,
        "coding": 0.10,
        "uptime": 0.05,
    },
    "P4_balanced": {
        "discount": 0.20,
        "intelligence": 0.20,
        "coding": 0.20,
        "latency": 0.20,
        "uptime": 0.20,
    },
    "P5_lab_heavy": {
        "lab": 0.40,
        "discount": 0.25,
        "intelligence": 0.20,
        "coding": 0.10,
        "uptime": 0.05,
    },
}


def feature_vector(m) -> dict:
    return {
        "discount": float(m.discount_pct),
        "intelligence": float(m.intelligence_index or 0.0),
        "coding": float(m.coding_index or 0.0),
        "latency": float(m.latency_p50_ms or 0.0),
        "throughput": float(m.throughput_p50 or 0.0),
        "uptime": float(m.uptime_pct or 0.0),
        "lab": float(LAB_SCORE.get(m.id.split("/")[0], 0.5)),
    }


def normalise(vectors: dict[str, dict], weights: dict[str, float]) -> dict[str, dict]:
    """Min-max each weighted feature across candidates. Latency is inverted:
    lower is better, so the scale runs max->min instead of min->max."""
    keys = [k for k in weights if k in next(iter(vectors.values()))]
    out = {cid: {} for cid in vectors}
    for k in keys:
        vals = [v[k] for v in vectors.values()]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        for cid, v in vectors.items():
            if span == 0:
                out[cid][k] = 0.5  # no spread: neutral
            elif k == "latency":
                out[cid][k] = (hi - v[k]) / span
            else:
                out[cid][k] = (v[k] - lo) / span
    return out


def python_scores(norm: dict[str, dict], weights: dict[str, float]) -> dict[str, float]:
    return {
        cid: sum(weights.get(k, 0.0) * v for k, v in feats.items())
        for cid, feats in norm.items()
    }


async def ask(client, task, weights, cid_for, limit):
    models = cid_for["models"]
    raw = {m.id: feature_vector(m) for m in models}
    norm = normalise(raw, weights)
    scores = python_scores(norm, weights)

    criteria = {}
    for m in models:
        f = norm[m.id]
        criteria[m.id] = {
            "name": m.name or m.id,
            "normalised_features": {k: round(v, 3) for k, v in f.items()},
            "raw": {
                "discount_pct": round(m.discount_pct),
                "intelligence": m.intelligence_index,
                "coding": m.coding_index,
                "latency_p50_ms": round(m.latency_p50_ms or 0),
                "uptime_pct": m.uptime_pct,
            },
            "python_score": round(scores[m.id], 4),
        }

    weight_line = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
    instructions = (
        "Each candidate lists normalised_features (0=worst, 1=best across the "
        "candidates) and a python_score, which is the weighted sum of those "
        f"features using these weights: {weight_line}. The score is already "
        "computed and correct. Choose the candidate with the highest "
        "python_score, unless it is clearly a poor fit for the request, in "
        "which case choose the highest-scoring candidate that is a good fit. "
        "Answer with one of the supplied options."
    )

    payload = {
        "model": MODEL,
        "state": {
            "request": task,
            "weights": weights,
            "note": "prefer the highest python_score",
        },
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
        return None, f"HTTP {resp.status_code}: {resp.text[:150]}"
    try:
        a = resp.json()["answers"]["target"]
        return (float(a["confidence"]), a.get("choice")), scores
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
    models = _rank_candidates(pool)[:CAP]
    print(f"pool: {len(pool)} | shown: {CAP} | runs: {args.runs}\n")

    print("candidates:")
    for m in models:
        print(
            f"  {m.id[:40]:40s} disc={m.discount_pct:3.0f}% intel={m.intelligence_index} "
            f"code={m.coding_index} lat={m.latency_p50_ms and round(m.latency_p50_ms)}ms"
        )
    print()

    holder = {"models": models}

    rows = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for name, weights in PROFILES.items():
            agree = 0
            total = 0
            confs = []
            for _ in range(args.runs):
                for task in TASKS:
                    parsed, scores = await ask(client, task, weights, holder, CAP)
                    if not isinstance(parsed, tuple):
                        print(f"  {name}: {scores}")
                        continue
                    conf, chosen = parsed
                    argmax = max(scores, key=scores.get)
                    total += 1
                    confs.append(conf)
                    if chosen == argmax:
                        agree += 1
                    rows.append((name, task, chosen, argmax, conf, scores))
            norms = normalise({m.id: feature_vector(m) for m in models}, weights)
            sc = python_scores(norms, weights)
            top = max(sc, key=sc.get)
            print(
                f"{name:22s} agree={agree}/{total}  "
                f"mean_conf={statistics.mean(confs) if confs else 0:.3f}  "
                f"argmax={top.split('/')[-1]}"
            )

    print("\n=== where Jev disagreed with the arithmetic ===")
    for name, task, chosen, argmax, conf, scores in rows:
        if chosen != argmax:
            print(f"  {name} | {task[:40]}")
            print(f"      jev  : {chosen}  (conf {conf:.2f})")
            print(f"      argmax: {argmax}")
            top2 = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
            print(
                f"      scores: "
                + ", ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in top2)
            )

    print("\n=== python argmax per profile (normalised) ===")
    for name, weights in PROFILES.items():
        norms = normalise({m.id: feature_vector(m) for m in models}, weights)
        s = python_scores(norms, weights)
        order = sorted(s.items(), key=lambda kv: -kv[1])
        print(
            f"{name}: "
            + " > ".join(f"{k.split('/')[-1]}({v:.2f})" for k, v in order[:3])
        )


if __name__ == "__main__":
    asyncio.run(main())
