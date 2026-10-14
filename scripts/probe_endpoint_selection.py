"""Compare endpoint-selection rules on the real endpoint data.

Reads /tmp/epdata/endpoints.json (dumped by the sibling probe) and reports,
per model, which endpoint each rule picks. Used to tune the weights before
they are wired into src/inventory.py.

    uv run python scripts/probe_endpoint_selection.py
"""

import json
import statistics
from pathlib import Path

DATA = Path("/tmp/epdata/endpoints.json")

MIN_REQUESTS = 100
MIN_UPTIME = 95.0


def facts(e: dict) -> dict | None:
    p = e.get("pricing") or {}
    wl = (e.get("perf_last_30m_by_workload") or {}).get("text_generation") or {}
    lat = (e.get("latency_last_30m") or {}).get("p50")
    tp = (e.get("throughput_last_30m") or {}).get("p50")
    return {
        "prov": e.get("provider_name") or "?",
        "disc": (p.get("discount") or 0.0),
        "pin": float(p.get("prompt") or 0) * 1e6,
        "pout": float(p.get("completion") or 0) * 1e6,
        "lat": lat,
        "tp": tp,
        "up": e.get("uptime_last_1d"),
        "reqs": wl.get("request_count", 0),
        "quant": e.get("quantization") or "?",
    }


def eligible(f: dict) -> bool:
    return bool(f["lat"] and f["reqs"] >= MIN_REQUESTS and (f["up"] or 0) >= MIN_UPTIME)


def load():
    raw = json.loads(DATA.read_text())
    out = {}
    for mid, eps in raw.items():
        fs = [f for f in (facts(e) for e in eps) if f]
        if fs:
            out[mid] = fs
    return out


def score(f: dict, w_disc: float, w_lat: float) -> float:
    """Higher is better. Latency normalized log-wise: p50 spread is 413ms-56s,
    so a linear scale lets one 56s endpoint dominate every comparison."""
    disc = f["disc"]
    # log1p keeps ordinal meaning while compressing the extreme tail.
    lat = f["lat"] or 1e6
    return w_disc * disc - w_lat * (lat / 1000.0) ** 0.5


def pick(fs: list[dict], w_disc: float, w_lat: float) -> dict:
    ok = [f for f in fs if eligible(f)]
    if not ok:
        # Never drop a model: fall back to max discount among all endpoints.
        return max(fs, key=lambda f: f["disc"])
    return max(ok, key=lambda f: score(f, w_disc, w_lat))


def main() -> None:
    data = load()
    rules = {
        "max_discount": None,
        "w_disc=1.0 lat=0.0": (1.0, 0.0),
        "w_disc=0.8 lat=0.2": (0.8, 0.2),
        "w_disc=0.6 lat=0.4": (0.6, 0.4),
        "w_disc=0.5 lat=0.5": (0.5, 0.5),
        "w_disc=0.4 lat=0.6": (0.4, 0.6),
    }

    picks: dict[str, dict[str, dict]] = {name: {} for name in rules}
    for mid, fs in data.items():
        picks["max_discount"][mid] = max(fs, key=lambda f: f["disc"])
        for name, w in rules.items():
            if w is None:
                continue
            picks[name][mid] = pick(fs, *w)

    print(f"models: {len(data)}  endpoints: {sum(len(v) for v in data.values())}\n")

    print(
        f"{'rule':22s} {'#differ':>8s} {'mean disc':>10s} {'med p50':>9s} {'mean p50':>10s} {'med tok/s':>10s}"
    )
    for name in rules:
        d_ = l_ = t_ = 0.0
        n = 0
        differ = 0
        lats, tps = [], []
        for mid, fs in data.items():
            f = picks[name][mid]
            d_ += f["disc"]
            lats.append(f["lat"] or 0)
            tps.append(f["tp"] or 0)
            n += 1
            if picks[name][mid]["prov"] != picks["max_discount"][mid]["prov"]:
                differ += 1
        print(
            f"{name:22s} {differ:8d} {d_/n*100:9.1f}% "
            f"{statistics.median(lats):8.0f} {statistics.mean(lats):9.0f} "
            f"{statistics.median(tps):9.0f}"
        )

    print("\n=== where the weighted rule differs from max_discount ===")
    # Use the middle setting as the representative.
    rep = "w_disc=0.8 lat=0.2"
    for mid in sorted(data):
        a, b = picks["max_discount"][mid], picks[rep][mid]
        if a["prov"] != b["prov"]:
            print(
                f"{mid[:40]:40s}\n"
                f"    max_discount : {a['prov'][:18]:18s} {a['disc']*100:3.0f}% "
                f"p50={a['lat'] or 0:6.0f}ms up={a['up'] or -1:3.0f}% reqs={a['reqs']}\n"
                f"    weighted     : {b['prov'][:18]:18s} {b['disc']*100:3.0f}% "
                f"p50={b['lat'] or 0:6.0f}ms up={b['up'] or -1:3.0f}% reqs={b['reqs']}"
            )


if __name__ == "__main__":
    main()
