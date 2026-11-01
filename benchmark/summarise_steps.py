"""Summarise ranking_data steps_*.jsonl into the full step-level benchmark of v121 (plan A5).

python -m benchmark.summarise_steps --data <ranking_data out dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(recs: list[dict]) -> dict:
    ok = [r for r in recs if "error" not in r]
    ranked = [r["rank_v121"] for r in ok if r.get("rank_v121")]
    out = {"steps": len(recs), "errors": len(recs) - len(ok),
           "proposed": np.mean([r["proposed"] for r in ok]) if ok else np.nan,
           "as_equivalent": np.mean([r["equivalent"] for r in ok]) if ok else np.nan,
           "kept": np.mean([(r["p_gold"] or 0) >= 0.5 or r["gold_type11"] for r in ok]) if ok else np.nan,
           "cands": np.mean([r["n_candidates"] for r in ok]) if ok else np.nan,
           "n_kept": np.mean([r["n_kept"] for r in ok]) if ok else np.nan}
    # A8: reachable = gold move proposed (incl. its lone-pair resonance equivalent) or its product
    # state generated under another label; single-move steps only
    sp = [r["proposed"] or r["state_proposed"] for r in ok if "state_proposed" in r]
    out["state_prop"] = np.mean(sp) if sp else np.nan
    if ranked:
        rk = np.array(ranked)
        out.update(top1=np.mean(rk <= 1), top3=np.mean(rk <= 3), top5=np.mean(rk <= 5), median=float(np.median(rk)))
    return out


def fmt(name: str, s: dict, width: int = 34) -> str:
    base = (f"{name:{width}s} {s['steps']:7d} {s['errors']:5d} {s['proposed']:6.3f} {s['kept']:6.3f} "
            f"{s['cands']:7.0f} {s['n_kept']:6.1f} {s['state_prop']:6.3f}")
    if "top1" in s:
        base += f" {s['top1']:6.3f} {s['top3']:6.3f} {s['top5']:6.3f} {s['median']:6.0f}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    recs = [json.loads(line) for f in sorted(glob.glob(str(Path(args.data) / "steps_*.jsonl"))) for line in open(f)]
    header = f"{'':34s} {'steps':>7s} {'err':>5s} {'prop':>6s} {'kept':>6s} {'cands':>7s} {'nkept':>6s} {'reach':>6s} {'top1':>6s} {'top3':>6s} {'top5':>6s} {'med':>6s}"
    print("v121 on every mapped gold step of mech-USPTO-31k (rank among all of AIM's candidates)\n")
    print(header)
    print(fmt("ALL", stats(recs)))
    groups = defaultdict(list)
    for r in recs:
        groups["split=" + r["split"]].append(r)
    for k in sorted(groups):
        print(fmt(k, stats(groups[k])))
    print("\nby gold move type")
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["gold_type"]].append(r)
    for t in sorted(by_type):
        print(fmt(f"type {t}", stats(by_type[t])))
    print("\nby reaction class (>= 200 steps), sorted by top-3")
    by_class = defaultdict(list)
    for r in recs:
        by_class[r["class"]].append(r)
    rows = [(c, stats(v)) for c, v in by_class.items() if len(v) >= 200]
    for c, s in sorted(rows, key=lambda x: x[1].get("top3", 0)):
        print(fmt(c[:34], s))


if __name__ == "__main__":
    main()
