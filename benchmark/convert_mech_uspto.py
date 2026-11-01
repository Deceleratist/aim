"""Run the mech-USPTO-31k -> AIM converter over the dataset and report coverage.

python -m benchmark.convert_mech_uspto [--limit N] [--out converted.jsonl]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from benchmark.mech_uspto import convert

root_dir = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root_dir / "data" / "mech-USPTO-31k.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="write one JSON line per reaction")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)

    totals = Counter()
    move_types = Counter()
    arrows_per_step = Counter()
    moves_per_step = Counter()
    unmatched_patterns = Counter()
    errors = Counter()
    per_class = defaultdict(Counter)
    out = open(args.out, "w") if args.out else None
    t0 = time.time()

    for row_idx, r in df.iterrows():
        c = convert(r.updated_reaction, r.mechanistic_label)
        cls = per_class[r.mechanistic_class]
        totals["reactions"] += 1
        cls["reactions"] += 1
        if c.error:
            errors[c.error.split(":")[0]] += 1
            totals["error"] += 1
            cls["error"] += 1
        if c.final_match:
            totals["final_match"] += 1
            cls["final_match"] += 1
        if c.match_up_to_protons:
            totals["match_up_to_protons"] += 1
            cls["match_up_to_protons"] += 1
        if c.trailing:
            totals["trailing"] += 1
        if c.match_up_to_protons and c.fully_mapped:
            totals["usable"] += 1
            cls["usable"] += 1
        for s in c.steps:
            totals["steps"] += 1
            arrows_per_step[len(s.arrows)] += 1
            if s.unmatched:
                totals["steps_unmatched"] += 1
                unmatched_patterns[" ".join(a.kind for a in s.arrows)] += 1
            else:
                moves_per_step[len(s.aim_moves)] += 1
                if s.n_splits:
                    totals["steps_split"] += 1
                for m in s.aim_moves:
                    move_types[m[0]] += 1
        if out:
            out.write(json.dumps({
                "row": int(row_idx),
                "class": r.mechanistic_class,
                "final_match": c.final_match,
                "match_up_to_protons": c.match_up_to_protons,
                "fully_mapped": c.fully_mapped,
                "error": c.error,
                "steps": [{"n_arrows": len(s.arrows), "aim_moves": s.aim_moves,
                           "n_splits": s.n_splits, "unmatched": bool(s.unmatched),
                           "smiles": s.smiles} for s in c.steps],
            }) + "\n")

    if out:
        out.close()
    n = totals["reactions"]
    print(f"{n} reactions in {time.time() - t0:.0f}s")
    for key in ["final_match", "match_up_to_protons", "usable", "error", "trailing"]:
        print(f"  {key:12s} {totals[key]:6d}  ({100 * totals[key] / n:.1f}%)")
    print(f"  steps {totals['steps']}, unmatched {totals['steps_unmatched']}, needed splits {totals['steps_split']}")
    print("arrows per step:", dict(sorted(arrows_per_step.items())))
    print("AIM moves per mapped step:", dict(sorted(moves_per_step.items())))
    print("AIM move types:", dict(sorted(move_types.items())))
    print("errors:", errors.most_common(8))
    print("unmatched step patterns:", unmatched_patterns.most_common(8))
    print("\nworst classes by usable fraction (n >= 20):")
    rows = [(k, v["reactions"], v["usable"] / v["reactions"], v["final_match"] / v["reactions"])
            for k, v in per_class.items() if v["reactions"] >= 20]
    for k, cnt, usable, fm in sorted(rows, key=lambda x: x[2])[:12]:
        print(f"  {k:52s} n={cnt:5d} usable={usable:.2f} final_match={fm:.2f}")


if __name__ == "__main__":
    main()
