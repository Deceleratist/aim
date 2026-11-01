"""Recompute the path ladder (full / org / gold) for saved path_benchmark records with the current
matcher (norm_key), without re-running the search. 'found' is unchanged: it uses the product
fragments only. Prints old vs new per class.

python -m benchmark.rescore_paths <dir with paths_*.jsonl> [more dirs]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import pandas as pd
from rdkit import RDLogger

from benchmark.path_benchmark import follows, gold_route, has_carbon, norm_frags, norm_key

RDLogger.DisableLog("rdApp.*")
COLS = ["found", "org_top3", "full_top3", "gold_top3"]


def rescore(rec: dict, row) -> dict:
    _, _, gold_states, _, full = gold_route(row.updated_reaction, row.mechanistic_label)
    org = {k for k in full if has_carbon(k)}
    ends = [norm_frags(p["states"][-1]) for p in rec["paths"]]
    walks = [[norm_key(s) for s in p["states"]] for p in rec["paths"]]
    return {"found": rec["found"], "org_top3": any(org <= e for e in ends),
            "full_top3": any(full <= e for e in ends), "gold_top3": any(follows(gold_states, w) for w in walks)}


def main() -> None:
    df = pd.read_csv("data/mech-USPTO-31k.csv")
    for d in sys.argv[1:]:
        agg = defaultdict(lambda: defaultdict(float))
        for f in sorted(glob.glob(d + "/paths_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if "found" not in r:
                    continue          # skipped or error
                new = rescore(r, df.iloc[r["row"]]) if r["found"] else {c: False for c in COLS}
                for key in (f"{r['scorer']} all", f"{r['scorer']} {r['class']}"):
                    a = agg[key]
                    a["n"] += 1
                    for c in COLS:
                        a["old_" + c] += bool(r.get(c))
                        a["new_" + c] += new[c]
        print(f"\n{d}\n  {'scorer class':36s} {'n':>4s}  found   org@3 old→new   full@3 old→new   gold@3 old→new")
        for key in sorted(agg, key=lambda k: (k.split()[0], not k.endswith(" all"), k)):
            a, n = agg[key], agg[key]["n"]
            cells = "   ".join(f"{a['old_' + c] / n:.3f}→{a['new_' + c] / n:.3f}" for c in COLS[1:])
            print(f"  {key:36s} {int(n):4d}  {a['new_found'] / n:.3f}   {cells}")


if __name__ == "__main__":
    main()
