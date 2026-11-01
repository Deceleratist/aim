"""Summarise path_benchmark output (plan A7/B2): per scorer and split, how often the beam search
reaches the product and how often the expert route is its first / among its top-3 mechanisms.

python -m benchmark.summarise_paths --data <path_benchmark out dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    recs = [json.loads(line) for f in glob.glob(str(Path(args.data) / "paths_*.jsonl")) for line in open(f)]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["scorer"], "all")].append(r)
        groups[(r["scorer"], r["split"])].append(r)
    cols = ["full_top1", "full_top3", "org_top1", "org_top3", "gold_top1", "gold_top3"]
    print(f"{'scorer':8s} {'split':12s} {'n':>5s} {'skip':>5s} {'err':>4s} {'found':>6s} "
          + " ".join(f"{c.replace('_top', '@'):>7s}" for c in cols) + f" {'sec':>6s}")
    for (scorer, split), rs in sorted(groups.items()):
        ok = [r for r in rs if "found" in r]
        n = len(ok) or 1
        print(f"{scorer:8s} {split:12s} {len(ok):5d} {sum('skipped' in r for r in rs):5d} {sum('error' in r for r in rs):4d} "
              f"{sum(r['found'] for r in ok) / n:6.3f} "
              + " ".join(f"{sum(r.get(c, False) for r in ok) / n:7.3f}" for c in cols)
              + f" {sum(r['seconds'] for r in ok) / n:6.1f}")
    # per class, both scorers side by side (classes with >= 8 reactions)
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if "found" in r:
            by[r["class"]][r["scorer"]].append(r)
    print("\nper class: found / full@3 / gold@3 (n)")
    for cls, d in sorted(by.items(), key=lambda x: -sum(len(v) for v in x[1].values())):
        if max(len(v) for v in d.values()) < 8:
            continue
        print(f"  {cls[:40]:40s} " + "  ".join(
            f"{s} {sum(r['found'] for r in v) / len(v):.2f}/{sum(r.get('full_top3', False) for r in v) / len(v):.2f}"
            f"/{sum(r['gold_top3'] for r in v) / len(v):.2f} ({len(v)})" for s, v in sorted(d.items())))


if __name__ == "__main__":
    main()
