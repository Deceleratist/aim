"""Rank the gold move under several saved rankers on stored candidate sets (plan E1c).

Reads evalfull_*.npz + steps_*.jsonl from a ranking_data output dir and reports, per model, the
gold move's rank among all AIM candidates: overall and by step number within the reaction, so a
single failing step (e.g. Mitsunobu step 1, PPh3 adding to DEAD) is visible. v121 uses the
probabilities stored at data-building time.

python -m benchmark.eval_rankers --data <dir> --model rank_v1=models/rank_v1/ranker.json --model e1c=<path> [--out results.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import xgboost

from benchmark.ranking_data import decode
from benchmark.train_ranker import gold_ranks, load_eval, load_steps, summarise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", action="append", default=[], help="name=path to an XGBRanker json")
    parser.add_argument("--by", choices=["step", "class", "split"], default="step",
                        help="group results by step number, reaction class or split")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data)

    X8, is_gold, qid, p_v121 = load_eval(data_dir)
    X = decode(X8)
    scores = {"v121": p_v121}
    for spec in args.model:
        name, path = spec.split("=", 1)
        m = xgboost.XGBRanker()
        m.load_model(path)
        scores[name] = m.predict(X)
    steps = load_steps(data_dir)
    qs = list(dict.fromkeys(qid))
    by_step = defaultdict(list)
    for q in qs:
        key = steps[q]["step"] + 1 if args.by == "step" else steps[q][args.by]
        by_step[key].append(q)
    n_rows = len({steps[q]["row"] for q in qs})
    print(f"{len(qs)} steps from {n_rows} reactions; top-k = gold among the k best of all candidates "
          f"(ties pessimistic); top10 = survives the beam's per-state cut")

    results = {"n_steps": len(qs), "n_reactions": n_rows, "overall": {}, "by_step": {}}
    names = list(scores)
    for name in names:
        ranks = gold_ranks(scores[name], is_gold, qid)
        results["overall"][name] = summarise(ranks, qs)
        for k, kq in sorted(by_step.items()):
            results["by_step"].setdefault(str(k), {})[name] = summarise(ranks, kq)

    head = f"  {'model':10s} {'n':>5s} {'top1':>6s} {'top3':>6s} {'top10':>6s} {'median':>7s}"
    for title, block in [("all steps", results["overall"])] + \
            [(f"{args.by} {k}", v) for k, v in results["by_step"].items()]:
        print(f"\n[{title}]\n{head}")
        for name in names:
            r = block[name]
            if r["n"]:
                print(f"  {name:10s} {r['n']:5d} {r['top1']:6.3f} {r['top3']:6.3f} {r['top10']:6.3f} {r['median_rank']:7.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
