"""Train move-ranking models on ranking_data output and compare them with v121 (plan B1).

Models, all scored on exactly the same held-out candidate sets (evalfull_*.npz):
  v121     Richard's binary filter, as shipped
  clf_new  binary XGBoost classifier trained on the mech-USPTO steps (effect of more data)
  ranker   XGBoost rank:pairwise trained on the same groups (effect of the ranking objective)

Metric: rank of the gold (expert) move among all of AIM's candidates for that step.
top-k = gold is within the k best-scored candidates. Top-3 is the headline, since a few
plausible moves may legitimately compete.

python -m benchmark.train_ranker --data <ranking_data out dir> --out <dir> [--threads 16]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost

from benchmark.ranking_data import _hash01, decode


def load_train(data_dir: Path):
    xs, ys, qs = [], [], []
    for f in sorted(glob.glob(str(data_dir / "train_*.npz"))):
        d = np.load(f)
        if len(d["y"]):
            xs.append(d["X"]); ys.append(d["y"]); qs.append(d["qid"])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(qs)


def load_eval(data_dir: Path):
    xs, gs, qs, ps = [], [], [], []
    for f in sorted(glob.glob(str(data_dir / "evalfull_*.npz"))):
        d = np.load(f)
        if len(d["qid"]):
            xs.append(d["X"]); gs.append(d["is_gold"]); qs.append(d["qid"]); ps.append(d["p_v121"])
    return np.concatenate(xs), np.concatenate(gs), np.concatenate(qs), np.concatenate(ps)


def load_steps(data_dir: Path) -> dict:
    steps = {}
    for f in glob.glob(str(data_dir / "steps_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            steps[r["qid"]] = r
    return steps


def group_ids(qid: np.ndarray) -> np.ndarray:
    """Contiguous integer ids for a qid array whose groups are already contiguous."""
    change = np.r_[True, qid[1:] != qid[:-1]]
    return np.cumsum(change) - 1


def gold_ranks(scores: np.ndarray, is_gold: np.ndarray, qid: np.ndarray) -> dict:
    """qid -> 1-based rank of the gold candidate (ties counted pessimistically)."""
    ranks = {}
    gid = group_ids(qid)
    starts = np.r_[0, np.flatnonzero(np.diff(gid)) + 1, len(gid)]
    for a, b in zip(starts[:-1], starts[1:]):
        s, g = scores[a:b], is_gold[a:b]
        if g.any():
            gs = s[g][0]
            ranks[qid[a]] = int((s > gs).sum() + (s == gs).sum())   # pessimistic tie handling
    return ranks


def summarise(ranks: dict, qids) -> dict:
    r = np.array([ranks[q] for q in qids if q in ranks])
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "top1": float(np.mean(r <= 1)), "top3": float(np.mean(r <= 3)),
            "top5": float(np.mean(r <= 5)), "top10": float(np.mean(r <= 10)),
            "median_rank": float(np.median(r)), "mrr": float(np.mean(1 / r))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n_estimators", type=int, default=600)
    args = parser.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X8, y, qid = load_train(data_dir)
    X = decode(X8)
    del X8
    # 10 % of training reactions for early stopping (by reaction, so a step never straddles)
    is_val = np.array([_hash01("val", q.split(":")[0]) < 0.10 for q in qid])
    print(f"train rows {len(y)} ({y.sum()} gold), groups {len(set(qid))}, val rows {is_val.sum()}; "
          f"load {time.time() - t0:.0f}s", flush=True)

    common = dict(n_estimators=args.n_estimators, max_depth=8, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=args.threads, early_stopping_rounds=30)
    tr, va = ~is_val, is_val

    clf = xgboost.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=50)
    clf.save_model(out / "clf_new.json")
    print(f"clf_new done, best iteration {clf.best_iteration}, {time.time() - t0:.0f}s", flush=True)

    ranker = xgboost.XGBRanker(objective="rank:pairwise", eval_metric="ndcg@3", **common)
    ranker.fit(X[tr], y[tr], qid=group_ids(qid[tr]), eval_set=[(X[va], y[va])], eval_qid=[group_ids(qid[va])], verbose=50)
    ranker.save_model(out / "ranker.json")
    print(f"ranker done, best iteration {ranker.best_iteration}, {time.time() - t0:.0f}s", flush=True)
    del X

    Xe8, is_gold, qe, p_v121 = load_eval(data_dir)
    Xe = decode(Xe8)
    del Xe8
    scores = {"v121": p_v121, "clf_new": clf.predict_proba(Xe)[:, 1], "ranker": ranker.predict(Xe)}
    steps = load_steps(data_dir)
    eval_q = list(dict.fromkeys(qe))
    by_split = defaultdict(list)
    by_class = defaultdict(list)
    for q in eval_q:
        by_split[steps[q]["split"]].append(q)
        if steps[q]["split"] == "test_class":
            by_class[steps[q]["class"]].append(q)

    results = {"n_eval_steps": len(eval_q), "splits": {}, "held_out_classes": {}}
    for name, s in scores.items():
        ranks = gold_ranks(s, is_gold, qe)
        for split, qs in by_split.items():
            results["splits"].setdefault(split, {})[name] = summarise(ranks, qs)
        for cls, qs in by_class.items():
            results["held_out_classes"].setdefault(cls, {})[name] = summarise(ranks, qs)
    # how many candidates does each model need to cover the gold move in 90 % of steps?
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nheld-out evaluation on {len(eval_q)} steps (all AIM candidates per step)")
    for split, res in results["splits"].items():
        print(f"\n[{split}]  n = {next(iter(res.values()))['n']}")
        print(f"  {'model':8s} {'top1':>6s} {'top3':>6s} {'top5':>6s} {'top10':>6s} {'median':>7s} {'MRR':>6s}")
        for name, m in res.items():
            print(f"  {name:8s} {m['top1']:6.3f} {m['top3']:6.3f} {m['top5']:6.3f} {m['top10']:6.3f} {m['median_rank']:7.1f} {m['mrr']:6.3f}")
    print("\n[test_class] top-3 by held-out class:")
    for cls, res in sorted(results["held_out_classes"].items()):
        print(f"  {cls:24s} n={res['v121']['n']:4d}  " + "  ".join(f"{k} {v['top3']:.2f}" for k, v in res.items()))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
