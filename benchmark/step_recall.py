"""Step-level benchmark of AIM against mech-USPTO-31k gold steps.

For each gold step (state before the step + first AIM move of the converted step) ask:
  proposed  - does shared.rules generate the gold move as a candidate?
  accepted  - does the XGBoost filter keep it?
  rank      - where does it rank among all candidates by predicted probability?

python -m benchmark.step_recall [--limit 500] [--model v121]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from rdkit import Chem

from benchmark.mech_uspto import convert
from shared.ml_helper import aromatic_flags, make_fingerprint
from shared.rules import run_next_step

root_dir = Path(__file__).resolve().parents[1]


def aim_view_smiles(state) -> str:
    """Mapped SMILES of a converter state with H implicit again, as AIM would be given it."""
    params = Chem.RemoveHsParameters()
    params.removeMapped = False
    m = Chem.Mol(state.mol)
    Chem.SanitizeMol(m)
    m = Chem.RemoveHs(m, params)
    return Chem.MolToSmiles(m, canonical=False)


def lone_pair_equivalent(move: list) -> list | None:
    """Same step written with the source atom's lone pair instead of the pi bond it sits in.

    e.g. DMSO attacking as "S=O bond -> C" (dataset) vs "O lone pair -> C" (AIM never uses a
    pi bond as a source when its anchor atom has a lone pair). The products are resonance forms.
    """
    t, i, j, k, l = move
    return {2: [0, i, j, None, None], 3: [1, i, j, None, l], 5: [4, i, None, None, l]}.get(t)


def evaluate_step(smiles: str, gold_map: list, clf) -> dict:
    connected, mechs, mol_inp, _ = run_next_step(smiles, True, None, False, True)
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in mol_inp.GetAtoms() if a.GetAtomMapNum()}
    gold = [gold_map[0]] + [None if x is None else m2i[x] for x in gold_map[1:]]
    equivalent = False
    if gold not in mechs:
        alt = lone_pair_equivalent(gold)
        if alt in mechs:
            gold, equivalent = alt, True
    proposed = gold in mechs
    scored = [i for i, m in enumerate(mechs) if m[0] != 11]
    arom = aromatic_flags(mol_inp)
    fps = np.array([make_fingerprint(mol_inp, list(mechs[i]), arom) for i in scored], dtype=float)
    probs = clf.predict_proba(fps)[:, 1] if len(fps) else np.array([])
    result = {"proposed": proposed, "equivalent": equivalent, "n_candidates": len(mechs), "accepted": False, "rank": None, "p": None,
              "n_accepted": int((probs >= 0.5).sum())}
    if proposed:
        gi = mechs.index(gold)
        if gold[0] == 11:
            result.update(accepted=True, rank=None)
        else:
            p = probs[scored.index(gi)]
            result.update(accepted=bool(p >= 0.5), p=float(p), rank=int((probs > p).sum()) + 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root_dir / "data" / "mech-USPTO-31k.csv"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", default="v121")
    args = parser.parse_args()

    clf = xgboost.XGBClassifier()
    clf.load_model(str(root_dir / "models" / args.model / "xg_model.json"))
    df = pd.read_csv(args.data).sample(n=args.limit, random_state=0)

    tot = Counter()
    ranks = []
    by_type = defaultdict(Counter)
    by_class = defaultdict(Counter)
    t0 = time.time()
    for _, r in df.iterrows():
        c = convert(r.updated_reaction, r.mechanistic_label)
        if not (c.match_up_to_protons and c.fully_mapped):
            continue
        for s in c.steps:
            gold = s.aim_moves[0]
            try:
                res = evaluate_step(aim_view_smiles(s.start), gold, clf)
            except Exception:
                tot["aim_error"] += 1
                continue
            t = gold[0]
            tot["steps"] += 1
            by_type[t]["steps"] += 1
            by_class[r.mechanistic_class]["steps"] += 1
            tot["candidates"] += res["n_candidates"]
            tot["n_accepted"] += res["n_accepted"]
            if res["equivalent"]:
                tot["equivalent"] += 1
            if res["proposed"]:
                tot["proposed"] += 1
                by_type[t]["proposed"] += 1
                by_class[r.mechanistic_class]["proposed"] += 1
            if res["accepted"]:
                tot["accepted"] += 1
                by_type[t]["accepted"] += 1
                by_class[r.mechanistic_class]["accepted"] += 1
            if res["rank"] is not None:
                ranks.append(res["rank"])

    n = tot["steps"]
    print(f"{n} gold steps from {args.limit} reactions in {time.time() - t0:.0f}s (AIM errors: {tot['aim_error']})")
    print(f"  proposed by rules : {tot['proposed'] / n:.3f}  (of which as lone-pair equivalent {tot['equivalent'] / n:.3f})")
    print(f"  kept by {args.model:9s} : {tot['accepted'] / n:.3f}")
    print(f"  mean candidates/step {tot['candidates'] / n:.1f}, mean kept/step {tot['n_accepted'] / n:.1f}")
    if ranks:
        ranks = np.array(ranks)
        print(f"  rank of gold by p  : top1 {np.mean(ranks == 1):.3f}  top3 {np.mean(ranks <= 3):.3f}"
              f"  top5 {np.mean(ranks <= 5):.3f}  median {np.median(ranks):.0f}")
    print("\nby AIM move type (steps / proposed / kept):")
    for t in sorted(by_type):
        v = by_type[t]
        print(f"  {t:2d}  {v['steps']:5d}  {v['proposed'] / v['steps']:.2f}  {v['accepted'] / v['steps']:.2f}")
    print("\nworst classes by proposal rate (>= 15 steps):")
    rows = [(k, v["steps"], v["proposed"] / v["steps"], v["accepted"] / v["steps"])
            for k, v in by_class.items() if v["steps"] >= 15]
    for k, cnt, p, a in sorted(rows, key=lambda x: x[2])[:10]:
        print(f"  {k:52s} steps={cnt:4d} proposed={p:.2f} kept={a:.2f}")


if __name__ == "__main__":
    main()
