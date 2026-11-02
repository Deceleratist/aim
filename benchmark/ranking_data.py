"""Build per-step benchmark records and ranking training data from mech-USPTO-31k.

For every mapped gold step, AIM is put in the gold state before the step and generates all its
candidate moves. One pass gives:
  steps_<chunk>.jsonl   one record per step: was the gold move proposed, its v121 probability and
                        rank among all candidates, how many v121 keeps (plan A5)
  train_<chunk>.npz     gold fingerprint + up to K random wrong candidates per step, for training
                        a ranking model (plan B1). Only steps in the 'train' split.
  evalfull_<chunk>.npz  every candidate's fingerprint for a fixed sample of test steps, so a new
                        model can be ranked against v121 on exactly the same candidates.

Fingerprints are stored as uint8 (0, 1, 255 = NaN/unused).
Splits (fixed, reproducible): 'test_class' = whole held-out reaction classes; 'test_random' =
10 % of the remaining reactions (by row hash); 'train' = the rest.
AIM_TRAIN_HALF_OF=<class>[,<class>] (plan E1c) moves half of each named held-out class's reactions (by row hash)
into 'train'; the other half stays 'test_class'. Every tool that calls split_of follows it.

python -m benchmark.ranking_data --chunk 3 --nchunks 128 --out <dir>
python -m benchmark.ranking_data --chunk 0 --nchunks 8 --only_class Mitsunobu --evalfull_fraction 1 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from benchmark.mech_uspto import convert
from rdkit import Chem

from benchmark.step_recall import aim_view_smiles, lone_pair_equivalent, state_norm
from shared.ml_helper import aromatic_flags, make_fingerprint
from shared.rules import run_next_step

root_dir = Path(__file__).resolve().parents[1]

# chemically varied classes, ~12 % of reactions; never seen in training
HELD_OUT_CLASSES = {
    "reductive_amination", "SNAr(para)", "Mitsunobu", "Swern_oxidation",
    "aldol_condensation", "Grignard", "Wittig", "nitrile_reduction",
}
N_NEGATIVES = 50          # wrong candidates kept per training step
EVALFULL_FRACTION = 0.25  # share of test steps whose full candidate set is stored


def _hash01(*parts) -> float:
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_of(row: int, cls: str) -> str:
    if cls in HELD_OUT_CLASSES:
        if cls in os.environ.get("AIM_TRAIN_HALF_OF", "").split(",") and _hash01("half", row) < 0.5:
            return "train"
        return "test_class"
    return "test_random" if _hash01("split", row) < 0.10 else "train"


def encode(fps: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(fps), 255, fps).astype(np.uint8)


def decode(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32)
    out[x == 255] = np.nan
    return out


def process_step(smiles, gold_map, clf, gold_next=None):
    """gold_next: SMILES of the gold state after this step, given only when the step is a single
    AIM move. Then 'state_proposed' records whether any candidate reaches that state, whatever its
    move label (clean_mol keeps one label per product, so a reachable step can be listed under an
    equivalent move, e.g. silicate ligand loss as a degenerate type 3; plan A8)."""
    connected, mechs, mol, _ = run_next_step(smiles, True, None, False, True)
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    gold = [gold_map[0]] + [None if x is None else m2i[x] for x in gold_map[1:]]
    equivalent = False
    if gold not in mechs:
        alt = lone_pair_equivalent(gold)
        if alt in mechs:
            gold, equivalent = alt, True
    scored = [list(m) for m in mechs if m[0] != 11]
    arom = aromatic_flags(mol)
    fps = np.array([make_fingerprint(mol, list(m), arom) for m in scored], dtype=float) if scored else np.zeros((0, 182))
    probs = clf.predict_proba(fps)[:, 1] if len(fps) else np.array([])
    gi = scored.index(gold) if gold in scored else -1
    rec = {"proposed": gold in mechs, "equivalent": equivalent, "gold_type11": gold[0] == 11,
           "n_candidates": len(scored), "n_kept": int((probs >= 0.5).sum()), "gold_index": gi,
           "p_gold": float(probs[gi]) if gi >= 0 else None,
           "rank_v121": int((probs > probs[gi]).sum()) + 1 if gi >= 0 else None}
    if gold_next is not None:
        target = state_norm(gold_next)
        hits = [k for k, m in enumerate(connected) if state_norm(Chem.MolToSmiles(m)) == target]
        rec.update(state_proposed=bool(hits), state_types=sorted({mechs[k][0] for k in hits}))
    return rec, fps, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root_dir / "data" / "mech-USPTO-31k.csv"))
    parser.add_argument("--model", default="v121")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--nchunks", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="max reactions in this chunk (testing)")
    parser.add_argument("--only_class", default=None, help="process only reactions of this class")
    parser.add_argument("--evalfull_fraction", type=float, default=EVALFULL_FRACTION)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clf = xgboost.XGBClassifier()
    clf.load_model(str(root_dir / "models" / args.model / "xg_model.json"))
    df = pd.read_csv(args.data)
    rows = [i for i in range(len(df)) if args.only_class in (None, df.mechanistic_class[i])]
    rows = [i for j, i in enumerate(rows) if j % args.nchunks == args.chunk][: args.limit]
    rng = np.random.default_rng(args.chunk)

    train_x, train_y, train_q = [], [], []
    eval_x, eval_q, eval_gold, eval_p = [], [], [], []
    t0 = time.time()
    n_steps = last_report = 0
    with open(out / f"steps_{args.chunk}.jsonl", "w") as fsteps:
        for row in rows:
            r = df.iloc[row]
            split = split_of(row, r.mechanistic_class)
            c = convert(r.updated_reaction, r.mechanistic_label)
            if c.error or not c.match_up_to_protons:
                continue
            for k, s in enumerate(c.steps):
                if not s.aim_moves:
                    continue   # unmatched step (e.g. bare H+ release): nothing for AIM to propose
                qid = f"{row}:{k}"
                base = {"qid": qid, "row": row, "step": k, "class": r.mechanistic_class, "split": split,
                        "gold_move_map": s.aim_moves[0], "gold_type": s.aim_moves[0][0]}
                try:
                    smiles = aim_view_smiles(s.start)
                    rec, fps, probs = process_step(smiles, s.aim_moves[0], clf,
                                                   s.smiles if len(s.aim_moves) == 1 else None)
                except Exception as e:
                    fsteps.write(json.dumps({**base, "error": f"{type(e).__name__}: {e}"}) + "\n")
                    continue
                fsteps.write(json.dumps({**base, **rec, "smiles": smiles}) + "\n")
                n_steps += 1
                gi = rec["gold_index"]
                if gi < 0:
                    continue
                if split == "train":
                    neg = [i for i in range(len(fps)) if i != gi]
                    if len(neg) > N_NEGATIVES:
                        neg = list(rng.choice(neg, N_NEGATIVES, replace=False))
                    idx = [gi] + neg
                    train_x.append(encode(fps[idx]))
                    train_y += [1] + [0] * len(neg)
                    train_q += [qid] * len(idx)
                elif _hash01("evalfull", qid) < args.evalfull_fraction:
                    eval_x.append(encode(fps))
                    eval_gold += [i == gi for i in range(len(fps))]
                    eval_q += [qid] * len(fps)
                    eval_p.append(probs.astype(np.float32))
            if n_steps // 200 > last_report:
                last_report = n_steps // 200
                print(f"chunk {args.chunk}: row {row}, {n_steps} steps, {time.time() - t0:.0f}s", flush=True)

    width = 182
    np.savez_compressed(out / f"train_{args.chunk}.npz",
                        X=np.concatenate(train_x) if train_x else np.zeros((0, width), np.uint8),
                        y=np.array(train_y, np.int8), qid=np.array(train_q))
    np.savez_compressed(out / f"evalfull_{args.chunk}.npz",
                        X=np.concatenate(eval_x) if eval_x else np.zeros((0, width), np.uint8),
                        is_gold=np.array(eval_gold, bool), qid=np.array(eval_q),
                        p_v121=np.concatenate(eval_p) if eval_p else np.zeros(0, np.float32))
    print(f"chunk {args.chunk} done: {len(rows)} reactions, {n_steps} steps, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
