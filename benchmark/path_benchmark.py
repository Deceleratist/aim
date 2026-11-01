"""Path-level benchmark (plan A7): beam search from reactants to the gold end state.

For each test reaction: start from the reactants, target = the molecules of the gold final state
that contain the recorded product's atoms (the converter's end state, so labels that stop before
the last proton transfer are still a fair target). Then report, per scorer:
  found      - a complete mechanism reaches the target within the depth limit
  gold_top1  - the best-scored mechanism passes through every gold intermediate, in order
  gold_top3  - one of the top-3 mechanisms does

python -m benchmark.path_benchmark --scorer v121 --chunk 0 --nchunks 50 --out <dir> [--ranker path.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "SPArrow"))

from beam_search import Scorer, beam_search, state_key  # noqa: E402

from benchmark.mech_uspto import ArrowState, parse_label, segment, to_aim_moves  # noqa: E402
from benchmark.ranking_data import _hash01, split_of  # noqa: E402
from benchmark.step_recall import aim_view_smiles  # noqa: E402


def gold_route(updated_reaction: str, label: str):
    """(start SMILES, target fragment keys, gold intermediate keys, number of AIM moves) or None."""
    reactants, product = updated_reaction.split(">>")
    state = ArrowState(reactants)
    steps, trailing = segment(state, parse_label(label))
    if trailing or not state.matches(product, ignore_protons=True):
        return None
    for s in steps:
        to_aim_moves(s)
    if any(s.unmatched for s in steps):
        return None
    prod_maps = {a.GetAtomMapNum() for a in Chem.MolFromSmiles(product).GetAtoms() if a.GetAtomMapNum()}
    final = Chem.MolFromSmiles(aim_view_smiles(state))
    targets = set()
    for frag in Chem.GetMolFrags(final, asMols=True):
        if {a.GetAtomMapNum() for a in frag.GetAtoms()} & prod_maps:
            targets.add(state_key(frag))
    starts = [state_key(aim_view_smiles(s.start)) for s in steps] + [state_key(aim_view_smiles(state))]
    n_moves = sum(len(s.aim_moves) for s in steps)
    return aim_view_smiles(steps[0].start), targets, starts[1:], n_moves


def follows(gold: list[str], path: list[str]) -> bool:
    """Is gold a subsequence of path?"""
    it = iter(path)
    return all(g in it for g in gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root_dir / "data" / "mech-USPTO-31k.csv"))
    parser.add_argument("--scorer", choices=["v121", "ranker"], required=True)
    parser.add_argument("--ranker", default=None)
    parser.add_argument("--n", type=int, default=400, help="test reactions sampled in total")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--nchunks", type=int, default=1)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scorer = Scorer(args.scorer, args.ranker)
    df = pd.read_csv(args.data)
    test = [i for i in range(len(df)) if split_of(i, df.mechanistic_class[i]) != "train"]
    test = sorted(test, key=lambda i: _hash01("path", i))[: args.n]
    mine = [r for j, r in enumerate(test) if j % args.nchunks == args.chunk]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"paths_{args.scorer}_{args.chunk}.jsonl", "w") as f:
        for row in mine:
            r = df.iloc[row]
            rec = {"row": row, "class": r.mechanistic_class, "split": split_of(row, r.mechanistic_class), "scorer": args.scorer}
            t0 = time.time()
            try:
                g = gold_route(r.updated_reaction, r.mechanistic_label)
                if g is None:
                    rec["skipped"] = "not fully mapped"
                else:
                    start, targets, gold_states, n_moves = g
                    depth = min(n_moves + 2, 14)
                    paths, info = beam_search(start, targets, scorer, depth=depth, beam=args.beam)
                    rec.update(n_gold_moves=n_moves, depth=depth, found=bool(paths),
                               gold_top1=bool(paths) and follows(gold_states, paths[0].states),
                               gold_top3=any(follows(gold_states, p.states) for p in paths),
                               best_len=len(paths[0].moves) if paths else None,
                               logps=[round(p.logp, 2) for p in paths], **info)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
