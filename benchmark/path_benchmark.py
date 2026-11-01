"""Path-level benchmark (plan A7): beam search from reactants to the gold end state.

For each test reaction: start from the reactants, target = the molecules of the gold final state
that contain the recorded product's atoms (the converter's end state, so labels that stop before
the last proton transfer are still a fair target). Then report, per scorer:
  found      - a complete mechanism reaches the product within the depth limit
  full_top3  - one of the top-3 ends with every reacting molecule of the gold end state, by-products
               included (a route that skips the reagent, e.g. amide from acid without DCC, fails)
  org_top3   - as full_top3 but only the carbon-containing gold molecules (ignores inorganic
               bookkeeping such as NaOH + H+ vs Na+ + H2O)
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


def norm_key(smiles: str) -> str:
    """state_key with H2 written one way. The converter's states keep H2 as [H][H] while AIM writes
    [HH], so without this every NaH reaction failed full/gold matching even on the textbook route.
    (RemoveHs with removeOnlyHNeighbors deletes H2 altogether, so it is rewritten here instead.)"""
    return ".".join(sorted("[HH]" if f == "[H][H]" else f for f in state_key(smiles).split(".")))


def norm_frags(state: str) -> set[str]:
    return {norm_key(s) for s in state.split(".")}


def has_carbon(key: str) -> bool:
    return any(a.GetAtomicNum() == 6 for a in Chem.MolFromSmiles(key).GetAtoms())


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
    targets, full_targets = set(), set()
    for frag in Chem.GetMolFrags(final, asMols=True):
        maps = {a.GetAtomMapNum() for a in frag.GetAtoms()} - {0}
        if maps & prod_maps:
            targets.add(state_key(frag))
        if maps:
            full_targets.add(norm_key(Chem.MolToSmiles(frag)))   # every reacting molecule incl. by-products (e.g. DCU)
    starts = [norm_key(aim_view_smiles(s.start)) for s in steps] + [norm_key(aim_view_smiles(state))]
    n_moves = sum(len(s.aim_moves) for s in steps)
    return aim_view_smiles(steps[0].start), targets, starts[1:], n_moves, full_targets


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
    parser.add_argument("--screen-charges", action="store_true", help="apply AIM's acid/base charge screen")
    parser.add_argument("--only_class", default=None, help="sample only test reactions of this class")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scorer = Scorer(args.scorer, args.ranker)
    df = pd.read_csv(args.data)
    test = [i for i in range(len(df)) if split_of(i, df.mechanistic_class[i]) != "train"
            and args.only_class in (None, df.mechanistic_class[i])]
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
                    start, targets, gold_states, n_moves, full_targets = g
                    depth = min(n_moves + 2, 14)
                    paths, info = beam_search(start, targets, scorer, depth=depth, beam=args.beam,
                                              screen_charges=args.screen_charges)

                    org_targets = {k for k in full_targets if has_carbon(k)}

                    def full(p):
                        # the end state contains every reacting molecule of the gold end state,
                        # by-products included, so a route that skips the reagent does not count
                        return full_targets <= norm_frags(p.states[-1])

                    def org(p):
                        # looser: only the carbon-containing gold molecules, so inorganic bookkeeping
                        # (e.g. NaOH + H+ vs Na+ + H2O) does not count against a route
                        return org_targets <= norm_frags(p.states[-1])

                    def gold(p):
                        return follows(gold_states, [norm_key(s) for s in p.states])

                    rec.update(n_gold_moves=n_moves, depth=depth, found=bool(paths),
                               full_top1=bool(paths) and full(paths[0]),
                               full_top3=any(full(p) for p in paths),
                               org_top1=bool(paths) and org(paths[0]),
                               org_top3=any(org(p) for p in paths),
                               gold_top1=bool(paths) and gold(paths[0]),
                               gold_top3=any(gold(p) for p in paths),
                               best_len=len(paths[0].moves) if paths else None,
                               logps=[round(p.logp, 2) for p in paths],
                               paths=[{"moves": p.moves, "states": p.states} for p in paths], **info)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
