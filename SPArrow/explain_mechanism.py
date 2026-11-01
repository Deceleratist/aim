"""Print the top-3 mechanisms from reactants to product, with a plausibility label per step.

    python SPArrow/explain_mechanism.py "C[O-].CCBr" "CCOC" [--scorer ranker|v121] [--depth 4]

Each step's label comes from where its move ranks among all of AIM's candidate moves in that state:
  favoured              the best-ranked move
  plausible             ranked 2-3
  less likely (used)    ranked lower; the route still needs it, which can be chemically genuine
The probability is the softmax share of that move among all candidates in the state.

By default the first-year charge rule already in AIM (rules.screen_charges) is applied: no cationic
intermediates in a net-negative (basic) mixture, no anionic ones in a net-positive (acidic) mixture.
Without it the ranker preferred E1 over E2 for tBuO- + 2-bromopropane, and lost OH- in acid (J23).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "SPArrow"))

from beam_search import Scorer, beam_search, candidates, fragment_keys  # noqa: E402
from shared.ml_helper import make_fingerprint  # noqa: E402
from shared.rules import run_next_step  # noqa: E402

MOVE_NAMES = {0: "lone pair -> atom", 1: "lone pair -> bond (form + break)", 2: "bond -> atom", 3: "bond -> bond",
              4: "proton transfer (lone pair takes H+)", 5: "bond takes H+", 6: "hydride -> atom",
              7: "hydride -> bond", 8: "E2 elimination", 9: "E1 deprotonation", 10: "heterolysis", 11: "H- + H+ -> H2"}


def label(rank: int) -> str:
    return "favoured" if rank == 1 else "plausible" if rank <= 3 else "less likely (used)"


def describe(smiles: str, move: list, scorer: Scorer, screen_charges: bool):
    """Rank and softmax probability of `move` among the allowed candidates in state `smiles`, plus atom labels."""
    _, mechs, mol, keep_idx = candidates(smiles, screen_charges)
    keep = [list(mechs[i]) for i in keep_idx]
    fps = np.array([make_fingerprint(mol, list(m)) for m in keep], dtype=float)
    s = scorer(fps)
    p = np.exp(s - s.max()); p /= p.sum()
    i = keep.index(move) if move in keep else None
    atoms = [f"{mol.GetAtomWithIdx(a).GetSymbol()}{mol.GetAtomWithIdx(a).GetAtomMapNum()}" if a is not None else "-"
             for a in move[1:]]
    if i is None:
        return None, None, atoms, len(keep)
    return int((s > s[i]).sum()) + 1, float(p[i]), atoms, len(keep)


def map_atoms(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    for i, a in enumerate(m.GetAtoms(), start=1):
        if not a.GetAtomMapNum():
            a.SetAtomMapNum(i)
    return Chem.MolToSmiles(m, canonical=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reactants")
    parser.add_argument("product")
    parser.add_argument("--scorer", default="ranker", choices=["ranker", "v121"])
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--no-charge-screen", action="store_true",
                        help="allow cations in basic and anions in acidic mixtures (off by default)")
    args = parser.parse_args()
    screen = not args.no_charge_screen

    ranker_path = str(root_dir / "models" / "rank_v1" / "ranker.json")
    scorer = Scorer(args.scorer, ranker_path if args.scorer == "ranker" else None)
    start = map_atoms(args.reactants)
    targets = fragment_keys(args.product)
    paths, info = beam_search(start, targets, scorer, depth=args.depth, beam=args.beam, k=3, screen_charges=screen)
    if not paths:
        print(f"No mechanism reached the product within {args.depth} moves (expanded {info['expanded']} states).")
        return
    for n, path in enumerate(paths, start=1):
        print(f"\nMechanism {n}   (overall log-plausibility {path.logp:.2f}; probability of this route ≈ {math.exp(path.logp):.2g})")
        smi = start
        for k, move in enumerate(path.moves, start=1):
            rank, prob, atoms, n_cand = describe(smi, move, scorer, screen)
            tag = label(rank) if rank else "?"
            print(f"  step {k}: {MOVE_NAMES[move[0]]:38s} atoms {atoms}   rank {rank}/{n_cand}, p = {prob:.2f}  -> {tag}")
            # follow the path: the product of this move is the next state
            products, mechs, _, _ = run_next_step(smi, True, None, False, True)
            smi = Chem.MolToSmiles(products[[list(m) for m in mechs].index(move)], canonical=False)
        print(f"  final: {Chem.MolToSmiles(Chem.MolFromSmiles(smi))}")


if __name__ == "__main__":
    main()
