"""List AIM's candidate next moves for one state, best first, as both scorers see them.

    python SPArrow/next_moves.py "C[O-].CCBr" [--top 10] [--no-charge-screen]

For playing with single steps: every move the rules propose in this state is scored by rank_v1
(shown as its softmax share among all candidates) and by v121 (its yes/no probability; >= 0.5 is
"kept"). Each row shows the move type, the atoms involved (element + map number) and the product.
Atoms are numbered in input order unless the SMILES already carries map numbers, so the numbers in
the output point back at the input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "SPArrow"))

from beam_search import Scorer, candidates  # noqa: E402
from explain_mechanism import MOVE_NAMES, map_atoms  # noqa: E402
from shared.ml_helper import aromatic_flags, make_fingerprint  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", help="SMILES of the current state, e.g. 'C[O-].CCBr'")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--no-charge-screen", action="store_true",
                        help="allow cations in basic and anions in acidic mixtures (off by default)")
    args = parser.parse_args()

    smiles = map_atoms(args.state)
    products, mechs, mol, keep = candidates(smiles, not args.no_charge_screen)
    if not keep:
        print("AIM proposes no moves in this state.")
        return
    moves = [list(mechs[i]) for i in keep]
    arom = aromatic_flags(mol)
    fps = np.array([make_fingerprint(mol, m, arom) for m in moves], dtype=float)
    s = Scorer("ranker", str(root_dir / "models" / "rank_v1" / "ranker.json"))(fps)
    share = np.exp(s - s.max()); share /= share.sum()
    p121 = 1 / (1 + np.exp(-Scorer("v121")(fps)))

    print(f"state: {smiles}")
    print(f"{len(moves)} candidate moves; v121 keeps {(p121 >= 0.5).sum()}. Top {min(args.top, len(moves))} by rank_v1:\n")
    print(f"{'#':>3}  {'rank_v1':>7}  {'v121':>5}  {'move':38s} {'atoms':24s} product")
    for r, i in enumerate(np.argsort(-s)[: args.top], start=1):
        atoms = " ".join(f"{mol.GetAtomWithIdx(a).GetSymbol()}{mol.GetAtomWithIdx(a).GetAtomMapNum()}"
                         for a in moves[i][1:] if a is not None)
        prod = products[keep[i]]
        for a in prod.GetAtoms():
            a.SetAtomMapNum(0)
        print(f"{r:3d}  {share[i]:7.2f}  {p121[i]:5.2f}  {MOVE_NAMES[moves[i][0]]:38s} {atoms:24s} {Chem.MolToSmiles(prod)}")


if __name__ == "__main__":
    main()
