"""Best-first (beam) mechanism search with a plausibility score (plan B2).

Unlike run_parallel (breadth-first, every kept move expanded, stop at the first product), this
expands the most plausible partial mechanisms first and returns a ranked list of complete ones.

Score of a path = sum over its moves of log P(move | state), where P is a softmax over all of
AIM's candidate moves in that state (so moves compete with each other, as in a chemist's head).
The per-move score comes from a scorer: v121 (logit of its probability) or a trained ranker.

Each AIM move is one depth level; a gold step that AIM writes as two moves takes two levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost
from rdkit import Chem, RDLogger

from shared.ml_helper import aromatic_flags, make_fingerprint
from shared.rules import run_next_step

RDLogger.DisableLog("rdApp.*")
root_dir = Path(__file__).resolve().parents[1]


class Scorer:
    """Turns candidate fingerprints into unnormalised log-scores (higher = more plausible)."""

    def __init__(self, kind: str, path: str | None = None):
        self.kind = kind
        if kind == "v121":
            self.model = xgboost.XGBClassifier()
            self.model.load_model(path or str(root_dir / "models" / "v121" / "xg_model.json"))
        elif kind == "ranker":
            self.model = xgboost.XGBRanker()
            self.model.load_model(path)
        else:
            raise ValueError(kind)

    def __call__(self, fps: np.ndarray) -> np.ndarray:
        if self.kind == "v121":
            p = np.clip(self.model.predict_proba(fps)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        return self.model.predict(fps)


def state_key(mol_or_smiles) -> str:
    """Map-free, stereo-free canonical SMILES of a whole state (used to merge identical states)."""
    m = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else Chem.Mol(mol_or_smiles)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def fragment_keys(smiles: str) -> set[str]:
    return set(state_key(smiles).split("."))


@dataclass(order=True)
class Path_:
    logp: float
    smiles: str = field(compare=False)
    moves: list = field(default_factory=list, compare=False)
    states: list = field(default_factory=list, compare=False)   # state keys after each move


def candidates(smiles: str, screen_charges: bool):
    """AIM's candidate moves for a state (type 11 excluded).

    With screen_charges, Richard's existing rule (rules.screen_charges) applies: in a net-negative
    (basic) state no product may contain a cation, in a net-positive (acidic) state no anion.
    """
    products, mechs, mol, allowed = run_next_step(smiles, True, None, screen_charges, True)
    allowed = set(allowed)
    keep = [i for i, m in enumerate(mechs) if m[0] != 11 and i in allowed]
    return products, mechs, mol, keep


def expand(smiles: str, scorer: Scorer, top_m: int, screen_charges: bool = False):
    """Children of a state: (log P(move), move, product SMILES), best top_m only."""
    products, mechs, mol, keep = candidates(smiles, screen_charges)
    if not keep:
        return []
    arom = aromatic_flags(mol)
    fps = np.array([make_fingerprint(mol, list(mechs[i]), arom) for i in keep], dtype=float)
    s = scorer(fps)
    logp = s - (s.max() + math.log(np.exp(s - s.max()).sum()))   # log-softmax over the candidates
    order = np.argsort(-logp)[:top_m]
    children = []
    for o in order:
        i = keep[o]
        try:
            children.append((float(logp[o]), list(mechs[i]), Chem.MolToSmiles(products[i], canonical=False)))
        except Exception:
            continue
    return children


def beam_search(start_smiles: str, target_fragments: set[str], scorer: Scorer, depth: int,
                beam: int = 10, top_m: int = 10, k: int = 3,
                screen_charges: bool = False) -> tuple[list[Path_], dict]:
    """Return up to k complete paths (best first) whose final state contains all target fragments."""
    frontier = [Path_(0.0, start_smiles)]
    done: list[Path_] = []
    seen_best: dict[str, float] = {}
    n_expanded = 0
    for _ in range(depth):
        children = []
        for p in frontier:
            n_expanded += 1
            try:
                kids = expand(p.smiles, scorer, top_m, screen_charges)
            except Exception:
                continue
            for lp, move, smi in kids:
                key = state_key(smi)
                total = p.logp + lp
                if seen_best.get(key, -math.inf) >= total:
                    continue        # reached the same state more plausibly before
                seen_best[key] = total
                child = Path_(total, smi, p.moves + [move], p.states + [key])
                if target_fragments <= set(key.split(".")):
                    done.append(child)
                else:
                    children.append(child)
        done.sort(key=lambda x: -x.logp)
        # stop once k complete paths are better than anything still open
        children.sort(key=lambda x: -x.logp)
        frontier = children[:beam]
        if len(done) >= k and (not frontier or done[k - 1].logp >= frontier[0].logp):
            break
        if not frontier:
            break
    return done[:k], {"expanded": n_expanded}
