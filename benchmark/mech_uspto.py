"""Convert mech-USPTO-31k arrow labels into elementary steps and AIM moves.

Label format (Chen et al., mech-USPTO-31k): a flat list of (source, sink) arrows over
atom-map numbers.
  source  a        lone pair on atom a
          [x, y]   electrons of bond x-y, ending up on y's side
  sink    z        atom z (new bond to the source end, or a lone pair if z is that end)
          [p, q]   bond p-q (one end is the source end; equivalent to sinking on the other end)
  x.d     the d-th hydrogen on heavy atom x (implicit in the SMILES)

Steps are not delimited in the labels. Arrows are applied one by one and a step is closed
as soon as no atom it touched breaks the octet/duet rule (greedy electron-flow segmentation).

Each step is then matched against AIM's move templates (shared/rules.py) to give moves
[type, i, j, k, l] in atom-map numbers; see map_to_aim_index() for AIM atom indices.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

from shared import rules

RDLogger.DisableLog("rdApp.*")
ptable = Chem.GetPeriodicTable()

TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))


# --- arrows -------------------------------------------------------------------------

@dataclass(frozen=True)
class Tok:
    """An atom as seen by an arrow: heavy atoms by map number, H atoms by their host."""
    map: int | None
    is_h: bool = False
    host: int | None = None   # heavy-atom map number the H is bonded to (None for bare H+)

    @property
    def aim(self) -> int | None:
        """Atom-map number AIM would use: H atoms are implicit, so use the host."""
        return self.host if self.is_h and self.host is not None else self.map


@dataclass
class Arrow:
    kind: str            # 'lp' or 'bond'
    src: tuple           # (a,) or (x, y) as raw label refs; electrons end on src[-1]
    sink: object         # raw label ref of the sink atom (after bond-sink normalisation)
    toks: tuple = ()     # Tok for each of src + (sink,), filled in when applied
    idx: tuple = ()      # (src atom indices, sink atom index), filled in when applied


def parse_label(label: str) -> list[Arrow]:
    arrows = []
    for src, snk in ast.literal_eval(label):
        if isinstance(src, list):
            kind, s = "bond", (src[0], src[1])
        else:
            kind, s = "lp", (src,)
        anchor = s[-1]
        if isinstance(snk, list):
            # a bond sink [p, q] is the same as sinking on whichever end is not the anchor
            if snk[0] == anchor:
                snk = snk[1]
            elif snk[1] == anchor:
                snk = snk[0]
            else:
                raise ValueError(f"bond sink {snk} does not contain source end {anchor}")
        arrows.append(Arrow(kind, s, snk))
    return arrows


# --- electron bookkeeping -----------------------------------------------------------

class ArrowState:
    """Reactant graph with explicit H on which arrows are applied."""

    def __init__(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("unparsable SMILES")
        mol = Chem.AddHs(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        for a in mol.GetAtoms():
            a.SetNoImplicit(True)
        self.mol = Chem.RWMol(mol)
        self.map_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
        self.h_labels: dict[str, int] = {}

    def copy(self) -> "ArrowState":
        new = ArrowState.__new__(ArrowState)
        new.mol = Chem.RWMol(self.mol)
        new.map_to_idx = self.map_to_idx
        new.h_labels = dict(self.h_labels)
        return new

    def resolve(self, ref) -> int:
        if isinstance(ref, int):
            return self.map_to_idx[ref]
        host = self.map_to_idx[int(ref)]
        key = repr(ref)
        hs_on_host = [n.GetIdx() for n in self.mol.GetAtomWithIdx(host).GetNeighbors() if n.GetAtomicNum() == 1]
        labelled = self.h_labels.get(key)
        if labelled is not None and labelled in hs_on_host:
            return labelled
        free = [h for h in hs_on_host if h not in self.h_labels.values()]
        if free:
            self.h_labels[key] = free[0]
            return free[0]
        if labelled is not None:   # the H has moved but keeps its original label
            return labelled
        if hs_on_host:
            return hs_on_host[0]
        raise KeyError(f"no hydrogen for {ref}")

    def tok(self, idx: int) -> Tok:
        a = self.mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() != 1:
            return Tok(a.GetAtomMapNum() or None)
        heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() != 1]
        host = heavy[0].GetAtomMapNum() if heavy else None
        return Tok(a.GetAtomMapNum() or None, True, host)

    def _bond_delta(self, i: int, j: int, delta: int) -> None:
        b = self.mol.GetBondBetweenAtoms(i, j)
        order = int(b.GetBondTypeAsDouble()) if b else 0
        new = order + delta
        if new < 0 or new > 3:
            raise ValueError(f"bond order {new} between {i} and {j}")
        if b:
            self.mol.RemoveBond(i, j)
        if new:
            self.mol.AddBond(i, j, Chem.BondType.values[new])

    def apply(self, arrow: Arrow) -> set[int]:
        """Move one electron pair; returns the atom indices it touched."""
        src = [self.resolve(r) for r in arrow.src]
        z = self.resolve(arrow.sink)
        arrow.toks = tuple(self.tok(i) for i in src + [z])
        arrow.idx = (tuple(src), z)
        return self.apply_idx(arrow.kind, src, z)

    def apply_idx(self, kind: str, src, z: int) -> set[int]:
        # electron ownership before/after: formal charge changes by (before - after)
        if kind == "lp":
            (a,) = src
            if z == a:
                raise ValueError("lone pair sinks on itself")
            before, after = {a: 2}, {a: 1, z: 1}
            self._bond_delta(a, z, +1)
        else:
            x, y = src
            if not self.mol.GetBondBetweenAtoms(x, y):
                raise ValueError("source bond does not exist")
            before = {x: 1, y: 1}
            self._bond_delta(x, y, -1)
            if z in (x, y):
                after = {z: 2}
            else:
                after = {y: 1, z: 1}
                self._bond_delta(y, z, +1)
        for idx in set(before) | set(after):
            atom = self.mol.GetAtomWithIdx(idx)
            atom.SetFormalCharge(atom.GetFormalCharge() + before.get(idx, 0) - after.get(idx, 0))
        return set(before) | set(after)

    def atom_ok(self, idx: int) -> bool:
        a = self.mol.GetAtomWithIdx(idx)
        z = a.GetAtomicNum()
        if z in TRANSITION_METALS:
            return True
        bonded = sum(int(b.GetBondTypeAsDouble()) for b in a.GetBonds())
        nonbonding = ptable.GetNOuterElecs(z) - bonded - a.GetFormalCharge()
        if nonbonding < 0:
            return False
        # Cl and Br never expand their octet in this chemistry (a neutral Br bonded to both P and C
        # made the Appel gold states unsanitisable, E1d); P, S, Si and hypervalent I may
        limit = 2 if z <= 2 else 8 if z <= 10 or z in (17, 35) else 12
        return nonbonding + 2 * bonded <= limit

    def snapshot(self) -> str | None:
        """Heavy-atom SMILES (no map numbers) of the current state, or None if it won't sanitise."""
        m = Chem.Mol(self.mol)
        try:
            Chem.SanitizeMol(m)
            m = Chem.RemoveHs(m)
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            return Chem.MolToSmiles(m)
        except Exception:
            return None

    def matches(self, product_smiles: str, ignore_protons: bool = False) -> bool:
        """Does the current state contain the (mapped) product, ignoring stereo?

        With ignore_protons the check is looser, to separate converter errors from quirks of
        the labels: charges and H counts are not compared, an aromatic bond matches a single
        or double one (protonated heteroaromatics are often not perceived as aromatic), and
        as a last resort the product may match a fragment by unmapped SMILES (some labels
        move a different but equivalent atom from the one the product's atom map names).
        """
        prod = Chem.MolFromSmiles(product_smiles)
        sim = Chem.Mol(self.mol)
        Chem.SanitizeMol(sim)
        sim = Chem.RemoveHs(sim)
        keep = {a.GetAtomMapNum() for a in prod.GetAtoms() if a.GetAtomMapNum()}

        def signature(m):
            sig = {}
            for a in m.GetAtoms():
                if a.GetAtomMapNum() not in keep:
                    continue
                nbrs = sorted((n.GetAtomMapNum(), m.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble())
                              for n in a.GetNeighbors())
                sig[a.GetAtomMapNum()] = (a.GetFormalCharge(), a.GetTotalNumHs(), nbrs)
            return sig

        p_sig, s_sig = signature(prod), signature(sim)
        if p_sig == s_sig:
            return True
        if not ignore_protons:
            return False

        def same_bonds(nb1, nb2):
            if [n for n, _ in nb1] != [n for n, _ in nb2]:
                return False
            return all(o1 == o2 or 1.5 in (o1, o2) for (_, o1), (_, o2) in zip(nb1, nb2))

        if p_sig.keys() == s_sig.keys() and all(same_bonds(p_sig[k][2], s_sig[k][2]) for k in p_sig):
            return True

        def unmapped(m):
            m = Chem.Mol(m)
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            Chem.RemoveStereochemistry(m)
            return Chem.MolToSmiles(m)

        frags = {unmapped(f) for f in Chem.GetMolFrags(sim, asMols=True)}
        return all(unmapped(f) in frags for f in Chem.GetMolFrags(prod, asMols=True))


# --- segmentation into steps -----------------------------------------------------

@dataclass
class Step:
    arrows: list[Arrow]
    smiles: str | None = None
    start: ArrowState | None = None   # state before the step's first arrow
    aim_moves: list[list] = field(default_factory=list)
    n_splits: int = 0          # concerted arrows that had to be split to fit AIM templates
    unmatched: list[Arrow] = field(default_factory=list)


def segment(state: ArrowState, arrows: list[Arrow]) -> tuple[list[Step], list[Arrow]]:
    """Greedily group arrows into steps; returns (steps, trailing arrows that never closed)."""
    steps, current, touched, start = [], [], set(), state.copy()
    for arrow in arrows:
        touched |= state.apply(arrow)
        current.append(arrow)
        if all(state.atom_ok(i) for i in touched):
            steps.append(Step(current, state.snapshot(), start))
            current, touched, start = [], set(), state.copy()
    return steps, current


# --- matching steps to AIM move templates ---------------------------------------
# Normalised arrow: (kind, src toks, sink tok). AIM uses implicit H, so H atoms become
# their host heavy atom; a bare proton (H+ with no host) is an atom in its own right.

def _is_h(t: Tok) -> bool:
    return t.is_h and t.host is not None


def _candidates(a: list[Arrow]):
    """Yield every (move, n_arrows) template that matches the head of `a`, longest first."""
    def lp(arw):
        return arw.kind == "lp"

    def bond(arw):
        return arw.kind == "bond"

    def lp_parts(arw):
        return arw.toks[0], arw.toks[1]

    def bond_parts(arw):
        return arw.toks[0], arw.toks[1], arw.toks[2]

    if len(a) >= 3 and lp(a[0]) and bond(a[1]) and bond(a[2]):
        i, h = lp_parts(a[0])
        x, j, k = bond_parts(a[1])
        k2, l, l2 = bond_parts(a[2])
        if _is_h(h) and x == h and h.host == j.map and not _is_h(k) and k2 == k and l2 == l:
            yield [8, i.aim, k.aim, j.aim, l.aim], 3           # E2

    if len(a) >= 2:
        a0, a1 = a[0], a[1]
        if bond(a0) and bond(a1):
            k, i, z = bond_parts(a0)
            x, j, z1 = bond_parts(a1)
            if _is_h(i) and _is_h(z) and x == z and z1 == j:
                yield [11, k.aim, j.aim, None, None], 2        # H- + H+ -> H2
            if _is_h(i) and not _is_h(z) and x == z and z1 == j:
                yield [7, None, z.aim, k.aim, j.aim], 2        # hydride to bond
            if not k.is_h and not _is_h(i) and _is_h(z) and x == z and j.map == z.host and z1 == j:
                yield [5, i.aim, None, k.aim, j.aim], 2        # bond takes a proton
            if not k.is_h and not _is_h(i) and not _is_h(z) and z not in (k, i) and x == z and z1 == j:
                yield [3, i.aim, z.aim, k.aim, j.aim], 2       # bond to bond
        if lp(a0) and bond(a1):
            i, z = lp_parts(a0)
            x, j, z1 = bond_parts(a1)
            if _is_h(z) and x == z and j.map == z.host:
                if z1 == j:
                    yield [4, i.aim, None, None, j.aim], 2     # proton transfer
                if not _is_h(z1):
                    yield [9, i.aim, z1.aim, j.aim, None], 2   # E1-type deprotonation
            if not _is_h(z) and x == z and z1 == j:
                yield [1, i.aim, z.aim, None, j.aim], 2        # attack + bond breaks

    a0 = a[0]
    if bond(a0):
        k, i, z = bond_parts(a0)
        # k is H: the proton leaves as bare H+ with no base. AIM has a move for it only with
        # AIM_PROTON_RELEASE=1 (plan E3), written as a heterolysis with the H slot empty
        if rules.PROTON_RELEASE and z == i and _is_h(k) and k.host == i.map:
            yield [10, None, None, None, i.aim], 1             # H+ lost to the solvent
        if _is_h(i) and not _is_h(z):
            yield [6, None, z.aim, k.aim, None], 1             # hydride to atom
        if z == i and not _is_h(i) and not k.is_h:
            yield [10, None, None, k.aim, i.aim], 1            # heterolysis
        if z not in (k, i) and not _is_h(z) and not k.is_h:
            yield [2, i.aim, z.aim, k.aim, None], 1            # bond to atom
    else:
        i, z = lp_parts(a0)
        if not _is_h(z):
            yield [0, i.aim, z.aim, None, None], 1             # lone pair to atom


def _split(arw: Arrow) -> list[Arrow] | None:
    """([x,y] -> z) == ([x,y] -> y) then (y -> z): lets AIM take a concerted push in two moves."""
    if arw.kind != "bond":
        return None
    x, y, z = arw.toks
    if z in (x, y):
        return None
    (xi, yi), zi = arw.idx
    first = Arrow("bond", arw.src, arw.src[1], (x, y, y), ((xi, yi), yi))
    second = Arrow("lp", (arw.src[1],), arw.sink, (y, z), ((yi,), zi))
    return [first, second]


def _decompose(pending: list[Arrow], state: ArrowState, budget: list[int]):
    """Cover `pending` with AIM moves so that every intermediate obeys the octet rule.

    Depth-first, trying templates before splits, so the first answer uses the fewest
    splits along its path. Returns (moves, n_splits) or None.
    """
    if not pending:
        return [], 0
    budget[0] -= 1
    if budget[0] < 0:
        return None
    for move, n in _candidates(pending):
        if move[0] == 10 and move[3] is None and \
                state.mol.GetAtomWithIdx(state.map_to_idx[move[4]]).GetAtomicNum() not in (7, 8, 16):
            continue   # E3 covers H+ loss from O, N, S only; C-H loss (e.g. tBu+ -> isobutylene) stays unmatched
        nxt = state.copy()
        touched = set()
        for arw in pending[:n]:
            touched |= nxt.apply_idx(arw.kind, arw.idx[0], arw.idx[1])
        if not all(nxt.atom_ok(i) for i in touched):
            continue
        rest = _decompose(pending[n:], nxt, budget)
        if rest is not None:
            return [move] + rest[0], rest[1]
    for p, arw in enumerate(pending):
        parts = _split(arw)
        if parts:
            rest = _decompose(pending[:p] + parts + pending[p + 1:], state, budget)
            if rest is not None:
                return rest[0], rest[1] + 1
    return None


def to_aim_moves(step: Step) -> None:
    found = _decompose(list(step.arrows), step.start, [200])
    if found is None:
        step.unmatched = list(step.arrows)
    else:
        step.aim_moves, step.n_splits = found


# --- top level ------------------------------------------------------------------

@dataclass
class Conversion:
    steps: list[Step]
    trailing: list[Arrow]
    final_match: bool
    error: str | None = None
    match_up_to_protons: bool = False

    @property
    def fully_mapped(self) -> bool:
        return not self.error and not self.trailing and all(not s.unmatched for s in self.steps)


def convert(updated_reaction: str, label: str) -> Conversion:
    reactants, product = updated_reaction.split(">>")
    try:
        arrows = parse_label(label)
        state = ArrowState(reactants)
        steps, trailing = segment(state, arrows)
    except Exception as e:  # malformed label or reaction
        return Conversion([], [], False, f"{type(e).__name__}: {e}")
    try:
        final = state.matches(product)
        heavy = final or state.matches(product, ignore_protons=True)
    except Exception as e:
        return Conversion(steps, trailing, False, f"final state: {type(e).__name__}: {e}")
    for s in steps:
        to_aim_moves(s)
    return Conversion(steps, trailing, final, match_up_to_protons=heavy)


def map_to_aim_index(reactant_smiles: str) -> dict[int, int]:
    """Atom-map number -> RDKit atom index as run_parallel parses the same reactant SMILES."""
    mol = Chem.MolFromSmiles(reactant_smiles)
    return {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
