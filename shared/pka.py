"""Approximate pKa look-up for AIM states (plan B4; values and scales in findings/pka_table.md, reviewed
by Junrui 5 Nov). Values only need to be right to within a 5-unit block.

atom_pka(mol, i)   pKa of the most acidic H on atom i (None if it has no H or no row matches)
atom_pkah(mol, i)  pKa of i's conjugate acid, i.e. how basic i is: protonate i and look up the result
                   (None if i can't take a proton)
The first matching row wins, so specific patterns come before general ones. Each pattern's first atom
is the one that carries the acidic H.
"""
from __future__ import annotations

from rdkit import Chem

# (SMARTS of the acid form, pKa, pka_table.md row)
_ROWS = [
    # cationic acids
    ("[NX2+;H1]#[#6]", -10, 1),
    ("[O+;H1;X2]=[#6]-[NX3]", -0.5, 8),               # X2 counts an H written as its own atom
    ("[O+;H1;X2]=[#6]-[O]", -6, 3),
    ("[O+;H1;X2]=[#6]", -7, 2),
    ("[O+;H1;X2]=[#15]", -2, 9),
    ("[OX3+;H3]", -1.7, 7),
    ("[OX3+;H2]", -2, 6),
    ("[OX3+;H1]", -3.5, 5),
    ("[#1+;D0]", -2, 4),
    ("[nH+;r5]", 7.0, 13),
    ("[nH+]", 5.2, 11),
    ("[NX3+;H1,H2]=[#6](-[NX3])-[NX3]", 13.5, 18),
    ("[NX3+;H1,H2]=[#6]-[NX3]", 12.5, 17),
    ("[NX3+;H1,H2]=[#6]", 8, 14),
    ("[NX4+;H1,H2,H3]-c", 4.6, 10),
    ("[NX4+;H1,H2,H3,H4]", 10, 16),
    ("[PX4+;H1]-c", 2.7, 12),
    ("[PX4+;H1]", 8.4, 15),
    ("[CX4;H1,H2]([P+])[#6]=O", 9, 46),
    ("[CX4;H1,H2,H3][P+]", 22, 49),
    ("[CX4;H1,H2,H3][#6+]", -7, 55),
    # neutral O-H, N-H, S-H, halogen acids
    ("[IX1;H1]", -10, 19), ("[BrX1;H1]", -9, 19), ("[ClX1;H1]", -7, 19), ("[FX1;H1]", 3.2, 23),
    ("[OX2H1]S(=O)=O", -2.8, 20),
    ("[OX2H1]C(=O)C(F)(F)F", 0.5, 22),
    ("[OX2H1]C(=O)[O-]", 10.3, 33),
    ("[OX2H1]C(=O)O", 6.4, 26),
    ("[OX2H1]C(=O)N", 5, 25),
    ("[OX2H1]C=O", 4.8, 24),
    ("[NX2H1]=[N+]=[N-]", 4.7, 28),
    ("[SX2H1]c", 6.5, 27),
    ("[SX2H1]", 10.6, 32),
    ("[NX3H1](C=O)C=O", 9, 29),
    ("[NX3;H1,H2]S(=O)=O", 10, 31),
    ("[OX2H1]c", 10, 30),
    ("[OX2H1]C=C", 11, 34),
    ("[NX3H1](C=O)N", 12.5, 35),
    ("[OX2H2]", 15.7, 36),
    ("[OX2H1][CX4]([#6])([#6])[#6]", 18, 38),
    ("[OX2H1]", 16, 37),
    ("[NX3;H1,H2]C=O", 17, 39),
    ("[NX3H3]", 38, 41),
    ("[NX3;H1,H2]", 36, 40),
    # C-H acids
    ("[CX2H1]#N", 9.2, 42),
    ("[CX4;H1,H2](C(=O)O)C(=O)O", 13, 45),
    ("[CX4;H1,H2](C=O)C=O", 9, 43),
    ("[CX4;H1,H2,H3][N+](=O)[O-]", 10, 44),
    ("[CX4;H1,H2,H3][CX3H1]=O", 17, 47),
    ("[CX4;H1,H2,H3]C(=O)[#6]", 19.5, 48),
    ("[CX4;H1,H2,H3]C(=O)[O,N]", 25, 50),
    ("[CX4;H1,H2,H3]C#N", 25, 51),
    ("[CX2H1]#C", 25, 52),
    ("[#1X1][#1X1]", 35, 53),
    ("[cH]", 43, 54), ("[CX3;!H0]", 44, 54), ("[CX4;!H0]", 50, 54),
]
_PATTERNS = [(Chem.MolFromSmarts(s), v, row) for s, v, row in _ROWS]
assert all(p is not None for p, _, _ in _PATTERNS)


def _prepare(mol):
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    except Exception:
        pass
    return mol


def _lookup(mol, idx):
    for patt, value, row in _PATTERNS:
        for match in mol.GetSubstructMatches(patt, uniquify=False):
            if match[0] == idx:
                return value, row
    return None


def atom_pka(mol, idx: int):
    """pKa of the most acidic H on atom idx, or None."""
    mol = _prepare(mol)
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() == 1 and a.GetDegree() == 0 and a.GetFormalCharge() == 1:
        return -2.0                                    # bare H+ (row 4)
    if a.GetTotalNumHs(includeNeighbors=True) == 0:
        return None
    hit = _lookup(mol, idx)
    return None if hit is None else float(hit[0])


def atom_pkah(mol, idx: int):
    """pKa of the conjugate acid of atom idx (its basicity), or None if it can't take a proton."""
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() == 1 and a.GetFormalCharge() == -1:
        return 35.0                                    # hydride -> H2 (row 53)
    if a.GetAtomicNum() not in (6, 7, 8, 9, 15, 16, 17, 35, 53):
        return None
    rw = Chem.RWMol(mol)
    b = rw.GetAtomWithIdx(idx)
    b.SetNumExplicitHs(b.GetTotalNumHs() + 1)
    b.SetNoImplicit(True)
    b.SetFormalCharge(b.GetFormalCharge() + 1)
    try:
        m = rw.GetMol()
        m.UpdatePropertyCache(strict=False)
    except Exception:
        return None
    return atom_pka(m, idx)


def proton_label(dpka: float) -> str:
    """ΔpKa = pKaH(base) - pKa(acid); scale (a) in pka_table.md."""
    return "good" if dpka >= 0 else "meh" if dpka >= -5 else "poor" if dpka >= -10 else "bad"


def leaving_label(pkah: float) -> str:
    """pKaH of the leaving group; scale (b) in pka_table.md (Junrui 5 Nov, cut-off moved to alcohols 6 Nov)."""
    return "good" if pkah < 5 else "meh" if pkah <= 18 else "bad"   # alcohols (tBuOH 18) are the cut-off
