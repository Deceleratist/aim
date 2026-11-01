"""The cached-aromaticity fingerprint must be bit-identical to the original implementation."""
from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
from rdkit import Chem

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from shared import ml_helper  # noqa: E402
from shared.rules import run_next_step  # noqa: E402

# aromatic, heteroaromatic, charged and acyclic states, incl. a pyridinium that loses aromaticity bits
STATES = [
    "[CH3:1][O-:2].[CH3:3][CH2:4][Br:5]",
    "Nc1scc(Br)c1-c1nccs1.OC(=O)CN1C(=O)CCc2ncccc21.C1CCC(N=C=NC2CCCCC2)CC1",
    "O=[N+]([O-])c1ccc(F)cc1.[NH2:9]C",
    "c1cc[nH+]cc1.CC(=O)[O-]",
    "CC(=O)c1ccccc1.[BH4-].CO",
]


def original_module():
    src = subprocess.run(["git", "show", "45981633:shared/ml_helper.py"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("ml_helper_original", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(src, mod.__dict__)
    return mod


def same(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def test_fingerprints_identical_to_original():
    old = original_module()
    n = 0
    for smi in STATES:
        _, mechs, mol, _ = run_next_step(smi, True, None, False, True)
        for m in mechs:
            if m[0] == 11:
                continue
            assert same(ml_helper.make_fingerprint(mol, list(m)), old.make_fingerprint(mol, list(m))), (smi, m)
            n += 1
    assert n > 100


def test_cache_recomputes_when_molecule_changes():
    mol = Chem.RWMol(Chem.MolFromSmiles("C1=CC=CC=C1"))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    assert all(ml_helper.aromatic_flags(mol))
    mol.GetBondBetweenAtoms(0, 1).SetBondType(Chem.BondType.SINGLE)   # break the ring's aromaticity
    mol.GetAtomWithIdx(0).SetNumExplicitHs(1)
    mol.GetAtomWithIdx(1).SetNumExplicitHs(1)
    mol.UpdatePropertyCache(strict=False)
    assert not all(ml_helper.aromatic_flags(mol))
