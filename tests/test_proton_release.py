"""H+ lost to the solvent (plan E3, shared.rules, AIM_PROTON_RELEASE)."""
import numpy as np
import pytest
from rdkit import Chem, RDLogger

from shared import rules
from shared.ml_helper import make_fingerprint

RDLogger.DisableLog("rdApp.*")


@pytest.fixture
def rule_on(monkeypatch):
    monkeypatch.setattr(rules, "PROTON_RELEASE", True)


def releases(smiles):
    products, mechs, mol, _ = rules.run_next_step(smiles, True, None, False, True)
    out = {}
    for p, m in zip(products, mechs):
        if m[0] == 10 and m[3] is None:
            out[mol.GetAtomWithIdx(m[4]).GetSymbol()] = Chem.MolToSmiles(p)
    return out, mol, [list(m) for m in mechs]


def test_off_by_default():
    assert not releases("CC[OH2+].O")[0]


@pytest.mark.parametrize("smiles, symbol, product", [
    ("CC[OH2+].O", "O", "CCO.O.[H+]"),                       # oxonium
    ("CC=[OH+].O", "O", "CC=O.O.[H+]"),                     # protonated carbonyl (implicit H)
    ("C[NH3+].[Cl-]", "N", "CN.[Cl-].[H+]"),                # ammonium in a neutral mixture
])
def test_release(rule_on, smiles, symbol, product):
    out, _, _ = releases(smiles)
    assert symbol in out
    assert set(out[symbol].split(".")) == set(Chem.MolToSmiles(Chem.MolFromSmiles(product)).split("."))


def test_explicit_h_from_free_proton(rule_on):
    out, _, _ = releases("[H][O+]=CC.O")
    assert set(out["O"].split(".")) == {"CC=O", "O", "[H+]"}


def test_not_in_basic_mixture(rule_on):
    assert not releases("CC[OH2+].[O-]C.[O-]C")[0]


def test_carbon_and_neutral_acids_untouched(rule_on):
    # C-H next to a cation and neutral O-H acids are not part of E3 (Junrui, 2 Nov)
    assert not releases("CC(C)=[CH+].CC(=O)O")[0]


def test_fingerprint_shape(rule_on):
    _, mol, mechs = releases("CC[OH2+].O")
    move = next(m for m in mechs if m[0] == 10 and m[3] is None)
    fp = make_fingerprint(mol, move)
    assert len(fp) == 182 and not np.isnan(fp[10])
