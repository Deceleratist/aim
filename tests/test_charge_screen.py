"""Sharper charge screen (plan E1f, AIM_SHARP_CHARGE_SCREEN=1).

The reductive-amination state is a trimmed version of mech-USPTO row 21487: aldehyde, morpholine,
H+ and Na+ BH3CN- (net +1, so acidic).
"""
import pytest
from rdkit import Chem, RDLogger

from shared import rules

RDLogger.DisableLog("rdApp.*")

RED_AM = "O=Cc1ccccc1.C1COCCN1.[H+].[BH3-]C#N.[Na+]"
PROTONATED_CO = Chem.MolFromSmarts("[O+]=[CX3]")
ALKOXIDE = Chem.MolFromSmarts("[CX4][OX1-]")


@pytest.mark.parametrize("smiles, ions", [
    ("CC[O-]", (1, 0)), ("CC(=O)[O-]", (0, 0)), ("CS(=O)(=O)[O-]", (0, 0)), ("[BH3-]C#N", (0, 0)),
    ("[Br-]", (0, 0)), ("[O-][P+](c1ccccc1)(c1ccccc1)c1ccccc1", (0, 0)), ("[N-]=[N+]=N", (0, 0)),
    ("CC(C)=C[O-]", (1, 0)), ("[NH2-]", (1, 0)), ("CC[OH2+]", (0, 1)), ("C[OH+]C", (0, 1)), ("[H+]", (0, 1)),
    ("[NH4+]", (0, 0)), ("C[N+](C)(C)C", (0, 0)),
])
def test_strong_ions(smiles, ions):
    assert rules.strong_ions(Chem.MolFromSmiles(smiles)) == ions


def kept_products(smiles, monkeypatch, sharp):
    monkeypatch.setenv("AIM_SHARP_CHARGE_SCREEN", "1" if sharp else "0")
    products, _, mol, keep = rules.run_next_step(smiles, True, None, True, True)
    return [products[i] for i in keep]


def test_old_screen_blocks_spectator_anion(monkeypatch):
    # R16: with Richard's screen every product still holding BH3CN- is rejected, so C=O can't even be protonated
    kept = kept_products(RED_AM, monkeypatch, sharp=False)
    assert not any(m.HasSubstructMatch(PROTONATED_CO) for m in kept)


def test_sharp_screen(monkeypatch):
    kept = kept_products(RED_AM, monkeypatch, sharp=True)
    assert any(m.HasSubstructMatch(PROTONATED_CO) for m in kept)       # acid catalysis allowed
    assert not any(m.HasSubstructMatch(ALKOXIDE) for m in kept)        # hydride -> C=O giving RO- in acid: gone
    parent = rules.strong_ions(Chem.MolFromSmiles(RED_AM))[0]
    assert all(rules.strong_ions(m)[0] <= parent for m in kept)


def test_neutral_mixture_unscreened(monkeypatch):
    n_sharp = len(kept_products("CC[O-].CCBr.[Na+]", monkeypatch, sharp=True))
    n_none = len(rules.run_next_step("CC[O-].CCBr.[Na+]", True, None, False, True)[3])
    assert n_sharp == n_none
