"""Halophilic attack (plan E1h, shared.rules, AIM_HALOPHILIC_ATTACK): PPh3 + CBr4 -> Ph3P+-Br + -CBr3."""
import pytest
from rdkit import Chem, RDLogger

from shared import rules

RDLogger.DisableLog("rdApp.*")

APPEL = "OCCc1ccccc1.BrC(Br)(Br)Br.c1ccc(P(c2ccccc2)c2ccccc2)cc1"


@pytest.fixture
def rule_on(monkeypatch):
    monkeypatch.setattr(rules, "HALOPHILIC_ATTACK", True)


def attacks(smiles):
    products, mechs, mol, _ = rules.run_next_step(smiles, True, None, False, True)
    out = set()
    for p, m in zip(products, mechs):
        if m[0] == 1 and mol.GetAtomWithIdx(m[1]).GetSymbol() == "P" and \
                mol.GetAtomWithIdx(m[2]).GetSymbol() in ("Cl", "Br", "I"):
            out.add(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(p))))
    return out


def test_off_by_default():
    assert not attacks(APPEL)


def test_appel_first_step(rule_on):
    out = attacks(APPEL)
    assert len(out) == 1
    frags = set(next(iter(out)).split("."))
    assert "Br[C-](Br)Br" in frags
    assert Chem.MolToSmiles(Chem.MolFromSmiles("Br[P+](c1ccccc1)(c1ccccc1)c1ccccc1")) in frags


@pytest.mark.parametrize("smiles", [
    "CCBr.c1ccc(P(c2ccccc2)c2ccccc2)cc1",                 # ordinary alkyl halide: SN2 at C, not at Br
    "BrC(Br)(Br)Br.O=P(c1ccccc1)(c1ccccc1)c1ccccc1",       # phosphine oxide is not a nucleophile here
    "ClC(Cl)(Cl)Cl.CN(C)C",                                # only P nucleophiles (narrow by design)
])
def test_not_proposed(rule_on, smiles):
    assert not attacks(smiles)


def test_chloroform_and_ccl4(rule_on):
    assert attacks("ClC(Cl)(Cl)Cl.CP(C)C")
    assert attacks("ClC(Cl)Cl.CP(C)C")       # CHCl3: C keeps its H
