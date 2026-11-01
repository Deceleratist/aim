"""Ligand loss from a neutral 5-coordinate phosphorane (shared.rules, AIM_PHOSPHORANE_LIGAND_LOSS).

The states are trimmed versions of real mech-USPTO gold states (Mitsunobu qid 2100, Appel qid 664).
"""
import pytest
from rdkit import Chem, RDLogger

from shared import rules

RDLogger.DisableLog("rdApp.*")

MITSUNOBU = ("[OH+:101]([CH2:1][CH3:2])[P:301](c1ccccc1)(c1ccccc1)(c1ccccc1)"
             "[N:305]([NH:304][C:303](=[O:302])OCC)C(=O)OCC.[O-:32]c1ccccc1")
APPEL = "[O:101]([CH2:2][CH3:3])[P:301]([Br:1])(c1ccccc1)(c1ccccc1)c1ccccc1.BrC(Br)Br"
# P with five bonds counting P=O / S=O: must not gain ligand loss
NOT_PHOSPHORANES = ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "COP(=O)(OC)OC", "Cc1ccc(S(=O)(=O)Cl)cc1"]


def heterolyses(smiles):
    _, mechs, mol, _ = rules.run_next_step(smiles, True, None, False, True)
    idx2map = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    sym = {a.GetIdx(): a.GetSymbol() for a in mol.GetAtoms()}
    return {(sym[m[3]], idx2map[m[3]], sym[m[4]], idx2map[m[4]]) for m in mechs if m[0] == 10}


@pytest.fixture
def rule_on(monkeypatch):
    monkeypatch.setattr(rules, "PHOSPHORANE_LIGAND_LOSS", True)


def test_off_by_default_no_p_heterolysis():
    assert not rules.PHOSPHORANE_LIGAND_LOSS
    assert not any(h[0] == "P" for h in heterolyses(MITSUNOBU) | heterolyses(APPEL))


def test_mitsunobu_hydrazide_leaves(rule_on):
    assert ("P", 301, "N", 305) in heterolyses(MITSUNOBU)


def test_appel_bromide_leaves(rule_on):
    het = heterolyses(APPEL)
    assert ("P", 301, "Br", 1) in het
    assert not any(h[0] == "P" and h[2] == "C" for h in het)    # P-C bonds never break this way


@pytest.mark.parametrize("smiles", NOT_PHOSPHORANES)
def test_no_ligand_loss_from_p_o_or_s_o(rule_on, smiles):
    assert not any(h[0] in ("P", "S") for h in heterolyses(smiles))


def test_silicate_already_loses_ligand():
    # anionic 5-coordinate Si: the alkoxide product is generated without the new rule
    # (benchmark caveat: clean_mol may keep it under an equivalent type-3 label)
    sm = "CC(C)(C)[Si-](C)(C)(OCC)F"
    mols, _, _, _ = rules.run_next_step(sm, True, None, False, True)
    smi = {Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m))) for m in mols}
    assert Chem.MolToSmiles(Chem.MolFromSmiles("CC(C)(C)[Si](C)(C)F.CC[O-]")) in smi
