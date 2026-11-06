"""pKa look-up (plan B4): values only need to land in the right 5-unit block of findings/pka_table.md."""
import pytest
from rdkit import Chem

from shared.pka import atom_pka, atom_pkah, leaving_label, proton_label


@pytest.mark.parametrize("smiles, idx, lo, hi", [
    ("CC(=O)O", 3, 0, 5),                 # carboxylic acid
    ("Oc1ccccc1", 0, 5, 15),              # phenol
    ("CCO", 2, 15, 20),                   # ethanol
    ("O", 0, 15, 20),                     # water
    ("CC(C)(C)O", 4, 15, 20),             # tBuOH
    ("C[NH3+]", 1, 5, 15),                # ammonium
    ("c1cc[nH+]cc1", 3, 0, 10),           # pyridinium
    ("CC[OH2+]", 2, -5, 0),               # protonated alcohol
    ("[OH3+]", 0, -5, 0),                 # hydronium
    ("CC(C)=[OH+]", 3, -10, -5),          # protonated ketone
    ("CC(N)=[OH+]", 3, -5, 0),            # protonated amide
    ("CC(OC)=[OH+]", 4, -10, -5),         # protonated ester
    ("CC(C)=O", 0, 15, 25),               # acetone alpha C-H
    ("CCOC(C)=O", 4, 20, 30),             # ester alpha C-H
    ("CC#N", 0, 20, 30),                  # nitrile alpha C-H
    ("Br", 0, -10, -5),                   # HBr
    ("CC", 0, 30, 60),                    # alkane
    ("C[P+](c1ccccc1)(c1ccccc1)c1ccccc1", 0, 20, 30),   # Wittig salt
    ("CC[NH]C", 2, 30, 40),               # dialkylamine N-H
])
def test_acid_blocks(smiles, idx, lo, hi):
    v = atom_pka(Chem.MolFromSmiles(smiles), idx)
    assert v is not None and lo <= v <= hi, (smiles, v)


@pytest.mark.parametrize("smiles, idx, lo, hi", [
    ("CC[O-]", 2, 15, 20),                # alkoxide
    ("CC(=O)[O-]", 3, 0, 5),              # acetate
    ("[OH-]", 0, 15, 20),                 # hydroxide
    ("[Br-]", 0, -10, -5),                # bromide: very weak base
    ("CCN(CC)CC", 2, 5, 15),              # triethylamine
    ("c1ccncc1", 3, 0, 10),               # pyridine
    ("O", 0, -5, 0),                      # water as a base
    ("CC(C)=O", 3, -10, -5),              # carbonyl O as a base
    ("[H-]", 0, 30, 40),                  # hydride
    ("[CH2-]C(C)=O", 0, 15, 25),          # enolate carbon
])
def test_base_blocks(smiles, idx, lo, hi):
    v = atom_pkah(Chem.MolFromSmiles(smiles), idx)
    assert v is not None and lo <= v <= hi, (smiles, v)


def test_no_h_no_pka():
    assert atom_pka(Chem.MolFromSmiles("CC(C)=O"), 3) is None     # carbonyl O has no H


def test_labels():
    assert proton_label(15.7 - 4.8) == "good"     # hydroxide + acetic acid
    assert proton_label(15.7 - 19.5) == "meh"     # hydroxide + ketone alpha-H (aldol)
    assert proton_label(-9 - 50) == "bad"         # bromide + alkane C-H
    assert leaving_label(-9) == "good" and leaving_label(4.8) == "good"
    assert leaving_label(10) == "meh" and leaving_label(15.7) == "meh" and leaving_label(18) == "meh"
    assert leaving_label(36) == "bad"                # amide anion R2N-


def test_h_written_as_its_own_atom():
    # AIM states keep a proton picked up from a free H+ as a separate H atom
    p = Chem.SmilesParserParams()
    p.removeHs = False
    m = Chem.MolFromSmiles("[H][O+]=C(C)C", p)
    o = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "O")
    assert -10 <= atom_pka(m, o) <= -5
