from rdkit import Chem
import numpy as np

"""
Fingerprint layout notes
------------------------

The canonical feature layout lives in `shared.fingerprint_schema`.
Use that module to:

- build the feature index map used by `explain_mech.ipynb`
- validate that schema labels still match the implementation here
- export `shared/fingerprint_schema.json`


"""

electronegativity_dict = {1: 2.2,
 3: 0.98,
 4: 1.57,
 5: 2.04,
 6: 2.55,
 7: 3.04,
 8: 3.44,
 9: 3.98,
 11: 0.93,
 12: 1.31,
 13: 1.61,
 14: 1.9,
 15: 2.19,
 16: 2.58,
 17: 3.16,
 19: 0.82,
 20: 1.0,
 21: 1.36,
 22: 1.54,
 23: 1.63,
 24: 1.66,
 25: 1.55,
 26: 1.83,
 27: 1.88,
 28: 1.91,
 29: 1.9,
 30: 1.65,
 31: 1.81,
 32: 2.01,
 33: 2.18,
 34: 2.55,
 35: 2.96,
 36: 3.0,
 37: 0.82,
 38: 0.95,
 39: 1.22,
 40: 1.33,
 41: 1.6,
 42: 2.16,
 43: 1.9,
 44: 2.2,
 45: 2.28,
 46: 2.2,
 47: 1.93,
 48: 1.69,
 49: 1.78,
 50: 1.96,
 51: 2.05,
 52: 2.1,
 53: 2.66}

def aromatic_flags(mol):
    """Aromaticity of every atom after re-perceiving it on a copy of `mol` (mol itself is untouched).

    Perceiving aromaticity copies the molecule. Code that fingerprints many candidate moves of the
    same state should call this once and pass the result to make_fingerprint(..., arom=...).
    """
    new_mol = Chem.Mol(mol)
    Chem.SetAromaticity(new_mol)
    return [a.GetIsAromatic() for a in new_mol.GetAtoms()]


def make_atom_fingerprint(atom, other_atom_idx=None, other_atoms=[], is_aromatic=None):
    if atom is None:
        return [np.nan] * 40

    # Charge bitstring
    charge_bitstring = [0] * 2
    if atom.GetFormalCharge() < 0:
        charge_bitstring[0] = 1
    elif atom.GetFormalCharge() > 0:
        charge_bitstring[1] = 1

    # H bitstring
    h_number_bitstring = [0] * 4
    if atom.GetTotalNumHs() == 0:
        h_number_bitstring[0] = 1
    elif atom.GetTotalNumHs() == 1:
        h_number_bitstring[1] = 1
    elif atom.GetTotalNumHs() == 2:
        h_number_bitstring[2] = 1
    elif atom.GetTotalNumHs() > 2:
        h_number_bitstring[3] = 1

    # Hybridization bitstring
    hybridization_bitstring = [0] * 3
    if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
        hybridization_bitstring[0] = 1
    elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
        hybridization_bitstring[1] = 1
    elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
        hybridization_bitstring[2] = 1

    # Element bitstring
    # [AlkMet, B, C, N, O, F, Mg, Al, Si, P, S, Cl, Br, I, Met]
    ptable = Chem.GetPeriodicTable()
    element_bitstring = [0] * 15
    dict_elements = { 5: 1,
                      6: 2,
                      7: 3,
                      8: 4,
                      9: 5,
                      12: 6,
                      13: 7,
                      14: 8,
                      15: 9,
                      16: 10,
                      17: 11,
                      35: 12,
                      53: 13}
    atm_num = atom.GetAtomicNum()
    if atm_num in dict_elements:
        element_bitstring[dict_elements[atm_num]] = 1
    elif ptable.GetNOuterElecs(atm_num) == 1:
        element_bitstring[0] = 1
    elif atm_num <= 30 and atm_num >= 20:
        element_bitstring[14] = 1

    # aromaticity bitstring
    if is_aromatic is None:
        new_mol = Chem.Mol(atom.GetOwningMol())
        Chem.SetAromaticity(new_mol)
        is_aromatic = new_mol.GetAtomWithIdx(atom.GetIdx()).GetIsAromatic()
    if is_aromatic:
        aromatic_bit = [1]
    else:
        aromatic_bit = [0]

    # adjacency bitstring
    # TODO: add C(EN)3 for good leaving group
    # [C+, C-, CH0, CH, C-EN, C=EN, EN, EP]

    adjacency_bitstring = [0] * 10
    relative_adjacency = [0] * 4
    for a in atom.GetNeighbors():
        if other_atom_idx != None and a.GetIdx() == other_atom_idx:
            continue
        if a.GetIdx() in other_atoms:
            relative_adjacency[other_atoms.index(a.GetIdx())] = 1
        if a.GetAtomicNum() == 6:
            adjacency_bitstring[0] = 1
            if a.GetTotalNumHs() > 0:
                adjacency_bitstring[1] = 1
            if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2 or a.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                adjacency_bitstring[2] = 1
            if a.GetIsAromatic():
                adjacency_bitstring[3] = 1
            for a2 in a.GetNeighbors():
                if electronegativity_dict[a2.GetAtomicNum()] > 2.55:
                    for b in a.GetBonds():
                        if b.GetOtherAtom(a).GetIdx() == a2.GetIdx():
                            if b.GetBondTypeAsDouble() == 1:
                                adjacency_bitstring[4] = 1
                            elif b.GetBondTypeAsDouble() >= 2:
                                adjacency_bitstring[5] = 1    
        elif electronegativity_dict[a.GetAtomicNum()] > 2.55:
            adjacency_bitstring[6] = 1
        elif electronegativity_dict[a.GetAtomicNum()] < 2.05:
            adjacency_bitstring[7] = 1
        if a.GetFormalCharge() > 0:
            adjacency_bitstring[8] = 1
        elif a.GetFormalCharge() < 0:
            adjacency_bitstring[9] = 1
           
        

    three_membered_ring_bitstring = [0] * 1
    if atom.IsInRingSize(3):
        three_membered_ring_bitstring[0] = 1

 

    total_bit_string = charge_bitstring + h_number_bitstring + hybridization_bitstring + element_bitstring + aromatic_bit + adjacency_bitstring + relative_adjacency + three_membered_ring_bitstring
    return total_bit_string

def make_bond_fingerprint(bond):
    # bond type bitstring
    # [single, double, triple]
    bond_type_bitstring = [0] * 3
    if not bond:
        return bond_type_bitstring
    if bond.GetBondTypeAsDouble() == 1:
        bond_type_bitstring[0] = 1
    elif bond.GetBondTypeAsDouble() == 2:
        bond_type_bitstring[1] = 1
    elif bond.GetBondTypeAsDouble() == 3:
        bond_type_bitstring[2] = 1
    return bond_type_bitstring


def new_bond_fingerprint(mol, a1_idx, a2_idx):
    if a1_idx == None or a2_idx is None:
        return [np.nan] * 5
    a1 = mol.GetAtomWithIdx(a1_idx)
    a2 = mol.GetAtomWithIdx(a2_idx)
    # ring size bitstring
    # [3, 4, 5, 6, 7+]
    ring_size_bitstring = [np.nan] * 5
    fragments = Chem.GetMolFrags(mol)
    for f in fragments:
        if a1.GetIdx() in f and a2.GetIdx() in f:
            if a1.GetIdx() == a2.GetIdx():
                continue
            try:
                len_path = len(Chem.GetShortestPath(mol, a1.GetIdx(), a2.GetIdx()))
            except:
                print("Error in calculating shortest path")
                continue
            if len_path == 3:
                ring_size_bitstring[0] = 1
            elif len_path == 4:
                ring_size_bitstring[1] = 1
            elif len_path == 5:
                ring_size_bitstring[2] = 1
            elif len_path == 6:
                ring_size_bitstring[3] = 1
            elif len_path >= 7:
                ring_size_bitstring[4] = 1
    return ring_size_bitstring


def make_fingerprint(sm, mech, arom=None):
    """Fingerprint of one candidate move. `arom` = aromatic_flags(sm), passed in when fingerprinting
    many moves of the same state (it is the same for all of them)."""
    mech_type = [np.nan] * 11
    mech_type[mech[0]] = 1
    atom_fps = []

    # Current fix for elimination mechanism that preserves the information of C-H bond atom index for future implementations
    if mech[0] in [8, 9]:
        mech[3] = None
    if arom is None:
        arom = aromatic_flags(sm)
    for i in range(1, 5):
        if mech[i] is None:
            atom_fps.append(make_atom_fingerprint(None, other_atoms=mech[1:]))
        else:
            if i in [1, 2]:
                atom_fps.append(make_atom_fingerprint(sm.GetAtomWithIdx(mech[i]), mech[i + 2], mech[1:], arom[mech[i]]))
            elif i in [3, 4]:
                atom_fps.append(make_atom_fingerprint(sm.GetAtomWithIdx(mech[i]), mech[i - 2], mech[1:], arom[mech[i]]))
            else:
                atom_fps.append(make_atom_fingerprint(sm.GetAtomWithIdx(mech[i]), other_atoms=mech[1:], is_aromatic=arom[mech[i]]))
    if mech[3] is not None and mech[1] is not None:
        b1 = make_bond_fingerprint(sm.GetBondBetweenAtoms(mech[1], mech[3]))
    else:
        b1 = [np.nan] * 3
    if mech[4] is not None and mech[2] is not None:
        b2 = make_bond_fingerprint(sm.GetBondBetweenAtoms(mech[2], mech[4]))
    else:
        b2 = [np.nan] * 3
    r = new_bond_fingerprint(sm, mech[1], mech[2])
    sum_finger_print = mech_type + atom_fps[0] + atom_fps[1] + atom_fps[2] + atom_fps[3] + b1 + b2 + r
    return sum_finger_print
