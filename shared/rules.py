from rdkit import Chem # pyright: ignore[reportMissingImports]
from datetime import datetime
import os
import sys
ptable = Chem.GetPeriodicTable()
from .electronegativity import electronegativity

electronegativity_dict = electronegativity()


def clean_mol(list_mol, mechanisms, original_mol = None):
    existing_smiles = set([])
    clean_list = []
    updated_mech = []
    if original_mol is not None:       
        og_smiles = Chem.MolToSmiles(original_mol, kekuleSmiles=True)
    for idx in range(len(list_mol)):
        mol = list_mol[idx]
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        if smiles not in existing_smiles:
            if original_mol == None or smiles != og_smiles:
                existing_smiles.add(smiles)
                clean_list.append(mol)
                updated_mech.append(mechanisms[idx])
    return (clean_list, updated_mech)

def screen_charges(list_mol, charge = 0):
    if charge < 0: # base mediated reaction will not have cationic intermediates
        clean_list = []

        for idx in range(len(list_mol)):
            mol = list_mol[idx]
            keep = True
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() > 0:
                    keep = False
                    break
            if keep:
                clean_list.append(idx)
        return clean_list
    elif charge > 0: # acid mediated reaction will not have anionic intermediates
        clean_list = []

        for idx in range(len(list_mol)):
            mol = list_mol[idx]
            keep = True
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() < 0:
                    keep = False
                    break
            if keep:
                clean_list.append(idx)
        return clean_list
    else:
        # print('No screening')
        return list(range(len(list_mol)))


def strong_ions(mol):
    """(strongly basic anions, strongly acidic cations) in a state, for the sharper charge screen.

    Strongly basic anion: O-, N- or C- whose conjugate acid is weak (alkoxide, amide, carbanion,
    enolate). Not counted: an anion bonded to a cation (P+-O-, nitro, azide, ylide forms) and O- on
    an atom that also carries =O or =S (carboxylate, sulfonate, phosphate), which survive in acid.
    Strongly acidic cation: O+ carrying H (oxonium, protonated C=O or alcohol) or a bare H+.
    """
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    bases = acids = 0
    for atom in mol.GetAtoms():
        q, z = atom.GetFormalCharge(), atom.GetAtomicNum()
        if q < 0 and z in (6, 7, 8):
            nbrs = atom.GetNeighbors()
            if any(n.GetFormalCharge() > 0 for n in nbrs):
                continue
            if z == 8 and any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() in (8, 16)
                              for n in nbrs for b in n.GetBonds() if b.GetOtherAtom(n).GetIdx() != atom.GetIdx()):
                continue
            bases += 1
        elif q > 0 and ((z == 8 and atom.GetTotalNumHs(includeNeighbors=True) > 0) or (z == 1 and atom.GetDegree() == 0)):
            acids += 1
    return bases, acids


def screen_charges_sharp(list_mol, charge, parent):
    """Plan E1f (AIM_SHARP_CHARGE_SCREEN=1): in a net-positive (acidic) mixture drop only moves that
    create a strongly basic anion; in a net-negative (basic) mixture only moves that create a strongly
    acidic cation. Ions already present, and weak ones (halide, carboxylate, BH3CN-), are allowed,
    unlike screen_charges, which rejects any charged atom (and so any spectator anion; R16)."""
    if charge == 0:
        return list(range(len(list_mol)))
    b0, a0 = strong_ions(parent)
    k = 0 if charge > 0 else 1
    before = (b0, a0)[k]
    return [i for i, m in enumerate(list_mol) if strong_ions(m)[k] <= before]


def identify_source(mol, p_block = False):
    for atom in mol.GetAtoms():
        # Look for hydrides - M-H bond
        if electronegativity_dict[atom.GetAtomicNum()] < 2.05 and atom.GetFormalCharge() < 0 and atom.GetTotalNumHs() > 0:
            atom.SetNumExplicitHs(atom.GetTotalNumHs())
            atom.SetProp('source', 'hydride')

        # Look for lone pairs
        valence_electrons = ptable.GetNOuterElecs(atom.GetAtomicNum())
        lone_electrons = valence_electrons - atom.GetTotalValence() - atom.GetFormalCharge()
        if lone_electrons > 0:
            atom.SetProp('source', 'lone pair')
            
    for bond in mol.GetBonds():
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]

        if bond.GetBondTypeAsDouble() == 1:
            # This is an experimental feature - unlikely to be a very common reaction mode
            if electronegativity_dict[atoms[0].GetAtomicNum()] < 2.05 and electronegativity_dict[atoms[1].GetAtomicNum()] < 2.05:
                bond.SetProp('source', 'sigma bond')
                bond.SetProp('source_bond_atom', 'both')
                bond.SetProp('source_other_atom', 'both')
            # For any element less electronegative than boron (inclusive) - add sigma bond as source - going from more electronegative atom
            for a in atoms:
                if electronegativity_dict[a.GetAtomicNum()] < 2.05:
                    other = a.GetIdx()
                    source = bond.GetOtherAtomIdx(other)
                    # avoid breaking a sigma bond if the source atom instead has a lone pair
                    if mol.GetAtomWithIdx(source).HasProp('source') and mol.GetAtomWithIdx(source).GetProp('source') == 'lone pair':
                        continue
                    bond.SetProp('source', 'sigma bond')
                    bond.SetIntProp('source_bond_atom', source)
                    bond.SetIntProp('source_other_atom', other)
                
                if p_block:
                    if a.GetAtomicNum() == 15 and a.GetFormalCharge() < 0:
                        other = a.GetIdx()
                        source = bond.GetOtherAtomIdx(other)
                        bond.SetProp('source', 'sigma bond')
                        bond.SetIntProp('source_bond_atom', source)
                        bond.SetIntProp('source_other_atom', other)
        # Looking for double and triple bonds - ignoring aromatic systems
        if bond.GetBondTypeAsDouble() > 1.5:
            
            atoms = [bond.GetBeginAtom() ,bond.GetEndAtom()]

            # if bond is homonuclear then either atom can be source
            if atoms[0].GetAtomicNum() == atoms[1].GetAtomicNum():
                bond.SetProp('source', 'pi bond')
                bond.SetProp('source_bond_atom', 'both')
                bond.SetProp('source_other_atom', 'both')
            else:
                if electronegativity_dict[atoms[0].GetAtomicNum()] < electronegativity_dict[atoms[1].GetAtomicNum()]:
                    # avoid breaking pi bond is the source atom has a lone pair
                    if atoms[1].HasProp('source') and atoms[1].GetProp('source') == 'lone pair':
                        continue
                    bond.SetProp('source', 'pi bond')
                    bond.SetIntProp('source_bond_atom', atoms[1].GetIdx())
                    bond.SetIntProp('source_other_atom',atoms[0].GetIdx())
                else:
                    # avoid breaking pi bond is the source atom has a lone pair
                    if atoms[0].HasProp('source') and atoms[0].GetProp('source') == 'lone pair':
                        continue
                    bond.SetProp('source', 'pi bond')
                    bond.SetIntProp('source_bond_atom', atoms[0].GetIdx())
                    bond.SetIntProp('source_other_atom',atoms[1].GetIdx())

def identify_sink(mol, p_block=False):
   
    # Look for vacant orbitals
    for atom in mol.GetAtoms():
        if electronegativity_dict[atom.GetAtomicNum()] > 2.55 and atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() >= 0:
            atom.SetProp('sink', 'proton')
        valence_electrons = ptable.GetNOuterElecs(atom.GetAtomicNum())
        lone_electrons = valence_electrons - atom.GetTotalValence() - atom.GetFormalCharge()
        num_bond_electrons = atom.GetTotalValence()*2

        if lone_electrons + num_bond_electrons < 8 and valence_electrons > 2:
            atom.SetProp('sink', 'empty AO')
        # Special case for s block
        if valence_electrons < 3 and num_bond_electrons/2 < valence_electrons and atom.GetFormalCharge() >= 0:
            atom.SetProp('sink', 'empty AO')

        # The second row p block expansion pack
        if p_block:
            if atom.GetAtomicNum() < 17 and atom.GetAtomicNum() > 13 and atom.GetFormalCharge() >= 0 and atom.GetTotalValence() < 6:
                if atom.HasProp('sink') and atom.GetProp('sink') == 'proton':
                    continue
                atom.SetProp('sink', 'hypervalent')

    for bond in mol.GetBonds():
        # Conditional on all rules being swtiched on
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        # Identify possible sigma stars by considering leaving groups being any element that has a higher electronegativity than carbon
        if bond.GetBondTypeAsDouble() == 1:
            if electronegativity_dict[atoms[0].GetAtomicNum()] > 2.55 and electronegativity_dict[atoms[1].GetAtomicNum()] > 2.55:
                bond.SetProp('sink', 'sigma star')
                bond.SetProp('sink_bond_atom', 'both')
                bond.SetProp('sink_other_atom', 'both')
            else:
                for a in atoms:
                    # bond is considered a sink if there is an element more electronegative than carbon attached to it
                    if electronegativity_dict[a.GetAtomicNum()] > 2.55:
                        other = a.GetIdx()
                        sink = bond.GetOtherAtomIdx(other)
                        # do not add as a sink if there is a empty AO on the sink atom (unlikely case)
                        if mol.GetAtomWithIdx(sink).HasProp('sink') and mol.GetAtomWithIdx(sink).GetProp('sink') == 'empty AO':
                            continue
                        bond.SetProp('sink', 'sigma star')
                        bond.SetIntProp('sink_bond_atom', sink)
                        bond.SetIntProp('sink_other_atom', other)
                if p_block:
                    if a.GetAtomicNum() == 15 and a.GetFormalCharge() > 0:
                        other = a.GetIdx()
                        sink = bond.GetOtherAtomIdx(other)
                        # do not add as a sink if there is a empty AO on the sink atom (unlikely case)
                        if mol.GetAtomWithIdx(sink).HasProp('sink') and mol.GetAtomWithIdx(sink).GetProp('sink') == 'empty AO':
                            continue
                        bond.SetProp('sink', 'sigma star')
                        bond.SetIntProp('sink_bond_atom', sink)
                        bond.SetIntProp('sink_other_atom', other)
         # Find multiple bonds ignoring aromatic systems      
        if bond.GetBondTypeAsDouble() > 1.5:
            atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            if atoms[0].GetAtomicNum() == atoms[1].GetAtomicNum():
                if (atoms[0].HasProp('sink') and atoms[0].GetProp('sink') == 'empty AO') or (atoms[1].HasProp('sink') and atoms[1].GetProp('sink') == 'empty AO'):
                        continue
                bond.SetProp('sink', 'pi star')
                bond.SetProp('sink_bond_atom', 'both')
                bond.SetProp('sink_other_atom', 'both')
            else:
                if electronegativity_dict[atoms[0].GetAtomicNum()] < electronegativity_dict[atoms[1].GetAtomicNum()]:
                    if atoms[0].HasProp('sink') and atoms[0].GetProp('sink') == 'empty AO':
                        continue
                    bond.SetProp('sink', 'pi star')
                    bond.SetIntProp('sink_bond_atom', atoms[0].GetIdx())
                    bond.SetIntProp('sink_other_atom',atoms[1].GetIdx())
                else:
                    if atoms[1].HasProp('sink') and atoms[1].GetProp('sink') == 'empty AO':
                        continue
                    bond.SetProp('sink', 'pi star')
                    bond.SetIntProp('sink_bond_atom', atoms[1].GetIdx())
                    bond.SetIntProp('sink_other_atom', atoms[0].GetIdx())
    for atom in mol.GetAtoms():
        if atom.HasProp('sink') and atom.GetProp('sink') == 'hypervalent':
            atom.SetProp('sink', 'empty AO')

def identify_acidic_CH(mol):
    """
    This works by looking to see if any adjacent bonds are
    attached to heteroatoms.
    Checking for double bonds would help looking for enolisable
    protons however looking for any good leaving group allows
    elimination to occur in the next step.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() > -1:
            for atom2 in atom.GetNeighbors():
                if atom2.GetFormalCharge() > 0:
                    atom.SetProp('sink', 'CH')
                for atom3 in atom2.GetNeighbors():
                    if electronegativity_dict[atom3.GetAtomicNum()]>2.55:
                        atom.SetProp('sink', 'CH')
                        break

def prepare_molecule(mol, p_block = True, setNoImplicitFalse = True):
    if setNoImplicitFalse:
        for a in mol.GetAtoms():
            a.SetNoImplicit(False)
    identify_acidic_CH(mol)
    identify_sink(mol, p_block=p_block)
    identify_source(mol, p_block=p_block)
    return None

def b_to_a(mol, i, j):
    bond = mol.GetBondBetweenAtoms(i,j)
    bond_order = int(bond.GetBondTypeAsDouble())
    a_l = mol.GetAtomWithIdx(i)
    c = a_l.GetFormalCharge()
    a_l.SetFormalCharge(c+1)
    a_b = mol.GetAtomWithIdx(j)
    c = a_b.GetFormalCharge()
    a_b.SetFormalCharge(c-1)
    e_mol = Chem.EditableMol(mol)
    e_mol.RemoveBond(i,j)
    if bond_order != 1:
        e_mol.AddBond(i,j, order=Chem.BondType.values[bond_order-1])
    return e_mol.GetMol()

def a_to_b(mol, i, j):
    bond = mol.GetBondBetweenAtoms(i,j)
    a_l = mol.GetAtomWithIdx(i)
    c = a_l.GetFormalCharge()
    a_l.SetFormalCharge(c+1)
    a_b = mol.GetAtomWithIdx(j)
    c = a_b.GetFormalCharge()
    a_b.SetFormalCharge(c-1)
    e_mol = Chem.EditableMol(mol)
    if not bond:
        e_mol.AddBond(i,j, order=Chem.BondType.values[1])
    else:
        e_mol.RemoveBond(i,j)
        bond_order = int(bond.GetBondTypeAsDouble())
        e_mol.AddBond(i,j, order=Chem.BondType.values[bond_order+1])
    return e_mol.GetMol()


def lose_hydride(mol, i):
    h_mol = Chem.Mol(mol)
    a = h_mol.GetAtomWithIdx(i)
    a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
    a.SetFormalCharge(a.GetFormalCharge() + 1)
    return h_mol

def transfer_proton(mol, j, i):
    p_mol = Chem.Mol(mol)
    a = p_mol.GetAtomWithIdx(j)
    a.SetFormalCharge(a.GetFormalCharge() - 1)
    if a.GetNumExplicitHs() > 0:
        a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
    elif a.GetTotalNumHs() == 1:
            a.SetNoImplicit(True)
    if i != None:
        a2 = p_mol.GetAtomWithIdx(i)
        a2.SetFormalCharge(a2.GetFormalCharge() + 1)
        if a2.GetNumExplicitHs() > 0 or a2.GetFormalCharge() == 0:
            a2.SetNumExplicitHs(a2.GetNumExplicitHs() + 1)
    else:
        p_mol = Chem.CombineMols(p_mol, Chem.MolFromSmiles("[H][H]"))
    return p_mol
    #return (mechanisms, connected_mols)


def c_h_elimination(mol, i, j, mechanisms, connected_mols):
    c_mol = Chem.Mol(mol)
    c_mol.UpdatePropertyCache(strict=False)
    a = c_mol.GetAtomWithIdx(j)
    a.SetFormalCharge(a.GetFormalCharge() - 1)
    if a.GetNumExplicitHs() > 0:
        a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
    elif a.GetTotalNumHs() == 1:
        a.SetNoImplicit(True)
    a2 = c_mol.GetAtomWithIdx(i)
    a2.SetFormalCharge(a2.GetFormalCharge() + 1)
    # Required for sensible RDKit behaviour or else it produces radicals
    if a2.GetNumExplicitHs() > 0 or a2.GetFormalCharge() == 0:
        a2.SetNumExplicitHs(a2.GetNumExplicitHs() + 1)
    for atom2 in a.GetNeighbors():
        for b in atom2.GetBonds():
            # Find any bonds that have a leaving group for elimination
            if b.HasProp('sink') and b.GetProp('sink_bond_atom') == 'both':
                k = atom2.GetIdx()
                l = b.GetOtherAtomIdx(k)
            elif b.HasProp('sink') and atom2.GetIdx() == int(b.GetProp('sink_bond_atom')):
                k = atom2.GetIdx()
                l = int(b.GetProp('sink_other_atom'))
            else: 
                continue
            if mol.GetAtomWithIdx(l).GetFormalCharge() >= 0:
                intermediate = a_to_b(Chem.Mol(c_mol), j, k)
                connected_mols.append(b_to_a(Chem.Mol(intermediate), k, l))
                mechanisms.append([8, i, k, j, l])

        if atom2.GetFormalCharge() > 0 and atom2.HasProp('sink') and atom2.GetProp('sink')== 'empty AO':
            k = atom2.GetIdx()
            mechanisms.append([9, i, k, j, None])
            connected_mols.append(a_to_b(Chem.Mol(c_mol), j, k))
    return (mechanisms, connected_mols)
    
def bond_to_hydride(mol, i, j, l, mechanisms, connected_mols):
    m_mol = Chem.Mol(mol)
    m_mol = b_to_a(m_mol, j, l)
    m_mol.GetAtomWithIdx(j).SetNumExplicitHs(m_mol.GetAtomWithIdx(j).GetNumImplicitHs() + 1)
    m_mol.GetAtomWithIdx(j).SetFormalCharge(m_mol.GetAtomWithIdx(j).GetFormalCharge() - 1)
    connected_mols.append(m_mol)
    mechanisms.append([7, None, j, i, l])
    return (mechanisms, connected_mols)

def break_bond_source(mol, b, a_list):
    order = []
    new_mol = []
    if b.GetProp('source_bond_atom')=='both':
        counter = 0
        for a in a_list:
            i = a.GetIdx()
            k = b.GetOtherAtom(a).GetIdx()
            if mol.GetAtomWithIdx(k).GetFormalCharge() > 0:
                continue
            new_mol.append(Chem.Mol(mol))
            new_mol[counter] = b_to_a(new_mol[counter], k, i)
            order.append(i)
            counter += 1
    else:
        i = b.GetIntProp('source_bond_atom')
        k = b.GetIntProp('source_other_atom')
        if mol.GetAtomWithIdx(k).GetFormalCharge() > 0:
            return (new_mol, order)
        order.append(i)
        new_mol.append(Chem.Mol(mol))
        new_mol[0] = b_to_a(new_mol[0], k, i)
    return (new_mol, order)

def break_bond_sink(new_mol, order, a, b1, b2, mechanisms, connected_mols):
    j = a.GetIdx()
    l = b2.GetOtherAtom(a).GetIdx()
    for counter in range(len(new_mol)):
        n_mol = Chem.Mol(new_mol[counter])
        i = order[counter]
        k = b1.GetOtherAtom(n_mol.GetAtomWithIdx(i)).GetIdx()
        if i == j or n_mol.GetAtomWithIdx(l).GetFormalCharge() < 0:
            continue
        n_mol = b_to_a(n_mol, j, l)
        connected_mols.append(a_to_b(n_mol, i, j))
        mechanisms.append([3, i, j, k, l])
    return (mechanisms, connected_mols)

# Ligand loss from a neutral 5-coordinate phosphorane, e.g. Ph3P(OR)(NR-NR) -> Ph3P+-OR + hydrazide
# (Mitsunobu) or Ph3P(OR)Br -> Ph3P+-OR + Br- (Appel). Off by default; set AIM_PHOSPHORANE_LIGAND_LOSS=1.
# Only P with five single-bonded neighbours (no P=O / P=C), leaving N, O or halogen. Anionic
# 5-coordinate silicates already lose ligands through the charge -1 branch below. Approved by Junrui 1 Nov
# for P and Si only; S (Swern) stays as direct substitution at S+.
PHOSPHORANE_LIGAND_LOSS = os.environ.get("AIM_PHOSPHORANE_LIGAND_LOSS", "0") == "1"
LIGAND_LOSS_LEAVING = {7, 8, 9, 17, 35, 53}


def is_phosphorane(atom):
    return (atom.GetAtomicNum() == 15 and atom.GetFormalCharge() == 0 and atom.GetTotalDegree() == 5
            and all(b.GetBondTypeAsDouble() == 1 for b in atom.GetBonds()))


def carbocat(mol, sink_idx, connected_mols, mechanisms):
    b = mol.GetBondWithIdx(sink_idx)
    if b.GetProp('sink_bond_atom') == 'both':
        return (mechanisms, connected_mols)
    if b.GetBondTypeAsDouble() != 1:
        return (mechanisms, connected_mols)
    j = b.GetIntProp('sink_bond_atom')
    l = b.GetIntProp('sink_other_atom')
    if mol.GetAtomWithIdx(l).GetFormalCharge() < 0:
        return (mechanisms, connected_mols)
    if mol.GetAtomWithIdx(j).GetAtomicNum() == 6:
        if mol.GetAtomWithIdx(j).GetTotalNumHs() > 1:
            return (mechanisms, connected_mols)
    elif PHOSPHORANE_LIGAND_LOSS and is_phosphorane(mol.GetAtomWithIdx(j)) and \
            mol.GetAtomWithIdx(l).GetAtomicNum() in LIGAND_LOSS_LEAVING:
        pass
    else:
        if mol.GetAtomWithIdx(j).GetFormalCharge() > -1:
            return (mechanisms, connected_mols)

    new_mol = Chem.Mol(mol)
    connected_mols.append(b_to_a(new_mol, j, l))
    mechanisms.append([10, None, None, j, l])
    return (mechanisms, connected_mols)

def run_next_step(smiles, kekulise = False, chain_mol = None, screen_charges_enabled = True, p_block = True):
    if chain_mol is not None:
        mol_inp = chain_mol
        for a in mol_inp.GetAtoms():
            a.ClearProp('source')
            a.ClearProp('sink')
            if a.GetFormalCharge() == 0:
                a.SetNoImplicit(False)
        for b in mol_inp.GetBonds():
            b.ClearProp('source')
            b.ClearProp('sink')
            b.ClearProp('source_bond_atom')
            b.ClearProp('sink_bond_atom')
            b.ClearProp('source_other_atom')
            b.ClearProp('sink_other_atom')
    else:
        mol_inp = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol_inp)
        if mol_inp is None:
            raise ValueError("Invalid SMILES string")
        if kekulise:
            Chem.Kekulize(mol_inp, clearAromaticFlags=True)
        
    prepare_molecule(mol_inp, p_block=p_block)
    connected_mols, mechanisms, mol_inp, clean_list = execute_step(mol_inp, kekulise, screen_charges_enabled)
    return (connected_mols, mechanisms, mol_inp, clean_list)

def execute_step(mol_inp, kekulise = False, screen_charges_enabled = True):


    connected_mols = []

    mechanisms = []

    sources_atoms = {}
    sinks_atoms = {}
    sources_bonds = {}
    sinks_bonds = {}
 

    for atom1 in mol_inp.GetAtoms():
        if atom1.HasProp('source'):
            sources_atoms[atom1.GetIdx()] = atom1.GetProp('source')
        if atom1.HasProp('sink'):
            sinks_atoms[atom1.GetIdx()] = atom1.GetProp('sink')
    for bond1 in mol_inp.GetBonds():
        if bond1.HasProp('source'):
            sources_bonds[bond1.GetIdx()] = bond1.GetProp('source')
        if bond1.HasProp('sink'):
            sinks_bonds[bond1.GetIdx()] = bond1.GetProp('sink')

    #atoms to atoms

    for source_idx, source_type in sources_atoms.items():
        i = source_idx
        if mol_inp.GetAtomWithIdx(i).GetFormalCharge() > 0:
            continue
        hydride_flag = False
        if source_type == 'hydride':
                hydride_flag = True
                h_mol = lose_hydride(mol_inp, i)
        for sink_idx, sink_type in sinks_atoms.items():
            j = sink_idx

            if mol_inp.GetAtomWithIdx(j).GetFormalCharge() < 0:
                continue
            if i == j:
                continue

            if sink_type == 'proton':
                if hydride_flag:
                    p_mol = transfer_proton(h_mol, j, None)
                    connected_mols.append(p_mol)
                    # This is not an ideal fix - introducing yet another arrow type
                    mechanisms.append([11, i, j, None, None])   
                else:
                    p_mol = transfer_proton(mol_inp, j, i)
                    connected_mols.append(p_mol)
                    mechanisms.append([4, i, None, None, j])   
                continue

            if sink_type == 'CH':
                if source_type == 'hydride':
                    # use [H-] in these cases
                    continue
                # Same proton transfer behaviour as above
                mechanisms, connected_mols = c_h_elimination(mol_inp, i, j, mechanisms, connected_mols)
                continue

            if hydride_flag:
                m_mol = Chem.Mol(h_mol)
                if m_mol.GetAtomWithIdx(j).GetNumExplicitHs()>0:
                    m_mol.GetAtomWithIdx(j).SetNumExplicitHs(m_mol.GetAtomWithIdx(j).GetNumExplicitHs() + 1)
                m_mol.GetAtomWithIdx(j).SetFormalCharge(m_mol.GetAtomWithIdx(j).GetFormalCharge() - 1)
                connected_mols.append(m_mol)
                mechanisms.append([6, None, j, i, None])
                continue

            new_mol = Chem.Mol(mol_inp)
            connected_mols.append(a_to_b(new_mol, i, j))
            mechanisms.append([0, i, j, None, None])

    # atoms to bonds

    for source_idx, source_type in sources_atoms.items():
        i = source_idx
        if mol_inp.GetAtomWithIdx(i).GetFormalCharge() > 0:
            continue
        hydride_flag = False
        if source_type == 'hydride':
                h_mol = lose_hydride(mol_inp, i)
                hydride_flag = True
                a = h_mol.GetAtomWithIdx(i)
        for sink_idx, sink_type in sinks_bonds.items():
            b = mol_inp.GetBondWithIdx(sink_idx)

            if b.GetProp('sink_bond_atom') == 'both':
                a_list = [b.GetBeginAtom(), b.GetEndAtom()]
                for a in a_list:
                    j = a.GetIdx()
                    l = b.GetOtherAtom(a).GetIdx()
                    if mol_inp.GetAtomWithIdx(l).GetFormalCharge()<0:
                        continue
                    if i == j:
                        continue

                    if hydride_flag:
                        mechanisms, connected_mols = bond_to_hydride(h_mol, i, j, l, mechanisms, connected_mols)
                        continue

                    new_mol = Chem.Mol(mol_inp)
                    new_mol = b_to_a(new_mol, j, l)
                    new_mol = a_to_b(new_mol, i, j)
                    connected_mols.append(new_mol)
                    mechanisms.append([1, i, j, None, l])
            else:
                j = b.GetIntProp('sink_bond_atom')
                l = b.GetIntProp('sink_other_atom')
                if mol_inp.GetAtomWithIdx(l).GetFormalCharge()<0:
                        continue
                if i == j:
                        continue
                
                if hydride_flag:
                    mechanisms, connected_mols = bond_to_hydride(h_mol, i, j, l, mechanisms, connected_mols)
                    continue
                
                new_mol = Chem.Mol(mol_inp)
                new_mol = b_to_a(new_mol,j, l)
                connected_mols.append(a_to_b(new_mol, i, j))
                mechanisms.append([1, i, j, None, l])


    #bonds to atoms

    for source_idx, source_type in sources_bonds.items():
        b = mol_inp.GetBondWithIdx(source_idx)
        a_list = [b.GetBeginAtom(), b.GetEndAtom()]

        new_mol, order = break_bond_source(mol_inp, b, a_list)
        if new_mol == []:
            continue
       
        for sink_idx, sink_type in sinks_atoms.items():
            j = sink_idx
            if mol_inp.GetAtomWithIdx(j).GetFormalCharge() < 0:
                continue
            for counter in range(len(new_mol)):
                n_mol = Chem.Mol(new_mol[counter])
                i = order[counter]
                k = b.GetOtherAtomIdx(i)
                if i == j:
                    continue
                if mol_inp.GetAtomWithIdx(i).GetFormalCharge() > 0:
                    continue

                if sink_type == 'proton':
                    p_mol = transfer_proton(n_mol, j, i)
                    connected_mols.append(p_mol)
                    mechanisms.append([5, i, None, k, j])

                    continue
                if sink_type == 'CH':
                    # Same proton transfer behaviour as above
                    mechanisms, connected_mols = c_h_elimination(n_mol, i, j, mechanisms, connected_mols)
                    continue
                   
                connected_mols.append(a_to_b(Chem.Mol(n_mol), i, j))
                mechanisms.append([2, i, j, k, None])



    #bonds to bonds
    for source_idx, source_type in sources_bonds.items():
        b1 = mol_inp.GetBondWithIdx(source_idx)
        a1_list = [b1.GetBeginAtom(), b1.GetEndAtom()]
        new_mol, order = break_bond_source(mol_inp, b1, a1_list)

        if new_mol == []:
            continue
        
        for sink_idx, sink_type in sinks_bonds.items():
            if sink_idx == source_idx:
                continue
            b2 = mol_inp.GetBondWithIdx(sink_idx)
            a2_list = [b2.GetBeginAtom(), b2.GetEndAtom()]
            if b2.GetProp('sink_bond_atom') == 'both':
                for a in a2_list:
                    mechanisms, connected_mols = break_bond_sink(new_mol, order, a, b1, b2, mechanisms, connected_mols)
                        
            else:
                a = mol_inp.GetAtomWithIdx(b2.GetIntProp('sink_bond_atom'))
                mechanisms, connected_mols = break_bond_sink(new_mol, order, a, b1, b2, mechanisms, connected_mols)

    # Rule 5: Forming carbocations

    for sink_idx, sink_type in sinks_bonds.items():
        mechanisms, connected_mols = carbocat(mol_inp, sink_idx, connected_mols, mechanisms)

    connected_mols, mechanisms = clean_mol(connected_mols, mechanisms, mol_inp)

    if screen_charges_enabled:
        charge = 0
        for atom in mol_inp.GetAtoms():
            charge += atom.GetFormalCharge()
        if os.environ.get("AIM_SHARP_CHARGE_SCREEN") == "1":
            clean_list = screen_charges_sharp(connected_mols, charge, mol_inp)
        else:
            clean_list = screen_charges(connected_mols, charge)
    else:
        clean_list = list(range(len(connected_mols)))
    
    # A candidate RDKit cannot sanitise has an impossible valence: drop it and keep the rest.
    # (This used to save debug images to a path on Richard's machine; without Chem.Draw imported
    # that raised, and the caller lost every candidate for the state.)
    valid = []
    for idx, mol in enumerate(connected_mols):
        try:
            Chem.SanitizeMol(mol)
            if kekulise:
                Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            continue
        valid.append(idx)
    if len(valid) < len(connected_mols):
        new_pos = {old: new for new, old in enumerate(valid)}
        clean_list = [new_pos[i] for i in clean_list if i in new_pos]
        connected_mols = [connected_mols[i] for i in valid]
        mechanisms = [mechanisms[i] for i in valid]

    return (connected_mols, mechanisms, mol_inp, clean_list)
