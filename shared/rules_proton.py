from rdkit import Chem # pyright: ignore[reportMissingImports]
from datetime import datetime
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

def screen_charges(list_mol, mechanisms, charge = 0):
    new_mols = []
    new_mechs = []
    if charge == -1: # base mediated reaction will not have cationic intermediates
        for idx in range(len(list_mol)):
            mol = list_mol[idx]
            keep = True
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() > 0:
                    keep = False
                    break
            if keep:
                new_mols.append(list_mol[idx])
                new_mechs.append(mechanisms[idx])
        return (new_mols, new_mechs)
    elif charge == 1: # acid mediated reaction will not have anionic intermediates
        for idx in range(len(list_mol)):
            mol = list_mol[idx]
            keep = True
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() < 0:
                    keep = False
                    break
            if keep:
                new_mols.append(list_mol[idx])
                new_mechs.append(mechanisms[idx])
        return (new_mols, new_mechs)




def identify_source(mol, rules = "all"):
    for atom in mol.GetAtoms():
        # Look for lone pairs
        valence_electrons = ptable.GetNOuterElecs(atom.GetAtomicNum())
        lone_electrons = valence_electrons - atom.GetTotalValence() - atom.GetFormalCharge()
        if lone_electrons > 0:
            atom.SetProp('source', 'lone pair')
  

def identify_sink(mol, rules = "all", hypervalent_enabled = True):
   
    # Look for vacant orbitals
    for atom in mol.GetAtoms():
        if electronegativity_dict[atom.GetAtomicNum()] > 2.55 and atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() >= 0:
            atom.SetProp('sink', 'proton')


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

def prepare_molecule(mol, hypervalent_enabled = True):
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)
    identify_acidic_CH(mol)
    identify_sink(mol, hypervalent_enabled=hypervalent_enabled)
    identify_source(mol)
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
    a2 = p_mol.GetAtomWithIdx(i)
    a2.SetFormalCharge(a2.GetFormalCharge() + 1)
    if a2.GetNumExplicitHs() > 0 or a2.GetFormalCharge() == 0:
        a2.SetNumExplicitHs(a2.GetNumExplicitHs() + 1)
    return p_mol
    #return (mechanisms, connected_mols)


def c_h_elimination(mol, i, j, mechanisms, connected_mols):
    c_mol = Chem.Mol(mol)
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
                mechanisms.append([8, i, j, None, l])

        if atom2.GetFormalCharge() > 0 and atom2.HasProp('sink') and atom2.GetProp('sink')== 'empty AO':
            k = atom2.GetIdx()
            mechanisms.append([9, i, j, None, None])
            connected_mols.append(a_to_b(Chem.Mol(c_mol), j, k))
    return (mechanisms, connected_mols)
    
def bond_to_hydride(mol, i, j, l, mechanisms, connected_mols):
    m_mol = Chem.Mol(mol)
    m_mol = b_to_a(m_mol, j, l)
    m_mol.GetAtomWithIdx(j).SetNumExplicitHs(m_mol.GetAtomWithIdx(j).GetNumImplicitHs() + 1)
    m_mol.GetAtomWithIdx(j).SetFormalCharge(m_mol.GetAtomWithIdx(j).GetFormalCharge() - 1)
    connected_mols.append(m_mol)
    mechanisms.append([7,None, j, i, l])
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



def execute_step(mol_inp, kekulise = False,  screen_charges_enabled = True):


    connected_mols = []

    mechanisms = []

    sources_atoms = {}
    sinks_atoms = {}

 

    for atom1 in mol_inp.GetAtoms():
        if atom1.HasProp('source'):
            sources_atoms[atom1.GetIdx()] = atom1.GetProp('source')
        if atom1.HasProp('sink'):
            sinks_atoms[atom1.GetIdx()] = atom1.GetProp('sink')


    #atoms to atoms

    for source_idx, source_type in sources_atoms.items():
        i = source_idx
        if mol_inp.GetAtomWithIdx(i).GetFormalCharge() > 0:
            continue
        for sink_idx, sink_type in sinks_atoms.items():
            j = sink_idx

            if mol_inp.GetAtomWithIdx(j).GetFormalCharge() < 0:
                continue
            if i == j:
                continue

            if sink_type == 'proton':
                p_mol = transfer_proton(mol_inp, j, i)
                connected_mols.append(p_mol)
                mechanisms.append([4, i, None, None, j])   
                continue

            if sink_type == 'CH':
                # Same proton transfer behaviour as above
                mechanisms, connected_mols = c_h_elimination(mol_inp, i, j, mechanisms, connected_mols)
                continue

    connected_mols, mechanisms = clean_mol(connected_mols, mechanisms, mol_inp)

    if screen_charges_enabled:
        charge = 0
        for atom in mol_inp.GetAtoms():
            charge += atom.GetFormalCharge()
        connected_mols, mechanisms = screen_charges(connected_mols, mechanisms, charge)
    
    # Drop candidates RDKit cannot sanitise (impossible valence) and keep the rest; see rules.py.
    valid = []
    for idx, mol in enumerate(connected_mols):
        try:
            Chem.SanitizeMol(mol)
            if kekulise:
                Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            continue
        valid.append(idx)
    connected_mols = [connected_mols[i] for i in valid]
    mechanisms = [mechanisms[i] for i in valid]

    return (connected_mols, mechanisms, mol_inp)


def run_next_step(smiles, kekulise = False ,chain_mol = None, screen_charges_enabled = True, hypervalent_enabled = True):
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
        
    prepare_molecule(mol_inp, hypervalent_enabled=hypervalent_enabled)
    connected_mols, mechanisms, mol_inp = execute_step(mol_inp, kekulise, screen_charges_enabled)
    return (connected_mols, mechanisms, mol_inp)
