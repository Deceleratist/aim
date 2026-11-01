from rdkit import Chem
from .rules import execute_step
from .ml_helper import aromatic_flags, make_fingerprint
from .rules_proton import run_next_step
import numpy as np
import os
ptable = Chem.GetPeriodicTable()
from .electronegativity import electronegativity
import traceback
electronegativity_dict = electronegativity()




def prepare_atoms(atom_i, hypervalent_enabled = True):
    atom_list = [atom_i] + list(atom_i.GetNeighbors())
    b = []
    for a_i in atom_list:
        b = b + list(a_i.GetBonds())
    # Identify sources
    for atom in atom_list:
        # Look for hydrides - M-H bond
        if electronegativity_dict[atom.GetAtomicNum()] < 2.05 and atom.GetFormalCharge() < 0 and atom.GetTotalNumHs() > 0:
           h_num = atom.GetTotalNumHs()
           atom.SetNoImplicit(True)
           atom.SetNumExplicitHs(h_num)
           atom.SetProp('source', 'hydride')

        # Look for lone pairs
        valence_electrons = ptable.GetNOuterElecs(atom.GetAtomicNum())
        lone_electrons = valence_electrons - atom.GetTotalValence() - atom.GetFormalCharge()
        # Boron behaves badly with the way I've used RDKit
        if atom.GetAtomicNum() == 5 and atom.GetTotalDegree() == 4:
            continue
        if lone_electrons > 0:
            atom.SetProp('source', 'lone pair')
            
    for bond in b:
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        # Conditional on rules being on
        if bond.GetBondTypeAsDouble() == 1:
            # This is an experimental feature - unlikely to be a very common reaction mode
            if electronegativity_dict[atoms[0].GetAtomicNum()] < 2.05 and electronegativity_dict[atoms[1].GetAtomicNum()] < 2.05:
                bond.SetProp('source', 'sigma bond')
                bond.SetProp('source_bond_atom', 'both')
                bond.SetProp('source_other_atom', 'both')
            # For any element less electronegative than boron (inclusive) - add sigma bond as source - going from more electronegative atom
            for a in atoms:
                if electronegativity_dict[a.GetAtomicNum()] < 2.05:
                    other = a
                    source = bond.GetOtherAtom(other)
                    # avoid breaking a sigma bond if the source atom instead has a lone pair
                    if a.HasProp('source') and a.GetProp('source') == 'lone pair':
                        continue
                    bond.SetProp('source', 'sigma bond')
                    bond.SetIntProp('source_bond_atom', source.GetIdx())
                    bond.SetIntProp('source_other_atom', other.GetIdx())

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

    for atom in atom_list:

        valence_electrons = ptable.GetNOuterElecs(atom.GetAtomicNum())
        lone_electrons = valence_electrons - atom.GetTotalValence() - atom.GetFormalCharge()
        num_bond_electrons = atom.GetTotalValence()*2

        if lone_electrons + num_bond_electrons < 8 and valence_electrons > 2:
            atom.SetProp('sink', 'empty AO')
        # Special case for s block
        if valence_electrons < 3 and num_bond_electrons/2 < valence_electrons and atom.GetFormalCharge() >= 0:
            atom.SetProp('sink', 'empty AO')

    # The second row p block expansion pack
    if hypervalent_enabled:
        if atom.GetAtomicNum() < 17 and atom.GetAtomicNum() > 13 and atom.GetFormalCharge() >= 0 and atom.GetTotalValence() < 6:
            if atom.HasProp('sink') and atom.GetProp('sink') == 'proton':
                pass
            else:
                atom.SetProp('sink', 'hypervalent')

    for bond in b:
        # Conditional on all rules being swtiched on
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        
        # Identify possible sigma stars by considering leaving groups being any element that has a higher electronegativity than carbon
        if bond.GetBondTypeAsDouble() == 1:
            # TBD if to keep this rule or not - not particularly profound either way
            if electronegativity_dict[atoms[0].GetAtomicNum()] > 2.55 and electronegativity_dict[atoms[1].GetAtomicNum()] > 2.55:
                bond.SetProp('sink', 'sigma star')
                bond.SetProp('sink_bond_atom', 'both')
                bond.SetProp('sink_other_atom', 'both')
            else:
                for a in atoms:
                    # bond is considered a sink if there is an element more electronegative than carbon attached to it
                    if electronegativity_dict[a.GetAtomicNum()] > 2.55:
                        other = a
                        sink = bond.GetOtherAtom(other)
                        # do not add as a sink if there is a empty AO on the sink atom (unlikely case)
                        if a.HasProp('sink') and a.GetProp('sink') == 'empty AO':
                            continue
                        bond.SetProp('sink', 'sigma star')
                        bond.SetIntProp('sink_bond_atom', sink.GetIdx())
                        bond.SetIntProp('sink_other_atom', other.GetIdx())

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
    for atom in atom_list:
        if atom.HasProp('sink') and atom.GetProp('sink') == 'hypervalent':
            atom.SetProp('sink', 'empty AO')
    return None

def follow_on(next_mol, mech, model, screen_charges_enabled = True, hypervalent_enabled = True, clean_parent = True):
    fp_list = []
    clf = model
    smiles_list = []
    mechs_list = []
    accepted_follow_on = []
    for a in next_mol.GetAtoms():
        a.ClearProp('source')
        a.ClearProp('sink')
        if a.GetFormalCharge() == 0:
           a.SetNoImplicit(False)
    for b in next_mol.GetBonds():
        b.ClearProp('source')
        b.ClearProp('sink')
        b.ClearProp('source_bond_atom')
        b.ClearProp('sink_bond_atom')
        b.ClearProp('source_other_atom')

    for a in mech[1:]:
        if a is not None:
            prepare_atoms(next_mol.GetAtomWithIdx(a), hypervalent_enabled=hypervalent_enabled)
    connected_mols = []
    try:
        connected_mols, mechanism, mol_inp, clean_list = execute_step(next_mol, screen_charges_enabled=screen_charges_enabled)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    arom = aromatic_flags(mol_inp) if connected_mols else None
    for i, product in enumerate(connected_mols):
        if mechanism[i][0] == 11:
            # Do not consider H- H+ as part of the follow on steps
            continue
        fp = np.array(make_fingerprint(mol_inp, mechanism[i], arom))
        pred = clf.predict(fp.reshape(1,-1))[0]
        if pred and i in clean_list:
            # If we are relying on the follow-on reaction to satisfy charges, it should only be balanced
            # by resonance arrows, not by 'reaction arrows'
            if not clean_parent:
               # print(mechanism[i])
                if mechanism[i][0] not in [0, 1, 2, 3]:
                    continue
                if mechanism[i][2] not in [n_atom.GetIdx() for n_atom in mol_inp.GetAtomWithIdx(mechanism[i][1]).GetNeighbors()]:
                    continue
               # print("The above mechanism survived")
            accepted_follow_on.append(product)
            mechs_list.append(mechanism[i])
            smiles_list.append(Chem.MolToSmiles(product))
            fp_list.append(fp)
    
    if clean_parent:
        connected_mols_p = []

        try:
            connected_mols_p, mechanism_p, _ = run_next_step(None, chain_mol = next_mol, screen_charges_enabled=screen_charges_enabled, hypervalent_enabled=hypervalent_enabled)
        except Exception as e:
            print(e)
        
        for i, product in enumerate(connected_mols_p):
            fp = np.array(make_fingerprint(mol_inp, mechanism_p[i]))
            pred = clf.predict(fp.reshape(1,-1))[0]
            if pred:
                accepted_follow_on.append(product)
                mechs_list.append(mechanism_p[i])
                smiles_list.append(Chem.MolToSmiles(product))
                fp_list.append(fp)
    
    return (accepted_follow_on, mechs_list, fp_list)
