from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
import numpy as np
import time
from anytree import Node, RenderTree
from anytree.search import find
import xgboost
import os
import pickle
from multiprocessing import Pool, cpu_count, Manager
import itertools
from collections import Counter

# --- SLURM + scratch setup ---
jobid = os.environ.get("SLURM_JOB_ID", "backup")
scratch = os.environ.get("SCRATCH", os.getcwd())   # <<< NEW
os.makedirs(scratch, exist_ok=True)                # <<< NEW
filename = os.path.join(scratch, f"{jobid}")   # <<< CHANGED

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Now you can import it normally
from shared.rules import run_next_step
from shared.next_step import follow_on
from shared.ml_helper import make_fingerprint



def product_subset(reactant, product):
    reactant_counts = Counter(atom.GetSymbol() for atom in reactant.GetAtoms())
    product_counts = Counter(atom.GetSymbol() for atom in product.GetAtoms())

    for element, count in product_counts.items():
        if reactant_counts[element] < count:
            return False
    return True




last_save = time.time()
save_interval = 3600


def n_workers():
    """CPUs this process may use: the Slurm allocation on a cluster, not the whole node."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return cpu_count()

# Module-level so it can be pickled under the "spawn" start method (Windows/macOS),
# where workers do not inherit globals set in the parent process
def init_worker(model_path, bitvectors, inchis):
    global clf, unique_bitvectors, inchis_set
    clf = xgboost.XGBClassifier()
    clf.load_model(model_path)
    unique_bitvectors = bitvectors
    inchis_set = inchis
    RDLogger.DisableLog("rdApp.*")

# Save the tree before job timeout
def save_tree(signum, frame, job_name, product_nodes):
    print("Terminating safely")
    if product_nodes == []:
        print("No products found, not saving tree")
        #sys.exit(0)
    
    for i, p_n in enumerate(product_nodes):
        with open(f"{filename}_{job_name}_{i}.pkl", "wb") as f:
            pickle.dump(p_n, f)
    #sys.exit(0)


def next_nodes(packed):
    unique_mech_count_local = 0
    test_set_mech_count_local = 0
    inchi_set_local = set()
    local_id_counter = itertools.count()
    args, found_event = packed
    smiles, name, aromatic, inchi2, force_depth, force_correct_last_step, follow_on_enabled, fragment, no_duplicates, screen_charges_enabled = args
    results = []
    found_target = False
    if (found_event.is_set() and not force_depth):
        return (results, (unique_mech_count_local, test_set_mech_count_local, inchi_set_local))
    mol = Chem.MolFromSmiles(smiles)
    
    idx_to_mapnum = {}

    for atom in mol.GetAtoms():
        m = atom.GetAtomMapNum()
        if m == 0:
            raise ValueError("All atoms must have atom map numbers")
        i = atom.GetIdx()
        idx_to_mapnum[i] = m
    try:
        product_mols, mechs, mol_inp, clean_list = run_next_step(
            smiles, aromatic, None, screen_charges_enabled, True
        )
        # print(len(product_mols), "products generated")
    except Exception as e:
        print("Error in rule application")
        print(e)
        return (results, (unique_mech_count_local, test_set_mech_count_local, inchi_set_local))
    for i, product in enumerate(product_mols):
        if mechs[i][0] != 11:
            fp = np.array(make_fingerprint(mol_inp, mechs[i]))
            key = tuple(np.where(np.isnan(fp), None, fp))
            if key in unique_bitvectors:
                test_set_mech_count_local += 1
            else:
                unique_mech_count_local += 1
            pred = clf.predict(fp.reshape(1, -1))[0]
        else:
            pred = 1
        found_frag = False
        try:
            inchi1 = Chem.MolToInchi(product, options='/FixedH')
            if no_duplicates and inchi1 in inchis_set: 
                continue
        except Exception as e:
            print(f'Error processing product {i}: {e}')
            continue
        if fragment:
            for m_frag in Chem.GetMolFrags(product, asMols=True):
                try:
                    inchi_frag = Chem.MolToInchi(m_frag, options='/FixedH')
                except Exception:
                    print("No Inchi for fragment")
                    continue
                if inchi_frag == inchi2:
                    found_frag = True
                    # print(f"fragment inchi: {inchi_frag}")

         # Atom-map + SMILES for storage
        for j, atom in enumerate(product.GetAtoms()):  # <<< NEW
            try:
                atom.SetAtomMapNum(idx_to_mapnum[j])
            except KeyError:
                max_map_num = max(idx_to_mapnum.values())
                if mechs[i][0] == 11 and atom.GetAtomicNum() == 1:  
                    atom.SetAtomMapNum(max_map_num + 1)
                    idx_to_mapnum[j] = max_map_num + 1
                    continue
                print(f"Atom index {j} not found in idx_to_mapnum, skipping atom mapping for this product.")
                for atom in product.GetAtoms():
                    atom.SetAtomMapNum(0)
                break

        prod_smiles = Chem.MolToSmiles(product, canonical=False)  # <<< NEW
        
        

        if inchi1 == inchi2 or found_frag:   
            # print('Found the target molecule!')
            # print(f'Product SMILES: {prod_smiles}')   # <<< CHANGED
            # print(f'Inchi: {inchi1}')
            if force_correct_last_step and pred != 1:
                continue
            found_target = True
            found_event.set()
            prod_id = next(local_id_counter)
            results.append({
                    "parent_id": None,
                    "node_id": prod_id,
                    "parent":name,
                    "smiles":prod_smiles,               # <<< CHANGED
                    "mech":mechs[i],
                    "product":True
                    })
            
        elif pred == 1:

            if no_duplicates:
                inchi_set_local.add(inchi1)
            
            prod_id = next(local_id_counter)
            clean = i in clean_list
            results.append({
                "parent_id": None,
                "node_id": prod_id,
                "parent": name,
                "smiles": prod_smiles,
                "mech": mechs[i],
                "product": False,
                "pass_charge_screen": clean
            })

            if follow_on_enabled and mechs[i][0] != 11:   # Don't do follow on for simple proton transfers
                #print(prod_smiles, mechs[i])
                try:
                    f_mols, f_mechs, f_fps = follow_on(product, mechs[i], clf, screen_charges_enabled=screen_charges_enabled, clean_parent=clean)
                except Exception as e:
                    print(e)
                    continue

                for j, fm in enumerate(f_mols):
                    found_frag = False
                    if tuple(f_fps[j]) in unique_bitvectors:
                        test_set_mech_count_local += 1
                    else:
                        unique_mech_count_local += 1
                    try:
                        inchi_f = Chem.MolToInchi(fm, options='/FixedH')
                        if no_duplicates and (inchi_f in inchis_set) or (inchi_f in inchi_set_local):
                            continue
                        else:
                            if no_duplicates:
                                inchi_set_local.add(inchi_f)
                    except Exception:
                        print("No Inchi")
                        continue
                    if fragment:
                        for m_frag in Chem.GetMolFrags(fm, asMols=True):
                            try:
                                inchi_frag = Chem.MolToInchi(m_frag, options='/FixedH')
                            except Exception:
                                print("No Inchi for fragment")
                                continue
                            if inchi_frag == inchi2:
                                found_frag = True
                                # print(f"fragment inchi: {inchi_frag}")
                    
                    

                    for k, atom in enumerate(fm.GetAtoms()):
                        atom.SetAtomMapNum(idx_to_mapnum[k])
                    fm_smiles = Chem.MolToSmiles(fm, canonical=False)


                    if inchi_f == inchi2 or found_frag:
                        if no_duplicates:
                            inchi_set_local.remove(inchi_f)
                        follow_on_id = next(local_id_counter)
                        found_target = True
                        found_event.set()
                        # print('Found the target molecule!')
                        # print(f'Product SMILES: {prod_smiles}')   # <<< CHANGED
                        # print(f'Inchi: {inchi_f}')
                        results.append({
                            "parent_id": prod_id,
                            "node_id": follow_on_id,
                            "parent":name,
                            "smiles":fm_smiles,               # <<< CHANGED
                            "mech": f_mechs[j],
                            "product":True})
                        if not force_depth:
                            break
                        continue
                    follow_on_id = next(local_id_counter)
                    results.append({
                        "parent_id": prod_id,
                        "node_id": follow_on_id,
                        "parent": name,
                        "smiles": fm_smiles,
                        "mech": f_mechs[j],
                        "product": False
                    })

        if found_target and not force_depth:
            break
    # del mol
    global_var_updates = (unique_mech_count_local, test_set_mech_count_local, inchi_set_local)
    
    return (results, global_var_updates)





def run_parallel(r_smiles, p_smiles, job_name, no_duplicates=False, force_depth= False, depth_limit=3, follow_on_enabled=True, aromatic=True, screen_charges_enabled=False, hypervalent_enabled=False):

    # Set up
    sm_node = None
    global inchis_set
    inchis_set = set()
    global unique_bitvectors
    unique_bitvectors = set()
    unique_mech_count = 0
    test_set_mech_count = 0
    start_time_total = time.time()

    model_name = "v121"
    print("Using model ", model_name)
    
    model_dir = root_dir / "models" / model_name
    with open(model_dir / "unique_bitvectors.txt", "r") as f:
        for line in f:
            fp_elements = []
            for x in line.strip().split(','):
                if x == 'nan':
                    fp_elements.append(None)
                else:
                    fp_elements.append(np.float64(x))
                
            unique_bitvectors.add(tuple(fp_elements))


    

    # +
    # Parameters
    # Now featured as arguments
    force_correct_last_step = True
    # Initialise variables
    step_no = 0
    product_nodes = []
    times_per_layer = []
    signal_break = False
    unique_id = 0
    last_save = time.time()
    fragment = False


    # --- Initial molecules ---
    sm_smiles = r_smiles
    end_smiles = p_smiles

    # print("Starting SMILES:", sm_smiles)
    # print("Ending SMILES:", end_smiles)

    sm = Chem.MolFromSmiles(sm_smiles)
    end = Chem.MolFromSmiles(end_smiles)

    f1 = rdMolDescriptors.CalcMolFormula(sm)
    f2 = rdMolDescriptors.CalcMolFormula(end)

    if f1 != f2:
        subset = product_subset(sm, end)
        if not subset:
            print(f"Molecular formula mismatch: {f1} != {f2}")
            raise ValueError("Molecular formula mismatch, stoichiometry error")
        else:
            new_end = Chem.GetMolFrags(end, asMols=True)[0]
            end = new_end
            fragment = True
            print("Target is now ", Chem.MolToSmiles(end))
        


    # Atom-map the starting molecule
    for i, atom in enumerate(sm.GetAtoms(), start=1):   # <<< NEW
        atom.SetAtomMapNum(i)

    sm_smiles_mapped = Chem.MolToSmiles(sm, canonical=False)  # <<< NEW
    sm_node = Node("SM", smiles=sm_smiles_mapped)             # <<< CHANGED

    # print(sm_node.smiles)

    nodes = [sm_node]

    

    inchi2 = Chem.MolToInchi(end, options='/FixedH')
    RDLogger.DisableLog('rdApp.*')

    model_path = str(model_dir / "xg_model.json")


    while step_no < depth_limit:

        start_t = time.time()
        print(f'step no {step_no + 1}')
        
        tasks = [
            (
                int_node.smiles,
                int_node.name,
                aromatic,
                inchi2,
                force_depth,
                force_correct_last_step,
                follow_on_enabled, 
                fragment,
                no_duplicates,
                screen_charges_enabled
            )
            for int_node in nodes
            if not hasattr(int_node, "product")
        ]

        with Manager() as manager:
            found_event = manager.Event()
            with Pool(processes=n_workers(), initializer=init_worker,
                      initargs=(model_path, unique_bitvectors, inchis_set)) as pool:
                all_outputs = pool.map(next_nodes, [(task, found_event) for task in tasks])
        
        outputs = [a[0] for a in all_outputs]
        gbl_updates = [a[1] for a in all_outputs]



        for results in outputs:
            if results == []:
                continue
            temp_nodes = {}
            nodes_with_children = {}
            # node to which all new entries will be attached to
            parent_node_master = find(sm_node, lambda n: n.name == results[0]["parent"])
            # Save each new entry as a standalone node
            for res in results:
                unique_id += 1

                temp_nodes[res["node_id"]] = Node(
                    f"Mol {unique_id}",
                    smiles=res["smiles"],
                    mech=res["mech"],
                )
               
            
            for res in results:
                node = temp_nodes[res["node_id"]]
                # Attach all follow on products to their respective parent
                if res["parent_id"] is not None:
                    parent_node = temp_nodes[res["parent_id"]]
                    nodes_with_children[res["parent_id"]] = temp_nodes[res["parent_id"]]
                    node.parent = parent_node
                # Terminate early if product is found and force_depth = False
                    if "product" in res and res["product"]:
                        node.product = True
                        # print(node)
                        product_nodes.append(node)
                        if not force_depth:
                            signal_break = True

            for res in results:
                # Look at all top level nodes
                if res["parent_id"] is None:
                    # add the 2-level trees
                    if res["node_id"] in nodes_with_children:
                        node = temp_nodes[res["node_id"]]
                        node.parent = parent_node_master
                    # add the top level nodes if they pass the charge screen, so they are leaf nodes
                    if "pass_charge_screen" in res and res["pass_charge_screen"]:
                        node = temp_nodes[res["node_id"]]
                        unique_id += 1
                        duplicate_node = Node(
                            f"Mol {unique_id}",
                            smiles=res["smiles"],
                            mech=res["mech"],
                            parent=parent_node_master
                        )
                    if "product" in res and res["product"]:
                        node = temp_nodes[res["node_id"]]
                        unique_id += 1
                        duplicate_node = Node(
                            f"Mol {unique_id}",
                            smiles=res["smiles"],
                            mech=res["mech"],
                            parent=parent_node_master,
                            product=True
                        )
                        # print(duplicate_node)
                        product_nodes.append(duplicate_node)
                        if not force_depth:
                            signal_break = True
            
            


        for update in gbl_updates:
            unique_mech_count += update[0]
            test_set_mech_count += update[1]
            for inchi_u in update[2]:
                inchis_set.add(inchi_u)

        if signal_break:
            break
        
        for int_node in nodes:
            if not int_node.children and not (hasattr(int_node, "product") and int_node.product):
                int_node.parent = None

        if not sm_node.leaves:
            print("no more products can be formed, ending search")
            break
        

        nodes = sm_node.leaves


        if time.time() - last_save > save_interval:
            with open(filename, "wb") as f:
                pickle.dump(sm_node, f)

            print(f"[Checkpoint] Tree saved to {filename}")

            with open(f'{filename}.txt', 'w') as f:     # <<< CHANGED
                for pre, fill, node in RenderTree(sm_node):
                    f.write(f"{pre}{node.name} - {node.smiles}\n")  # <<< CHANGED

            last_save = time.time()

        step_no += 1
        times_per_layer.append(time.time() - start_t)

    print("search is over")

    for i, dur in enumerate(times_per_layer):
        print(f'Layer {i+1} took {dur:.2f} seconds')

    with open(f'{filename}_{job_name}.txt', 'w') as f:     # <<< CHANGED
            for pre, fill, node in RenderTree(sm_node):
                f.write(f"{pre}{node.name} - {node.smiles} \n")  # <<< CHANGED
                if hasattr(node, "mech"):
                    f.write(f"{pre}Mechanism: {node.mech}\n")

    for n in product_nodes:
        print(n)

    save_tree(None, None, job_name, product_nodes)
    save_tree(None, None, str(job_name) + "total", [sm_node])
    print(f"Total time taken: {time.time() - start_time_total:.2f} seconds")
    print(f"Unique mechanisms found: {unique_mech_count}")
    print(f"Test set mechanisms found: {test_set_mech_count}")

    return None
