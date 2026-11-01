"""How much of mech-USPTO-31k did AIM's training data already see?

Compares every labelled training state (sm.pkl in training/, finetuning/, finetuning_wittig/,
training_sets/*/) with every state along the mech-USPTO gold mechanisms (the reactants plus
the state after each step), by canonical SMILES without atom maps or stereo, at two levels:
  state   - the whole mixture is identical
  substrate - the largest molecule of the state is identical (same substrate, any reagents)

python -m benchmark.training_overlap --converted ../runs/.../converted_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
root_dir = Path(__file__).resolve().parents[1]


def canonical(mol) -> tuple[str, str] | None:
    """(whole state, largest fragment) as map-free, stereo-free canonical SMILES."""
    try:
        m = Chem.Mol(mol)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(m)
        Chem.SanitizeMol(m)
        m = Chem.RemoveHs(m)
        frags = Chem.GetMolFrags(m, asMols=True)
        largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        return Chem.MolToSmiles(m), Chem.MolToSmiles(largest)
    except Exception:
        return None


def training_states() -> dict[str, list]:
    out = {}
    for name in ["training", "finetuning", "finetuning_wittig"] + [f"training_sets/{p.name}" for p in (root_dir / "training_sets").glob("*")]:
        keys = []
        for sm in sorted((root_dir / name).glob("*/sm.pkl")):
            try:
                with open(sm, "rb") as f:
                    key = canonical(pickle.load(f))
            except Exception:
                key = None
            if key:
                keys.append(key)
        out[name] = keys
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted", required=True, help="converted_full.jsonl from convert_mech_uspto")
    parser.add_argument("--data", default=str(root_dir / "data" / "mech-USPTO-31k.csv"))
    args = parser.parse_args()

    train = training_states()
    all_states = {k for keys in train.values() for k, _ in keys}
    all_subs = {s for keys in train.values() for _, s in keys}
    for name, keys in train.items():
        print(f"{name:28s} {len(keys):5d} states")
    print(f"{'total unique':28s} {len(all_states):5d} states, {len(all_subs)} substrates")

    df = pd.read_csv(args.data)
    hit_state = hit_sub = n = 0
    by_class = Counter()
    class_n = Counter()
    for line in open(args.converted):
        rec = json.loads(line)
        reac = Chem.MolFromSmiles(df.loc[rec["row"], "updated_reaction"].split(">>")[0])
        keys = [canonical(reac)] if reac else []
        for s in rec["steps"]:
            if s["smiles"]:
                m = Chem.MolFromSmiles(s["smiles"])
                if m:
                    keys.append(canonical(m))
        keys = [k for k in keys if k]
        n += 1
        class_n[rec["class"]] += 1
        st = any(k in all_states for k, _ in keys)
        sb = any(s in all_subs for _, s in keys)
        hit_state += st
        hit_sub += sb
        by_class[rec["class"]] += sb
    print(f"\n{n} USPTO reactions")
    print(f"  some state identical to a training state : {hit_state} ({100 * hit_state / n:.1f} %)")
    print(f"  some substrate identical to a training one: {hit_sub} ({100 * hit_sub / n:.1f} %)")
    print("\nclasses with the most substrate overlap:")
    for cls, k in by_class.most_common(12):
        print(f"  {cls:52s} {k:5d} / {class_n[cls]:5d}")


if __name__ == "__main__":
    main()
