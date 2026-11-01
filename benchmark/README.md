# Benchmark: mech-USPTO-31k → AIM

`data/mech-USPTO-31k.csv` labels each reaction with a flat list of curly arrows.
`mech_uspto.py` turns those arrows into AIM's own move tuples `[type, i, j, k, l]`
(see `shared/rules.py`), so AIM can be scored step by step and the data can be used for training.

```bash
python -m benchmark.convert_mech_uspto --out converted.jsonl   # coverage report (~6 min)
python -m benchmark.step_recall --limit 300                     # does AIM propose/keep the gold step?
python -m pytest tests
```

## Label format

Each arrow is `(source, sink)` over atom-map numbers:

| token | meaning |
|---|---|
| `a` (source) | lone pair on atom `a` |
| `[x, y]` (source) | electrons of bond x–y; they end up on **y**'s side |
| `z` (sink) | atom z: a new bond y–z (or a–z), or a lone pair on z if z is the source end |
| `[p, q]` (sink) | bond p–q; one end is the source end, so it is equivalent to sinking on the other end |
| `x.d` | the d-th hydrogen on heavy atom x (implicit in the SMILES) |

## How conversion works

1. **Electron bookkeeping.** The reactants are loaded with explicit H and kekulised. Each arrow moves one
   electron pair. Formal charges follow from electron ownership: a lone pair counts 2 for its atom, a bond
   counts 1 for each end, and the charge changes by (owned before − owned after).
2. **Step segmentation.** The labels don't mark where steps end. Arrows are applied one at a time, and a
   step closes as soon as every atom it touched obeys the octet rule again (duet for H, ≤ 12 electrons
   past period 2; transition metals are not checked).
3. **Template matching.** Each step's arrows are covered by AIM templates (types 0–11), with the
   constraint that **every intermediate between AIM moves is octet-valid**, because AIM can never visit
   a pentavalent carbon. When no cover exists, a concerted arrow `[x,y] → z` is split into
   `[x,y] → y` followed by `y → z`, which is the same electron flow done in two moves. That is how
   AIM's `follow_on` builds such steps. The search is depth-first, templates before splits.
4. **Validation.** The final state is compared with the recorded product: exactly, and up to
   protonation state.

## Results on all 31,364 reactions

| | |
|---|---|
| final state = product (exact) | 64.7 % |
| final state = product up to protonation | 100 % |
| steps mapped to AIM moves | 97.3 % of 115,479 (91 % one AIM move, the rest 2–4 moves for concerted pushes) |
| reactions with every step mapped | 90.0 % |
| unmatched steps | 3,139, almost all bare-H⁺ release (X–H → X⁻ + H⁺ with no base), for which AIM has no move |

The exact-match gap is in the dataset, not the converter:
- **Arrows stop before the last proton transfer** (SN2, SN1, SNAr, acyl substitution,
  O-demethylation): they end at R–NH₃⁺ or R₂NH⁺–COR, while the product is the neutral amine.
  A benchmark run must either target the protonated state or allow a final work-up deprotonation.
- **Atom maps that contradict the arrows** (all Vilsmeier reactions): the arrows put water's O
  (map 304) into the aldehyde, but the product maps it as the DMF O (map 2). These are matched by
  unmapped SMILES.

## Outputs

`--out` writes one JSON line per reaction:
`{row, class, final_match, match_up_to_protons, fully_mapped, steps: [{n_arrows, aim_moves, n_splits, smiles}]}`.
Moves use **atom-map numbers**. `map_to_aim_index(reactant_smiles)` converts them to the atom indices
`run_parallel` uses. Those indices stay fixed along a path, because AIM never reorders atoms.

## Caveats / to do
- Step segmentation is greedy. A concerted step whose midpoint happens to be octet-valid would be cut in two.
  It hasn't been seen in the validation so far, but it is worth spot-checking a few hundred steps by eye.
- Stereochemistry is ignored throughout.
- The H label `x.d` is resolved to "an H currently on x"; the d-index is only used to keep labels distinct.
