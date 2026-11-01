"""States where some candidate fails sanitisation must still return the valid candidates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem  # noqa: E402

from shared.rules import run_next_step  # noqa: E402


def test_candidates_survive_an_unsanitisable_one():
    # mech-USPTO row 1536 (Swern), state before step 5 (J14): the CO by-product makes some
    # candidates unsanitisable; the original code raised and lost all 522 valid candidates.
    smi = ("[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][c:7]1[cH:8][cH:9][c:10](-[c:11]2[cH:12][cH:13]"
           "[c:14]([C@H:15]3[CH2:16][CH2:17][C@H:18]([CH2:19][OH:20])[CH2:21][CH2:22]3)[cH:23][cH:24]2)"
           "[n:25][cH:26]1.[Cl-:301].[C:302]=[O:303].[C:304](=[O:305])=[O:309].[Cl:306][S+:308]([CH3:307])C"
           ".CC[N:310](CC)CC")
    connected, mechs, mol, clean = run_next_step(smi, True, None, False, True)
    assert len(connected) == len(mechs) > 0
    assert all(i < len(connected) for i in clean)
    for m in connected:
        Chem.SanitizeMol(Chem.Mol(m))   # every returned candidate is a valid structure
