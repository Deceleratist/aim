"""Plan A8: a gold step counts as reachable when its product state is generated, whatever the label."""
from pathlib import Path

import xgboost
from rdkit import RDLogger

from benchmark.ranking_data import process_step

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def test_silicate_reachable_under_another_label():
    clf = xgboost.XGBClassifier()
    clf.load_model(str(ROOT / "models" / "v121" / "xg_model.json"))
    # TBS ether + F-: the pentacoordinate silicate loses the alkoxide (gold type 10 at Si)
    start = "CC(C)(C)[Si-:101](C)(C)([O:1][CH2:2][CH3:3])[F:301]"
    gold_next = "CC(C)(C)[Si](C)(C)F.CC[O-]"
    rec, _, _ = process_step(start, [10, None, None, 101, 1], clf, gold_next)
    assert not rec["proposed"]          # move label differs (clean_mol kept a type-3 duplicate)
    assert rec["state_proposed"]        # but the product state is generated
    assert 3 in rec["state_types"]


def test_no_state_check_for_multi_move_steps():
    clf = xgboost.XGBClassifier()
    clf.load_model(str(ROOT / "models" / "v121" / "xg_model.json"))
    rec, _, _ = process_step("C[O-:1].[CH3:2][CH2:3][Br:4]", [1, 1, 3, None, 4], clf)
    assert "state_proposed" not in rec
