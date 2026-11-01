from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "SPArrow"))

from beam_search import Scorer, beam_search, fragment_keys  # noqa: E402

START = "[CH3:1][O-:2].[CH3:3][CH2:4][Br:5]"


def check_sn2(scorer):
    paths, info = beam_search(START, fragment_keys("CCOC"), scorer, depth=2, beam=5, k=3)
    assert paths, "no mechanism found"
    best = paths[0]
    assert best.moves == [[1, 1, 3, None, 4]]       # O- lone pair -> C, C-Br breaks: SN2 in one move
    assert best.logp > -1.0


def test_sn2_v121():
    check_sn2(Scorer("v121"))


def test_sn2_rank_v1():
    check_sn2(Scorer("ranker", str(root / "models" / "rank_v1" / "ranker.json")))
