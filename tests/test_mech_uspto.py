from pathlib import Path
import sys

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.mech_uspto import convert, map_to_aim_index, parse_label  # noqa: E402
from shared.rules import run_next_step  # noqa: E402


def moves(conv):
    return [s.aim_moves for s in conv.steps]


def test_bond_sink_is_normalised_to_far_end():
    (arrow,) = parse_label("[(201, [201, 101])]")
    assert arrow.kind == "lp" and arrow.sink == 101


def test_sn2_single_step():
    c = convert("[Br:3][CH2:2][CH3:1].[O-:4][CH3:5]>>[CH3:1][CH2:2][O:4][CH3:5].[Br-:3]",
                "[(4, 2), ([2, 3], 3)]")
    assert c.final_match
    assert moves(c) == [[[1, 4, 2, None, 3]]]


def test_proton_transfer_uses_heavy_atom_of_proton():
    c = convert("[OH2:1].[BrH:2]>>[OH3+:1].[Br-:2]", "[(1, 2.1), ([2.1, 2], 2)]")
    assert c.final_match
    assert moves(c) == [[[4, 1, None, None, 2]]]


def test_hydride_reduction_then_protonation():
    # NaBH4 / MeOH reduction of a ketone (row 5 of mech-USPTO-31k, trimmed)
    c = convert("[CH3:1][C:2](=[O:3])[CH3:4].[BH4-:301].[CH3:5][OH:302]>>[CH3:1][CH:2]([OH:3])[CH3:4]",
                "[([301, 301.1], 2), ([2, 3], 3), (3, 302.1), ([302.1, 302], 302)]")
    assert c.final_match
    assert moves(c) == [[[7, None, 2, 301, 3]], [[4, 3, None, None, 302]]]


def test_concerted_push_is_split_into_octet_valid_aim_moves():
    # amine + carbonyl with the pi bond taking H+ in the same step: AIM needs addition then protonation
    c = convert("[NH3:1].[CH3:2][CH:3]=[O:4].[H+:5]>>[NH3+:1][CH:3]([CH3:2])[OH:4]",
                "[(1, 3), ([3, 4], 5)]")
    assert c.final_match
    (step,) = c.steps
    assert step.n_splits == 1
    assert step.aim_moves == [[1, 1, 3, None, 4], [0, 4, 5, None, None]]


def test_converted_move_is_generated_by_aim_rules():
    """The converted first move must be one of the candidates AIM's rule engine proposes."""
    reac = "[Br:3][CH2:2][CH3:1].[O-:4][CH3:5]"
    c = convert(reac + ">>[CH3:1][CH2:2][O:4][CH3:5].[Br-:3]", "[(4, 2), ([2, 3], 3)]")
    m2i = map_to_aim_index(reac)
    move = c.steps[0].aim_moves[0]
    want = [move[0]] + [None if x is None else m2i[x] for x in move[1:]]

    mol = Chem.MolFromSmiles(reac)
    _, mechs, _, _ = run_next_step(None, kekulise=True, chain_mol=mol, screen_charges_enabled=False)
    assert want in mechs


def test_bare_proton_loss_is_flagged_not_mapped():
    # E1 from a t-butyl cation written as a C-H bond pushing in, releasing H+ with no base
    c = convert("[CH3:1][C+:2]([CH3:3])[CH3:4]>>[CH2:1]=[C:2]([CH3:3])[CH3:4]",
                "[([1.1, 1], 2)]")
    (step,) = c.steps
    assert step.unmatched and not step.aim_moves
