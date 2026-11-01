from benchmark.path_benchmark import has_carbon, norm_frags, norm_key


def test_h2_written_one_way():
    # the converter keeps H2 as [H][H], AIM writes [HH]; both must give the same key (J26 bug)
    assert norm_key("[H][H]") == norm_key("[HH]")
    assert norm_frags("CC[O-].[H][H].[Na+]") == norm_frags("[Na+].[HH].CC[O-]")


def test_norm_key_keeps_other_hydrogens():
    assert norm_key("[H-]") != norm_key("[HH]")
    assert norm_key("[H+]") != norm_key("[H-]")
    assert norm_key("[CH3:1][OH:2]") == norm_key("CO")


def test_has_carbon():
    assert has_carbon("O=C(NC1CCCCC1)NC1CCCCC1")
    assert not has_carbon("[OH][Na]")
