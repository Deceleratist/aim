from benchmark.path_benchmark import norm_key


def test_phosphine_oxide_forms_match():
    assert norm_key("[O-][P+](c1ccccc1)(c1ccccc1)c1ccccc1") == norm_key("O=P(c1ccccc1)(c1ccccc1)c1ccccc1")


def test_sulfoxide_and_ylide_forms_match():
    assert norm_key("C[S+](C)[O-]") == norm_key("CS(C)=O")
    assert norm_key("[CH2-][P+](c1ccccc1)(c1ccccc1)c1ccccc1") == norm_key("C=P(c1ccccc1)(c1ccccc1)c1ccccc1")


def test_real_zwitterions_untouched():
    # N-oxide / nitro and the Mitsunobu betaine (P+ and N- not bonded) stay as they are
    assert norm_key("C[N+](C)(C)[O-]") == "C[N+](C)(C)[O-]"
    assert "+" in norm_key("CCOC(=O)[N-]N(C(=O)OCC)[P+](c1ccccc1)(c1ccccc1)c1ccccc1")
    assert norm_key("[HH].O") == "O.[HH]"
