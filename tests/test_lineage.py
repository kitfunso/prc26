from taxi.data import FEATURES, KEYS, LABELS, PLAN, STAND


def test_feature_names_carry_no_label():
    assert not set(FEATURES + PLAN + STAND) & set(LABELS)
    assert "month" not in FEATURES + PLAN + STAND


def test_feature_frame_has_only_declared_columns(feb_features):
    assert list(feb_features.columns) == list(KEYS + FEATURES + PLAN + STAND)
    assert feb_features.id.is_unique
    assert feb_features.z_anchor.notna().all()


def test_final_rounds_reads_every_champion_report():
    from taxi.m1 import final_rounds
    from taxi.champ import CHAMPIONS_A

    for _, tag, doc, _kind in CHAMPIONS_A.values():
        assert all(r > 0 for r in final_rounds(tag, doc))


def test_champion_cache_fingerprint_sees_feature_values(feb_features):
    from taxi.champ import CHAMPIONS_A, _fingerprint

    cols = {c for cs, *_ in CHAMPIONS_A.values() for c in cs}
    a = feb_features[feb_features.has_fpl].head(50).reset_index(drop=True)
    for c in ("d_ae", "w_sknt", "queue"):
        assert c in cols
    for c in sorted({"y", *cols} - set(a.columns)):
        a[c] = 0.0
    b = a.copy()
    b.loc[3, "d_ae"] = b.loc[3, "d_ae"] + 1.0
    assert _fingerprint(a, a) != _fingerprint(b, a)
    assert _fingerprint(a, a) == _fingerprint(a.copy(), a.copy())


def test_verdict_line_promotes_only_against_a_current_champion():
    from taxi.champ import CURRENT
    from taxi.verdict import Verdict
    from taxi.w1 import verdict_line

    v = Verdict(250.0, 240.0, 10.0, 8.0, 12.0)
    assert "MC1 promoted" not in verdict_line("MC1", "W1-A", v) and "NOT promoted" in verdict_line("MC1", "W1-A", v)
    assert f"D1 promoted" in verdict_line("D1", CURRENT["A"], v)
    assert "L1a promoted" in verdict_line("L1a", CURRENT["B"], v)
