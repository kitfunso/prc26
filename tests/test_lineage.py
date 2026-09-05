from taxi.data import FEATURES, KEYS, LABELS, PLAN


def test_feature_names_carry_no_label():
    assert not set(FEATURES + PLAN) & set(LABELS)
    assert "month" not in FEATURES + PLAN


def test_feature_frame_has_only_declared_columns(feb_features):
    assert list(feb_features.columns) == list(KEYS + FEATURES + PLAN)
    assert feb_features.id.is_unique
    assert feb_features.z_anchor.notna().all()


def test_final_rounds_reads_every_champion_report():
    from taxi.m1 import final_rounds
    from taxi.champ import CHAMPIONS_A

    for _, tag, doc in CHAMPIONS_A.values():
        assert all(r > 0 for r in final_rounds(tag, doc))
