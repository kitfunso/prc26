from taxi.data import FEATURES, KEYS, LABELS


def test_feature_names_carry_no_label():
    assert not set(FEATURES) & set(LABELS)
    assert "month" not in FEATURES


def test_feature_frame_has_only_declared_columns(feb_features):
    assert list(feb_features.columns) == list(KEYS + FEATURES)
    assert feb_features.id.is_unique
    assert feb_features.z_anchor.notna().all()
