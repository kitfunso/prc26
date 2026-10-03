from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from champion import fit_member, frame_hash, load_completed, verified_selection, write_record
from campaign import file_hash
from mix import fit_clf
from train import fit_predict


def sample() -> pd.DataFrame:
    return pd.DataFrame({"id": np.arange(120), "x": np.arange(120, dtype=float),
                         "apt": pd.Categorical(["A", "B"] * 60), "y": np.arange(120, dtype=float) ** 1.1})


def test_saved_tree_models_replay_predictions(tmp_path: Path) -> None:
    rows = sample()
    params = {"n_estimators": 5, "num_leaves": 7, "min_child_samples": 5, "verbose": -1, "n_jobs": 1}
    cfg = {"family": "lgb", "params": params, "cat": ["apt"], "num": ["x"],
           "clf": {"family": "lgb", "params": {**params, "objective": "binary"}}}
    path = tmp_path / "reg.model"
    p, _ = fit_predict(cfg, rows, rows, iters=5, model_path=path)
    np.testing.assert_allclose(p, lgb.Booster(model_file=str(path)).predict(rows[["apt", "x"]]))
    path = tmp_path / "clf.model"
    p, _ = fit_clf(cfg, rows, (rows.y > rows.y.median()).to_numpy(int), rows, None, None, 5, model_path=path)
    np.testing.assert_allclose(p[:, 1], lgb.Booster(model_file=str(path)).predict(rows[["apt", "x"]]))


def test_member_rejects_overlap_before_fitting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        fit_member("ra_cb_wz_t7", sample(), sample(), tmp_path)


def test_query_fingerprint_ignores_excluded_labels() -> None:
    rows = sample()
    columns = ["id", "apt", "x"]
    assert frame_hash(rows, columns) == frame_hash(rows.assign(y=-987654.), columns)
    assert frame_hash(rows, [*columns, "y"]) != frame_hash(rows.assign(y=-987654.), [*columns, "y"])


def test_catboost_fixed_budget_accepts_missing_best_iteration() -> None:
    rows = sample()
    params = {"task_type": "CPU", "iterations": 4, "verbose": False, "thread_count": 1,
              "random_seed": 0, "od_type": "Iter", "od_wait": 2}
    cfg = {"cat": ["apt"], "num": ["x"], "params": {**params, "loss_function": "RMSE"},
           "clf": {"family": "cb", "params": {**params, "loss_function": "Logloss"}}}
    p, count = fit_predict(cfg, rows, rows, iters=4)
    assert count == 4 and np.isfinite(p).all()
    p, count = fit_clf(cfg, rows, (rows.y > rows.y.median()).to_numpy(int), rows, None, None, 4)
    assert count == 4 and p.shape == (len(rows), 2)


def test_native_refit_restart_skips_saved_heads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import lanea

    rows = sample().assign(z_aobt=1200., y=lambda frame: 1200. + np.linspace(-1000., 1000., len(frame)))
    params = {"task_type": "CPU", "iterations": 4, "verbose": False, "thread_count": 1,
              "random_seed": 0, "od_type": "Iter", "od_wait": 2}
    cfg = {"cat": ["apt"], "num": ["x"], "reg": {"cat": ["apt"], "num": ["x"],
           "params": {**params, "loss_function": "RMSE"}},
           "clf": {"family": "cb", "params": {**params, "loss_function": "Logloss"}}}
    expected, _ = lanea.predict(cfg, rows, rows, iters=(4, 4, 4), model_dir=tmp_path)

    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("saved head was fitted again")

    monkeypatch.setattr(lanea, "fit_clf", reject)
    monkeypatch.setattr(lanea, "fit_predict", reject)
    actual, counts = lanea.predict(cfg, rows, rows, iters=(4, 4, 4), model_dir=tmp_path, resume=True)
    np.testing.assert_allclose(actual, expected)
    assert counts == (4, 4, 4) and not tuple(tmp_path.glob("*.tmp.model"))
    with pytest.raises(ValueError, match="budget changed"):
        lanea.predict(cfg, rows, rows, iters=(5, 4, 4), model_dir=tmp_path, resume=True)


@pytest.mark.parametrize("separator", ("/", "\\"))
def test_partial_recovery_requires_matching_scopes_and_model_hashes(tmp_path: Path, separator: str) -> None:
    directory = tmp_path / "selection"
    directory.mkdir()
    for head in ("gate", "tail", "body"):
        (directory / f"{head}.model").write_bytes(head.encode())
    hashes = {f"selection{separator}{path.name}": file_hash(path) for path in directory.iterdir()}
    original = {"config": {"loss": "RMSE"}, "fit_input_hash": "permitted-targets", "engine_hashes": {"train.py": "old"}}
    resumed = {**original, "engine_hashes": {"train.py": "repaired"}}
    write_record(tmp_path / "selection.json", {"specification": original, "iterations": (4, 4, 4)})
    with pytest.raises(ValueError, match="changed"):
        verified_selection(tmp_path, resumed)
    write_record(tmp_path / "recovery.json", {"original_specification": original,
                                            "resumed_engine_hashes": resumed["engine_hashes"], "artifact_hashes": hashes})
    assert verified_selection(tmp_path, resumed)["iterations"] == [4, 4, 4]
    with pytest.raises(ValueError, match="specifications"):
        verified_selection(tmp_path, {**resumed, "fit_input_hash": "different-targets"})
    (directory / "gate.model").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        verified_selection(tmp_path, resumed)


def test_completed_compatibility_is_explicit_and_keeps_scope_checks(tmp_path: Path) -> None:
    original = {"name": "ra_cb_wz_t7", "fit_input_hash": "same-permitted-targets", "engine_hashes": {"champion.py": "first-resume"}}
    resumed = {**original, "engine_hashes": {"champion.py": "windows-path-repair"}}
    rows = pd.DataFrame({"id": (1, 2), "p": (900., 1000.)})
    rows.to_parquet(tmp_path / "predictions.parquet")
    write_record(tmp_path / "complete.json", {"specification": original,
        "artifact_hashes": {"predictions.parquet": file_hash(tmp_path / "predictions.parquet")}})
    with pytest.raises(ValueError, match="changed"):
        load_completed(tmp_path, resumed, rows)
    write_record(tmp_path / "recovery.json", {"completed_engine_compatibility": {
        "execution_engine_hashes": original["engine_hashes"], "resumed_engine_hashes": resumed["engine_hashes"]}})
    np.testing.assert_array_equal(load_completed(tmp_path, resumed, rows)[0], rows.p)
    with pytest.raises(ValueError, match="changed"):
        load_completed(tmp_path, {**resumed, "fit_input_hash": "changed-targets"}, rows)


def test_donor_scopes_use_file_month_and_exclude_hidden_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import champion_neighbours as donor

    months = np.repeat((2, 3, 4, 5, 6, 8), 4)
    frame = pd.DataFrame({"id": np.arange(24), "file": [f"training_2025_{month:02d}.parquet" for month in months],
                          "month": months, "has_plan": 1, "y": np.arange(24, dtype=float), "z_aobt": 900.,
                          "apt": "TEST", "day": "2025-03-01", "runway": "R", "t_off": np.arange(24) * 60.})
    frame.loc[3, "month"] = 7
    allowed = pd.Index(frame.id.iloc[:20])
    calls = []

    def predict(name: str, fit: pd.DataFrame, query: pd.DataFrame, directory: Path) -> tuple:
        assert query.y.isna().all()
        assert fit.id.isin(allowed).all()
        assert not fit.id.isin(query.id).any()
        calls.append((tuple(fit.id), tuple(query.id)))
        return np.full(len(query), fit.y.mean() + 900.), {"runtime_seconds": 0., "artifact_hashes": {"predictions.parquet": "fake"},
                                                        "specification": {"config": {}}}

    monkeypatch.setattr(donor, "fit_member", predict)
    def raw(name: str, cfg: dict, query: pd.DataFrame, path: Path) -> np.ndarray:
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": query.id.values, "p": 900.}).to_parquet(path / "raw_predictions.parquet")
        return np.full(len(query), 900.)

    monkeypatch.setattr(donor, "raw_member_predictions", raw)
    before, _ = donor.with_champion_pw(frame, allowed, tmp_path / "before")
    changed = frame.copy()
    changed.loc[~changed.id.isin(allowed), "y"] = -987654.
    after, _ = donor.with_champion_pw(changed, allowed, tmp_path / "after")
    pd.testing.assert_frame_equal(before[donor.PW], after[donor.PW])
    assert any(3 in query for _, query in calls)
    assert donor.month_groups(frame, allowed) == ((2, 5), (3, 6), (4,))


def test_meta_weights_include_b_and_outer_labels_do_not_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import champion_pipeline as pipeline
    from scipy.optimize import nnls

    training = pd.DataFrame({"id": range(12), "has_plan": [1] * 9 + [0] * 3, "apt": "A", "y": np.arange(12) * 200.})
    for i, name in enumerate(pipeline.MEMBERS):
        training[name] = np.arange(12) * (150. + i * 20.) + 10.
        training.loc[training.has_plan == 0, name] = 1500.
    for stamp in pipeline.recal.Z:
        training[stamp] = 1000.
    outer = training.assign(id=lambda rows: rows.id + 100, y=-987654.)
    target = []
    original = pipeline.lgb.LGBMRegressor

    def regressor(**params: object) -> object:
        class Wrapped:
            def fit(self, data: pd.DataFrame, labels: pd.Series) -> object:
                target.extend(labels.tolist())
                self.model = original(**{**params, "n_jobs": 1, "n_estimators": 2}).fit(data, labels)
                self.booster_ = self.model.booster_
                return self

            def predict(self, data: pd.DataFrame) -> np.ndarray:
                return self.model.predict(data)
        return Wrapped()

    monkeypatch.setattr(pipeline.lgb, "LGBMRegressor", regressor)
    rows, record = pipeline.fit_meta(training, outer, tmp_path)
    expected, _ = nnls(training[list(pipeline.MEMBERS)].values, training.y.values)
    expected /= expected.sum()
    np.testing.assert_allclose(list(record["exact_weights"].values()), expected)
    assert record["meta_fit_rows"] == len(training)
    assert len(target) == 9 and min(target) > -987654.
    np.testing.assert_array_equal(rows.loc[rows.has_plan == 0, "clean_champion"], outer.loc[outer.has_plan == 0, pipeline.MEMBERS[0]])


def test_recording_context_excludes_adjacent_files() -> None:
    import laneb

    frame = pd.DataFrame({"id": (1, 2, 3), "apt": "LIRF", "file": ("February", "March", "March"),
                          "has_plan": (1, 1, 0), "day": "2025-02-28", "t_off": (10000., 11000., 10500.),
                          "z_aobt": (1000., 7000., np.nan), "z_sched": (1500., 8000., 9000.)})
    augmented = laneb.augment(frame)
    query = augmented[augmented.id == 3].iloc[0]
    assert query.a_za_p90 == query.w_za_p90 == 7000.
    changed = frame.copy()
    changed.loc[changed.file == "February", "z_aobt"] = 987654.
    other = laneb.augment(changed)
    pd.testing.assert_series_equal(augmented.loc[augmented.id == 3].iloc[0], other.loc[other.id == 3].iloc[0])
