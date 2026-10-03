"""Real label-mutation closure for the complete bounded Lane B predictors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laneb
from campaign import outer_masks
from data import DATA
from features import CAT
from mechanisms import candidates


def main() -> None:
    out = DATA / "runs" / "contender-20261003"
    f = pd.read_parquet(out / "cold_features.parquet")
    for c in CAT:
        f[c] = f[c].astype("category")
    fit, eva = outer_masks(f, (2, 3))
    f = laneb.augment(f)
    ids = pd.Index(f.loc[eva & (f.has_plan == 0), "id"])
    seen = []
    originals = laneb.fit_clf, laneb.fit_tau
    def guard(fn):
        def call(*args, **kwargs):
            tr, va = args[0], args[4 if fn.__name__ == "fit_clf" else 3]
            assert not tr.id.isin(f.loc[~fit, "id"]).any()
            if va is not None:
                assert not va.id.isin(f.loc[~fit, "id"]).any()
            seen.append({"component": fn.__name__, "fit_rows": len(tr), "stopping_rows": 0 if va is None else len(va)})
            return fn(*args, **kwargs)
        return call
    laneb.fit_clf, laneb.fit_tau = (guard(fn) for fn in originals)
    tr, te, p, iterations = laneb.outer_predict(f, fit, ids)
    p = laneb.hull(te, laneb.hull_prior(tr, te, p))
    first, _ = candidates(tr, te.assign(y=np.nan), p)
    f.loc[~fit, "y"] = -987654.
    tr2, te2, p2, it2 = laneb.outer_predict(f, fit, ids)
    p2 = laneb.hull(te2, laneb.hull_prior(tr2, te2, p2))
    second, _ = candidates(tr2, te2.assign(y=np.nan), p2)
    np.testing.assert_array_equal(p, p2)
    assert iterations == it2
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    old = pd.read_parquet(out / "screen-2-3.parquet").set_index("id")
    np.testing.assert_array_equal(p, te.id.map(old.control).values)
    record = {"whole_excluded_label_mutation": "PASS", "queries_with_missing_targets": "PASS",
              "real_stopping_call_guard": "PASS", "control_prediction_reproduction": "PASS",
              "mutated_rows": int((~fit).sum()), "evaluation_laneB_rows": len(te), "iterations": iterations, "calls": seen}
    (out / "scope_checks.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
