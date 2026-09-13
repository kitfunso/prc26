"""Ceiling on lane A gains from knowing the airport-day recorder state (docs/STATE_ORACLE.md).
Oracle columns (day/window class shares, tail shares, physical-residual means) are leave-one-out
group stats built straight from mix.label() on lane A train rows; they live only in memory here,
never written to data/cache/feat.parquet and never added to features.py/lanea.py. Same predict()
loop as lanea.cv, on ra_lgb_arr, lane A only: no donor merge, no docs/RESULTS.md write.
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/state_oracle.py`"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import build
from lanea import CONFIGS as LANEA_CONFIGS, predict
from mix import label
from train import FOLDS, HOLDOUT, boot_delta

WIN = 3600.0       # +-1h window group
BIG_EOBT = 300.0   # class-1 rows with |z_eobt - z_aobt| over this are a clean eobt copy, not a near-aobt tie
TAIL = 600.0       # matches lanea.TAIL: the tail classifier's own |c| threshold
CLIP = 1800.0
OUT = CACHE.parent / "analysis_state_oracle_summary.csv"


def _indicators(a: pd.DataFrame) -> dict[str, np.ndarray]:
    cls = label(a)
    c = (a["y"] - a["z_aobt"]).values
    big = (a["z_eobt"] - a["z_aobt"]).abs().values > BIG_EOBT
    is_phys = (cls == 3).astype(float)
    return {
        "cls0": (cls == 0).astype(float), "cls1": (cls == 1).astype(float),
        "cls1big": ((cls == 1) & big).astype(float), "cls2": (cls == 2).astype(float),
        "tail": (np.abs(c) > TAIL).astype(float),
        "physval": np.clip(c, -CLIP, CLIP) * is_phys, "physcnt": is_phys,
    }


def _loo_day_sums(a: pd.DataFrame, ind: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Leave-one-out (apt, day) sums: each row's group total minus its own value."""
    keys = [a["apt"], a["day"]]
    out = {name: pd.Series(v, index=a.index).groupby(keys, observed=True).transform("sum").values - v
           for name, v in ind.items()}
    n = a.groupby(["apt", "day"], observed=True)["id"].transform("size").values.astype(float)
    out["_n"] = n - 1
    return out


def _loo_window_sums(a: pd.DataFrame, ind: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Leave-one-out +-WIN-second sums within (apt, day): sort t_off, cumsum, searchsorted (features.py's own pattern)."""
    names = list(ind)
    n = len(a)
    out = {name: np.zeros(n) for name in names}
    cnt = np.zeros(n)
    t = a["t_off"].values
    for _, ix in a.groupby(["apt", "day"], observed=True).indices.items():
        ix = np.asarray(ix)
        order = np.argsort(t[ix])
        ts = t[ix][order]
        lo = np.searchsorted(ts, t[ix] - WIN, "left")
        hi = np.searchsorted(ts, t[ix] + WIN, "right")
        cnt[ix] = (hi - lo) - 1
        for name in names:
            cum = np.concatenate([[0.0], np.cumsum(ind[name][ix][order])])
            out[name][ix] = (cum[hi] - cum[lo]) - ind[name][ix]
    out["_n"] = cnt
    return out


def _cls_frame(sums: dict[str, np.ndarray], prefix: str, index: pd.Index) -> pd.DataFrame:
    n = sums["_n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = {f"{prefix}_cls0_shr": sums["cls0"] / n, f"{prefix}_cls1_shr": sums["cls1"] / n,
             f"{prefix}_cls1big_shr": sums["cls1big"] / n, f"{prefix}_cls2_shr": sums["cls2"] / n, f"{prefix}_n": n}
    return pd.DataFrame(d, index=index)


def _tail_frame(sums: dict[str, np.ndarray], prefix: str, index: pd.Index) -> pd.DataFrame:
    n = sums["_n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = {f"{prefix}_tail_shr": sums["tail"] / n, f"{prefix}_physc_mean": sums["physval"] / sums["physcnt"]}
    return pd.DataFrame(d, index=index)


def oracle_columns(a: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """All 14 oracle columns for lane A train rows `a`, plus the four named column groups."""
    ind = _indicators(a)
    day_s, win_s = _loo_day_sums(a, ind), _loo_window_sums(a, ind)
    day_cls, win_cls = _cls_frame(day_s, "day", a.index), _cls_frame(win_s, "win", a.index)
    day_tail, win_tail = _tail_frame(day_s, "day", a.index), _tail_frame(win_s, "win", a.index)
    out = pd.concat([day_cls, win_cls, day_tail, win_tail], axis=1)
    groups = {"day_cls": list(day_cls.columns), "win_cls": list(win_cls.columns),
              "day_tail": list(day_tail.columns), "win_tail": list(win_tail.columns)}
    return out, groups


def variant_cfg(extra: list[str]) -> dict:
    cfg = copy.deepcopy(LANEA_CONFIGS["ra_lgb_arr"])
    cfg["num"] = cfg["num"] + extra
    cfg["reg"]["num"] = cfg["reg"]["num"] + extra
    return cfg


def harness_check(ta: pd.DataFrame, ha: pd.DataFrame, oof: np.ndarray, ph: np.ndarray) -> None:
    """A0 must reproduce the cached ra_lgb_arr oof/holdout on lane A rows bit for bit (CPU LightGBM is deterministic)."""
    ref_oof = pd.read_parquet(CACHE / "ra_lgb_arr_oof.parquet").rename(columns={"oof": "ref"})
    ref_ho = pd.read_parquet(CACHE / "ra_lgb_arr_holdout.parquet").rename(columns={"p": "ref"})
    mo = pd.DataFrame({"id": ta["id"].values, "p": oof}).merge(ref_oof, on="id", how="left")
    mh = pd.DataFrame({"id": ha["id"].values, "p": ph}).merge(ref_ho, on="id", how="left")
    assert mo["ref"].notna().all() and mh["ref"].notna().all(), "id mismatch vs cached ra_lgb_arr oof/holdout"
    d_oof, d_ho = float((mo["p"] - mo["ref"]).abs().max()), float((mh["p"] - mh["ref"]).abs().max())
    r = rmse(ha["y"].values, ph)
    ok = d_oof < 1e-6 and d_ho < 1e-6
    print(f"HARNESS oof_max_abs_diff={d_oof:.3e} holdout_max_abs_diff={d_ho:.3e} holdout_laneA_rmse={r:.2f} "
          f"(ref 231.64) -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(f"HARNESS FAIL: oof_diff={d_oof:.3e} holdout_diff={d_ho:.3e} rmse={r:.2f}")


def run_variant(name: str, extra: list[str], ta: pd.DataFrame, ha: pd.DataFrame) -> dict:
    cfg = variant_cfg(extra)
    oof, its, fold_rmse, t0 = np.full(len(ta), np.nan), [], [], time.time()
    for i, months in enumerate(FOLDS):
        va = ta["month"].isin(months).values
        p, it = predict(cfg, ta[~va], ta[va], ta[va])
        oof[va] = p; its.append(it)
        r = rmse(ta["y"].values[va], p)
        fold_rmse.append(r)
        print(f"{name} fold {i} {months}: {r:.2f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    it = tuple(int(np.mean([x[j] for x in its]) * 1.1) for j in range(3))
    ph = np.maximum(predict(cfg, ta, ha, iters=it)[0], 0)
    ho_rmse = rmse(ha["y"].values, ph)
    print(f"{name} holdout: {ho_rmse:.2f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"id": ta["id"].values, "p": oof}).to_parquet(CACHE / f"analysis_state_oracle_{name}_oof.parquet")
    pd.DataFrame({"id": ha["id"].values, "p": ph}).to_parquet(CACHE / f"analysis_state_oracle_{name}_holdout.parquet")
    if name == "A0":
        harness_check(ta, ha, oof, ph)
    return dict(name=name, oof=oof, ph=ph, cv_rmse=rmse(ta["y"].values, oof), fold_rmse=fold_rmse,
                ho_rmse=ho_rmse, iters=it)


def main() -> None:
    f = build()
    a = f[(f["split"] == "train") & (f["has_plan"] == 1)].copy()
    extra, groups = oracle_columns(a)
    a = pd.concat([a, extra], axis=1)
    ta, ha = a[~a["month"].isin(HOLDOUT)], a[a["month"].isin(HOLDOUT)]
    print(f"lane A rows: {len(ta)} non-holdout, {len(ha)} holdout", flush=True)

    variant_cols = {
        "A0": [], "A1": groups["day_cls"], "A2": groups["win_cls"], "A3": groups["day_tail"],
        "A4": groups["win_tail"],
        "A5": groups["day_cls"] + groups["win_cls"] + groups["day_tail"] + groups["win_tail"],
    }
    results = {name: run_variant(name, cols, ta, ha) for name, cols in variant_cols.items()}

    y_ho, day_ho, p0 = ha["y"].values, ha["day"].values, results["A0"]["ph"]
    rows = []
    for name, r in results.items():
        delta, lo, hi = boot_delta(y_ho, p0, r["ph"], day_ho)
        rows.append({"variant": name, **{f"fold{i}": v for i, v in enumerate(r["fold_rmse"])},
                     "cv_laneA": r["cv_rmse"], "holdout_laneA": r["ho_rmse"], "iters": str(r["iters"]),
                     "delta_vs_A0": delta, "delta_lo": lo, "delta_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\n", out.round(3).to_string(index=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
