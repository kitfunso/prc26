"""Arrival taxi-in level around each departure: the block-time recorder's offset, seen from the arrival side.
An offset d on the recorder adds d to arrivals' taxi-in and takes d off departures' taxi-out, and arrivals'
taxi-in is visible in both splits. `PYTHONPATH=C:/Users/skf_s/prc26 python analysis/arr_level_test.py [vis|cv]`"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data import CACHE, load_raw, rmse, secs
from features import WS_CLIP, WS_TAIL, WZ, arr_recs, build
from phys_tail_test import run_variant
from state_oracle import oracle_columns
from train import FOLDS, HOLDOUT, boot_delta

SCALES = {"wi": (("split", "apt"), 3600.0), "wi30": (("split", "apt"), 1800.0),
          "wi3h": (("split", "apt"), 10800.0), "wia": (("split", "apt", "airline"), 10800.0)}
NEW = ["wi_n", "wi_dev", "wi_dev_sd", "wi_hi", "wi30_dev", "wi3h_dev", "wi3h_dev_sd", "wia_n", "wia_dev"]


def arrivals() -> pd.DataFrame:
    """Arrivals that copy no planned or scheduled stamp, with taxi-in minus its (apt, runway, hour) median."""
    a = arr_recs(load_raw())
    print("arrival TAXITIME non-null by split:",
          a.groupby("split")["TAXITIME_SEC_mvt"].apply(lambda s: s.notna().mean()).round(4).to_dict(), flush=True)
    a = a[(a["a1"] == 0) & (a["sc"] == 0)].copy()
    a["tin"] = secs(a["BLOCK_TIME_UTC_mvt"], a["MVT_TIME_UTC_mvt"]).astype(float)
    a = a[np.isfinite(a["tin"])]
    norm = a.groupby([a["apt"], a["RUNWAY_mvt"].astype(str), a["MVT_TIME_UTC_mvt"].dt.hour])["tin"].transform("median")
    a["dev"] = np.clip(a["tin"] - norm, -WS_CLIP, WS_CLIP)
    print(f"physical arrivals {len(a)}, tin median {a['tin'].median():.0f}, dev sd {a['dev'].std():.0f}", flush=True)
    return a


def arr_level(f: pd.DataFrame, a: pd.DataFrame, keys: tuple[str, ...], win: float, pre: str) -> pd.DataFrame:
    """Arrivals within +-win of each departure's estimated off-block, same keys: count, mean, sd, share above 600."""
    t = (f["t_off"] - f["z_anchor"]).values
    n = np.zeros(len(f))
    s1, s2, sh = (np.full(len(f), np.nan) for _ in range(3))
    grp = {}
    for key, g in a.groupby(list(keys), observed=True):
        o = np.argsort(g["t_block"].values, kind="stable")
        d = g["dev"].values[o]
        grp[key] = (g["t_block"].values[o],) + tuple(np.concatenate([[0.0], np.cumsum(v)])
                                                     for v in (d, d ** 2, (d > WS_TAIL).astype(float)))
    for key, ix in f.groupby(list(keys), observed=True).indices.items():
        if key not in grp:
            continue
        tb, c1, c2, ch = grp[key]
        lo, hi = np.searchsorted(tb, t[ix] - win, "left"), np.searchsorted(tb, t[ix] + win, "right")
        n[ix] = hi - lo
        s1[ix], s2[ix], sh[ix] = c1[hi] - c1[lo], c2[hi] - c2[lo], ch[hi] - ch[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
        return pd.DataFrame({f"{pre}_n": n, f"{pre}_dev": m, f"{pre}_dev_sd": np.sqrt(np.maximum(s2 / n - m ** 2, 0)),
                             f"{pre}_hi": sh / n}, index=f.index)


def shift(f: pd.DataFrame) -> None:
    """Lane A, standardised mean difference of each new column, rank month vs the same 2025 month."""
    a = f[f["has_plan"] == 1]
    for m in (1, 7):
        r, h = a[(a["split"] == "rank") & (a["month"] == m)][NEW], a[(a["split"] == "train") & (a["month"] == m)][NEW]
        smd = (r.mean() - h.mean()) / np.sqrt((r.var() + h.var()) / 2)
        print(f"month {m} SMD rank vs 2025: " + ", ".join(f"{k} {v:+.2f}" for k, v in smd.items()), flush=True)


def visibility(a: pd.DataFrame) -> None:
    """Linear R^2 of the oracle window state on WZ, then on WZ plus the arrival level, over the same rows."""
    orc = oracle_columns(a)[0]
    for col in ("win_tail_shr", "win_physc_mean"):
        y = orc[col].values
        ok = np.isfinite(y) & np.isfinite(a[WZ + NEW].values).all(1)
        for cols in (WZ, WZ + NEW):
            X = np.column_stack([np.ones(ok.sum()), a[cols].values[ok]])
            beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
            print(f"{col}: {len(cols)} cols R2 {1 - np.var(y[ok] - X @ beta) / np.var(y[ok]):.4f} on {ok.sum()} rows",
                  flush=True)
        print(f"{col} corr: " + ", ".join(f"{k} {v:+.3f}" for k, v in a[NEW].corrwith(orc[col]).items()), flush=True)


def cv(a: pd.DataFrame) -> None:
    """ra_lgb_wz plus the arrival level through the analysis harness, against the cached wz2 run (= ra_lgb_wz)."""
    ta, ha = a[~a["month"].isin(HOLDOUT)].copy(), a[a["month"].isin(HOLDOUT)].copy()
    base_oof = pd.read_parquet(CACHE / "analysis_window_state_wz2_oof.parquet").set_index("id")["oof"].reindex(ta["id"].values).values
    base_ho = pd.read_parquet(CACHE / "analysis_window_state_wz2_holdout.parquet").set_index("id")["p"].reindex(ha["id"].values).values
    oof, ph = run_variant("wi", WZ + NEW, ta, ha)
    pd.DataFrame({"id": ta["id"].values, "oof": oof}).to_parquet(CACHE / "analysis_arr_level_wi_oof.parquet")
    pd.DataFrame({"id": ha["id"].values, "p": ph}).to_parquet(CACHE / "analysis_arr_level_wi_holdout.parquet")
    y, day = ha["y"].values, ha["day"].values
    better = sum(rmse(ta["y"].values[va], oof[va]) < rmse(ta["y"].values[va], base_oof[va])
                 for va in (ta["month"].isin(m).values for m in FOLDS))
    point, lo, hi = boot_delta(y, base_ho, ph, day)
    print(f"[wi] holdout {rmse(y, ph):.3f} vs wz2 {rmse(y, base_ho):.3f}: {point:+.3f} [{lo:+.3f}, {hi:+.3f}], "
          f"folds better {better}/5", flush=True)


def main(mode: str) -> None:
    f, arr = build(), arrivals()
    new = pd.concat([arr_level(f, arr, k, w, p) for p, (k, w) in SCALES.items()], axis=1)[NEW]
    f = pd.concat([f, new], axis=1)
    print(new.describe().T.round(2).to_string(), flush=True)
    shift(f)
    a = f[(f["split"] == "train") & (f["has_plan"] == 1)]
    visibility(a)
    if mode == "cv":
        cv(a)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vis")
