"""Test whether label-free day/hour dispersion-of-gap features (docs/PHYS_TAIL.md, motivated by the
phys-tail's 18.4x day-level overdispersion) help ra_lgb_arr on lane A. Harness check first: unmodified
config, fold 0 only, must reproduce data/cache/ra_lgb_arr_oof.parquet exactly (CPU LightGBM determinism).
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/phys_tail_test.py`"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import build
from lanea import CONFIGS, predict
from phys_tail import dayvar_features
from train import FOLDS, HOLDOUT, boot_delta

VARIANTS = {
    "hour_disp": ["h_dae_std", "h_dal_std"],
    "full_disp": ["day_dae_std", "h_dae_std", "day_dal_std", "h_dal_std"],
}


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    f = build()
    f = f[f["split"] == "train"]
    f = pd.concat([f, dayvar_features(f)], axis=1)
    tr, ho = f[~f["month"].isin(HOLDOUT)], f[f["month"].isin(HOLDOUT)]
    return tr[tr["has_plan"] == 1].copy(), ho[ho["has_plan"] == 1].copy()


def harness_check(ta: pd.DataFrame) -> None:
    """Fold 0, unmodified ra_lgb_arr config, must reproduce the cached OOF to 1e-6 (CPU LightGBM determinism)."""
    va = ta["month"].isin(FOLDS[0]).values
    p, _ = predict(CONFIGS["ra_lgb_arr"], ta[~va], ta[va], ta[va])
    cached = pd.read_parquet(CACHE / "ra_lgb_arr_oof.parquet").set_index("id")["oof"]
    ref = cached.reindex(ta["id"].values[va]).values
    diff = np.abs(p - ref).max()
    print(f"harness check: max abs diff={diff:.2e}, allclose(1e-6)={np.allclose(p, ref, atol=1e-6)}", flush=True)
    if not np.allclose(p, ref, atol=1e-6):
        raise SystemExit("harness check FAILED, stopping before any variant runs")


def run_variant(name: str, extra: list[str], ta: pd.DataFrame, ha: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cfg = copy.deepcopy(CONFIGS["ra_lgb_arr"])
    cfg["num"] = cfg["num"] + extra
    cfg["reg"]["num"] = cfg["reg"]["num"] + extra
    oof, its, t0 = np.full(len(ta), np.nan), [], time.time()
    for i, months in enumerate(FOLDS):
        va = ta["month"].isin(months).values
        p, it = predict(cfg, ta[~va], ta[va], ta[va])
        oof[va] = p
        its.append(it)
        print(f"[{name}] fold {i} {months}: {rmse(ta['y'].values[va], p):.2f} at {it} iters, {time.time()-t0:.0f}s", flush=True)
    it = tuple(max(int(np.mean([x[j] for x in its]) * 1.1), 0) for j in range(3))
    ph = np.maximum(predict(cfg, ta, ha, iters=it)[0], 0)
    print(f"[{name}] holdout refit at {it} iters, {time.time()-t0:.0f}s", flush=True)
    return oof, ph


def main() -> None:
    ta, ha = _data()
    harness_check(ta)

    base_oof = pd.read_parquet(CACHE / "ra_lgb_arr_oof.parquet").set_index("id")["oof"].reindex(ta["id"].values).values
    base_ho = pd.read_parquet(CACHE / "ra_lgb_arr_holdout.parquet").set_index("id")["p"].reindex(ha["id"].values).values

    verdicts = {}
    for name, extra in VARIANTS.items():
        oof, ph = run_variant(name, extra, ta, ha)
        pd.DataFrame({"id": ta["id"].values, "oof": oof}).to_parquet(CACHE / f"analysis_phys_tail_{name}_oof.parquet")
        pd.DataFrame({"id": ha["id"].values, "p": ph}).to_parquet(CACHE / f"analysis_phys_tail_{name}_holdout.parquet")

        print(f"\n=== {name} vs ra_lgb_arr ===")
        fold_pass = True
        for i, months in enumerate(FOLDS):
            va = ta["month"].isin(months).values
            r_base, r_new = rmse(ta["y"].values[va], base_oof[va]), rmse(ta["y"].values[va], oof[va])
            fold_pass = fold_pass and r_new < r_base
            print(f"fold {i} {months}: base={r_base:.3f} new={r_new:.3f} {'better' if r_new < r_base else 'WORSE'}")
        r_base_ho, r_new_ho = rmse(ha["y"].values, base_ho), rmse(ha["y"].values, ph)
        ho_pass = r_new_ho < r_base_ho
        print(f"holdout: base={r_base_ho:.3f} new={r_new_ho:.3f} {'better' if ho_pass else 'WORSE'}")
        point, lo, hi = boot_delta(ha["y"].values, base_ho, ph, ha["day"].values)
        print(f"boot_delta holdout (base - new, positive=new better): {point:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        passed = fold_pass and ho_pass and lo > 0
        verdicts[name] = passed
        print(f"Verdict [{name}]: {'PASS' if passed else 'FAIL'}")

    print("\n" + " | ".join(f"{n}: {'PASS' if v else 'FAIL'}" for n, v in verdicts.items()))
    print("DONE")


if __name__ == "__main__":
    main()
