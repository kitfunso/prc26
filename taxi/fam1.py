"""FAM1: the four empty rows of the model-class register, one arm each, lane A.

`docs/COVERAGE.md` section 4 answered "what model" as prose indexed by lane id, so a family nobody tried produced no row and
the section read as complete; ridge went missing for 60 ledger rows that way. The register added there on 2026-09-08 is the
structural fix and it left four rows empty: bagged trees, randomised trees, k-nearest neighbours and kernel methods. This lane
empties them with measurements. It is a screen, not a champion hunt: the arms are single fixed cells and the gate is the blend.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder

from taxi.bf import best_blend
from taxi.champ import COLS_D2, lane_a
from taxi.l2 import floored
from taxi.nn1 import base_oof, blend_weight
from taxi.rc1 import CATS, DROP, Enc
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

FOLDS = ((2, 3), (6, 8), (11, 12))
BUDGET_S, GATE = 8 * 3600, 1.0
KNN_SUB, KNN_K, LANDMARKS, COMPS, CHUNK = 200_000, 50, 10_000, 500, 100_000
NUMS = [c for c in COLS_D2 if c not in CATS and c not in DROP]
CATCOLS = [c for c in COLS_D2 if c in CATS]


def _dense(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return d[cols].to_numpy(dtype="float64", na_value=np.nan)


def _num_block(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Median-imputed, train-standardised numerics. The only encoder a neighbour rule can carry: it cannot reweight columns."""
    a, b = _dense(tr, NUMS), _dense(te, NUMS)
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    a, b = np.where(np.isnan(a), med, a), np.where(np.isnan(b), med, b)
    sd = a.std(axis=0)
    sd[sd == 0] = 1.0
    mu = a.mean(axis=0)
    return ((a - mu) / sd).astype(np.float32), ((b - mu) / sd).astype(np.float32)


def _tree_block(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ordinal codes beside the numerics. scikit-learn's forests have no native categorical support, so the order is false."""
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(tr[CATCOLS].astype(str))
    a, b = _num_block(tr, te)
    return (np.hstack([a, oe.transform(tr[CATCOLS].astype(str)).astype(np.float32)]),
            np.hstack([b, oe.transform(te[CATCOLS].astype(str)).astype(np.float32)]))


def _forest(kind: type, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    a, b = _tree_block(tr, te)
    m = kind(n_estimators=60, max_features="sqrt", min_samples_leaf=50, n_jobs=12, random_state=0).fit(a, tr.c.values)
    del a
    gc.collect()
    return m.predict(b)


def arm_rf(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return _forest(RandomForestRegressor, tr, te)


def arm_et(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return _forest(ExtraTreesRegressor, tr, te)


def arm_knn(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """A seeded subsample is not a shortcut here: 1.38 M by 345 k brute-force pairs do not fit the lane's whole budget."""
    a, b = _num_block(tr, te)
    i = np.random.default_rng(0).choice(len(a), min(KNN_SUB, len(a)), replace=False)
    m = KNeighborsRegressor(n_neighbors=KNN_K, weights="distance", algorithm="brute", n_jobs=12)
    m.fit(a[i], tr.c.values[i])
    del a
    gc.collect()
    return np.concatenate([m.predict(b[j:j + CHUNK]) for j in range(0, len(b), CHUNK)])


def _gamma(x: sparse.csr_matrix) -> float:
    """Median heuristic. The scikit-learn default of 1/n_features is near-zero at this width and flattens the kernel."""
    s = x[np.random.default_rng(0).choice(x.shape[0], 2000, replace=False)].toarray()
    d = np.linalg.norm(s[:1000] - s[1000:], axis=1)
    return float(1.0 / (2.0 * np.median(d[d > 0]) ** 2))


def _project(ny: Nystroem, x: sparse.csr_matrix) -> np.ndarray:
    out = np.empty((x.shape[0], COMPS), dtype=np.float32)
    for j in range(0, x.shape[0], CHUNK):
        out[j:j + CHUNK] = ny.transform(x[j:j + CHUNK]).astype(np.float32)
    return out


def arm_nys(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """An RBF support-vector machine made linear-time: Nystroem draws the kernel's landmarks, ridge fits in that space."""
    enc = Enc(tr, COLS_D2, 500, 0, 3, None)
    a = enc(tr)
    ny = Nystroem(gamma=_gamma(a), n_components=COMPS, random_state=0)
    ny.fit(a[np.random.default_rng(1).choice(a.shape[0], LANDMARKS, replace=False)])
    za = _project(ny, a)
    del a
    gc.collect()
    m = Ridge(alpha=1.0, solver="lsqr").fit(za, tr.c.values)
    del za
    gc.collect()
    return m.predict(_project(ny, enc(te)))


ARMS = (("nys", "kernel (RBF via Nystroem, then ridge)", arm_nys), ("et", "randomised trees", arm_et),
        ("rf", "bagged trees (random forest)", arm_rf), ("knn", "k-nearest neighbours", arm_knn))


def oof(tra: pd.DataFrame, fn, budget: float, folds: tuple = FOLDS) -> tuple[np.ndarray, float, str]:
    """Out-of-fold `c` over the declared folds. The TIME trigger is measured on fold 1 and never on a result."""
    p, spent, note = np.full(len(tra), np.nan), 0.0, ""
    for n, fold in enumerate(folds):
        if n and spent / n * (len(folds) - n) > budget - spent:
            note = (f"stopped after {n} of {len(folds)} folds: {spent / 60:.0f} min spent, "
                    f"{spent / n * (len(folds) - n) / 60:.0f} min needed against {(budget - spent) / 60:.0f} min left")
            break
        m = tra.month.isin(fold).values
        t0 = time.time()
        p[m] = fn(tra[~m], tra[m])
        spent += time.time() - t0
    return p, spent, note


def cache_path(tag: str) -> str:
    return f"data/cache/fam1_{tag}_oof.parquet"


def score_arm(tra: pd.DataFrame, base: np.ndarray, tag: str, label: str, p: np.ndarray, spent: float, note: str) -> dict:
    """One arm's row, split out so the report rebuilds from the saved vectors instead of refitting the whole screen."""
    ok = np.isfinite(p) & np.isfinite(base)
    y = tra.y.values[ok]
    lvl = floored(p[ok] + tra.z_aobt.values[ok])
    b = floored(base[ok])  # the blend is floored, so an unfloored base makes the gate read the C1 floor and not the arm
    w = blend_weight(lvl, b, y)
    d = lvl - b
    r = dict(tag=tag, label=label, min=spent / 60, note=note, n=int(ok.sum()), w=w,
             raw=float(np.dot(d, y - b) / np.dot(d, d)),  # unclipped, so a zero weight is measured not clipped
             level=rmse(lvl - y), base=rmse(b - y), agree=best_blend(b - y, lvl - y)[2],
             blend=rmse(floored(w * lvl + (1 - w) * b) - y))
    print(f"| {tag} | {r['level']:.2f} | {r['blend']:.2f} | {r['base']:.2f} | {r['agree']:.4f} | {r['min']:.0f} |", flush=True)
    return r


def screen(tra: pd.DataFrame, base: np.ndarray, budget: float) -> list[dict]:
    rows = []
    for tag, label, fn in ARMS:
        left = budget - sum(r["min"] for r in rows) * 60
        if left <= 0:
            rows.append(dict(tag=tag, label=label, min=0.0, note="not started: the lane's budget was spent by earlier arms"))
            continue
        try:
            p, spent, note = oof(tra, fn, left)
        except MemoryError:
            rows.append(dict(tag=tag, label=label, min=0.0, note="MemoryError: this cell does not fit the free RAM on this box"))
            continue
        rows.append(score_arm(tra, base, tag, label, p, spent, note))
        pd.DataFrame({"id": tra.id.values, "pred": p}).to_parquet(cache_path(tag), index=False)
    return rows


def cached(tra: pd.DataFrame, base: np.ndarray, _budget: float) -> list[dict]:
    """Rescore the saved out-of-fold vectors. The arms are deterministic, so this reproduces the run without refitting it."""
    rows = []
    for tag, label, _fn in ARMS:
        if not os.path.exists(cache_path(tag)):
            rows.append(dict(tag=tag, label=label, min=0.0, note=f"no saved vector at {cache_path(tag)}"))
            continue
        c = pd.read_parquet(cache_path(tag))
        if not np.array_equal(c.id.values, tra.id.values):
            rows.append(dict(tag=tag, label=label, min=0.0, note=f"{cache_path(tag)} was written for a different row set"))
            continue
        rows.append(score_arm(tra, base, tag, label, c.pred.values, 0.0, ""))
    return rows


def main(out: str = "docs/FAM1.md", from_cache: bool = False) -> None:
    tra, tea = lane_a()
    t0 = time.time()
    print("| arm | own RMSE | blend | MC2 base | agreement | min |\n|---|---|---|---|---|---|", flush=True)
    rows = (cached if from_cache else screen)(tra, base_oof(tra), BUDGET_S)

    passed = [r for r in rows if r.get("blend") is not None and r["base"] - r["blend"] >= GATE]
    tbl, verd = [], []
    for r in rows:
        if "blend" not in r:
            tbl.append(f"| FAM1-{r['tag']} | {r['label']} | NOT RUN | | | | {r['note']} |")
            continue
        gap = r["base"] - r["blend"]
        tbl.append(f"| FAM1-{r['tag']} | {r['label']} | {r['level']:.2f} | {r['blend']:.2f} | {gap:+.2f} | {r['raw']:+.4f} | "
                   f"{r['agree']:.4f} | {'PASS' if gap >= GATE else 'closed at this cell'}"
                   f"{'; ' + r['note'] if r['note'] else ''} |")

    y, sb3 = tea.y.values, sb3_predict(tra, tea)
    e0 = sb3 - y
    for r in passed:
        fn = dict((t, f) for t, _, f in ARMS)[r["tag"]]
        p = floored(fn(tra, tea) + tea.z_aobt.values)
        v = paired_delta(e0, floored(r["w"] * p + (1 - r["w"]) * sb3) - y, tea.day.values)
        verd.append(f"- **FAM1-{r['tag']} + SB3** at w={r['w']:.4f}, holdout {rmse(p - y):.2f} s alone: "
                    + verdict_line(f"FAM1-{r['tag']}", "SB3", v, k=len(ARMS)))

    text = "\n".join([
        "# FAM1: the four empty model-class rows, lane A screen", "",
        f"Regenerate with `python -m taxi.fam1`. Declared in protocol section 4 before the code was written, and declared as "
        f"COVERAGE rather than as a champion hunt. Four arms, one fixed cell each, NO search inside any arm, k = {len(ARMS)}. "
        f"Folds are the three declared month pairs {FOLDS} that O2 and NN3 use, so the comparator needs no new fit: MC2's "
        + (f"cached out-of-fold vector on the same rows. Rescored from the saved arm vectors, so each arm's wall clock reads 0."
         if from_cache else
         f"cached out-of-fold vector on the same rows. Total {(time.time() - t0) / 3600:.1f} h of a {BUDGET_S / 3600:.0f} h "
         f"budget."), "",
        f"These four families were the empty rows of the model-class register in `docs/COVERAGE.md` section 4. That register "
        f"exists because this campaign lost a whole algorithm family for 60 ledger rows: section 4 answered \"what model\" as "
        f"prose indexed by lane id, so a class nobody tried produced no row at all and the section read as complete. Each arm "
        f"below now closes its row with a number instead of an argument.", "",
        "## The screen", "",
        f"Gate declared in advance: an arm meets the untouched holdout only if its blend with MC2 beats MC2 alone by "
        f"{GATE:.1f} s on these rows. An arm that fails is a closed register row and costs no holdout fit.", "",
        "| arm | family | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |",
        "|---|---|---|---|---|---|---|---|",
        *tbl, "",
        "## Verdicts against SB3" if verd else "## No arm passed the gate", "",
        *(verd or [f"Every arm failed the declared {GATE:.1f} s gate, so none met the holdout. That is the lane's result and it "
                   f"was the pre-registered prediction for three of the four. The four register rows are now closed by "
                   f"measurement rather than by argument, which is what this lane was declared to do."]), ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
