"""RC1: ridge regression on lane A with its parameters searched by Optuna, the campaign's first linear model.

Every lane-A model here is a boosted tree or a neural net, so Stage 3's linear baseline has never existed on this lane and
Stage 4's "a richer class earns its place only by beating the simple one" has never been answerable. A penalised linear fit is
also the most distant family from a GBDT this campaign can reach, and `docs/BF.md` prices a blend partner by level AND
agreement, so the objective below is the blend rather than ridge's own level.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from taxi.bf import best_blend
from taxi.champ import COLS_D1, COLS_D2, lane_a
from taxi.data import CATEGORICAL, CV_FOLDS, PLAN_CAT
from taxi.l2 import floored
from taxi.nn1 import base_oof, blend_weight
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

TRIALS, BUDGET_S, CHUNK = 40, 6 * 3600, 200_000
STORE, CACHE = "sqlite:///data/cache/rc1.db", "data/cache/rc1_holdout.parquet"
CATS = (*CATEGORICAL, *PLAN_CAT, "apt")  # apt is an object column sitting in the numeric half of COLS_D2
DROP = ("has_fpl",)  # constant True on lane A by the lane's own definition, so a zero-variance column
SPLINE = ("z_aobt", "z_anchor", "z_sched", "queue", "dur", "d_ae", "hour", "dep_before", "arr_anchor", "t_occ")


def _num(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return d[cols].to_numpy(dtype="float64", na_value=np.nan)


class Enc:
    """One sparse design matrix. Every statistic here is fitted on the training rows alone and applied unchanged elsewhere."""

    def __init__(self, tr: pd.DataFrame, cols: list[str], min_freq: int, knots: int, degree: int, wins: float | None):
        self.cats = [c for c in cols if c in CATS]
        self.nums = [c for c in cols if c not in CATS and c not in DROP]
        self.oh = OneHotEncoder(min_frequency=min_freq, handle_unknown="infrequent_if_exist", sparse_output=True,
                                dtype=np.float32).fit(tr[self.cats].astype(str))
        x = _num(tr, self.nums)
        self.miss = np.isnan(x).any(axis=0)
        med = np.nanmedian(x, axis=0)
        self.med = np.where(np.isfinite(med), med, 0.0)  # an all-NaN column would otherwise impute NaN into every row
        self.lo, self.hi = np.nanquantile(x, [wins, 1 - wins], axis=0) if wins else (None, None)
        self.six = np.array([i for i, c in enumerate(self.nums) if c in SPLINE], dtype=int)
        self.spl = (SplineTransformer(n_knots=knots, degree=degree, knots="quantile", include_bias=False)
                    .fit(self._clean(x)[:, self.six]) if knots else None)
        sc = StandardScaler()
        for i in range(0, len(tr), CHUNK):
            sc.partial_fit(self._blocks(x[i:i + CHUNK]))  # chunked: the full dense block is several GB at 1.7M rows
        self.sc = sc

    def _clean(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(x), self.med, np.clip(x, self.lo, self.hi) if self.lo is not None else x)

    def _blocks(self, x: np.ndarray) -> np.ndarray:
        z = self._clean(x)
        parts = [z, np.isnan(x[:, self.miss]).astype(np.float64)]
        if self.spl is not None:
            parts.append(self.spl.transform(z[:, self.six]))
        return np.hstack(parts)

    def __call__(self, d: pd.DataFrame) -> sparse.csr_matrix:
        out = []
        for i in range(0, len(d), CHUNK):
            part = d.iloc[i:i + CHUNK]
            dense = self.sc.transform(self._blocks(_num(part, self.nums))).astype(np.float32)
            out.append(sparse.hstack([sparse.csr_matrix(dense), self.oh.transform(part[self.cats].astype(str))], format="csr"))
        return sparse.vstack(out, format="csr", dtype=np.float32)


def suggest(t: optuna.Trial) -> dict:
    """Six knobs: the ridge penalty, the one-hot cutoff, the spline basis and its shape, winsorising, and the column set."""
    knots = t.suggest_categorical("knots", [0, 4, 6, 8])
    wins = t.suggest_categorical("wins", [0.0, 0.001, 0.005, 0.02])
    return dict(alpha=t.suggest_float("alpha", 1e-3, 1e5, log=True),
                min_freq=t.suggest_categorical("min_freq", [1, 20, 100, 500]),
                knots=knots, degree=t.suggest_int("degree", 2, 3) if knots else 3, wins=wins or None,
                cols=COLS_D2 if t.suggest_categorical("cols", ["d1", "d2"]) == "d2" else COLS_D1)


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cell: dict) -> np.ndarray:
    """Ridge on c, fit on tr and returned as a taxi time on te; lsqr never forms the normal equations."""
    enc = Enc(tr, cell["cols"], cell["min_freq"], cell["knots"], cell["degree"], cell["wins"])
    m = Ridge(alpha=cell["alpha"], solver="lsqr", fit_intercept=True).fit(enc(tr), tr.c.values)
    return floored(m.predict(enc(te)) + te.z_aobt.values)


def oof(tra: pd.DataFrame, cell: dict) -> np.ndarray:
    """Out-of-fold taxi times over all five declared month pairs; ridge is cheap enough not to need O2's three-fold subset."""
    p = np.full(len(tra), np.nan)
    for fold in CV_FOLDS:
        m = tra.month.isin(fold).values
        p[m] = fit_predict(tra[~m], tra[m], cell)
    return p


def score(cell: dict, tra: pd.DataFrame, base: np.ndarray) -> tuple[float, float, float]:
    """Pooled BLEND RMSE at a weight fitted on these rows, with ridge's own level and its agreement with the base beside it."""
    p = oof(tra, cell)
    ok = np.isfinite(p) & np.isfinite(base)
    y, r, b = tra.y.values[ok], p[ok], base[ok]
    w = blend_weight(r, b, y)
    return rmse(floored(w * r + (1 - w) * b) - y), rmse(r - y), best_blend(b - y, r - y)[2]


def holdout(tra: pd.DataFrame, tea: pd.DataFrame, cell: dict) -> np.ndarray:
    """The best cell fit on every training month and predicted on the holdout, cached so a report re-run is free."""
    key = json.dumps({k: (v if k != "cols" else len(v)) for k, v in cell.items()}, sort_keys=True)
    if os.path.exists(CACHE):
        c = pd.read_parquet(CACHE)
        if len(c) == len(tea) and c.key.iloc[0] == key and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    p = fit_predict(tra, tea, cell)
    pd.DataFrame({"id": tea.id.values, "pred": p, "key": key}).to_parquet(CACHE, index=False)
    return p


def main(out: str = "docs/RC1.md") -> None:
    tra, tea = lane_a()
    base, t0 = base_oof(tra), time.time()

    def objective(t: optuna.Trial) -> float:
        b, lvl, rho = score(suggest(t), tra, base)
        t.set_user_attr("level", lvl)
        t.set_user_attr("agreement", rho)
        t.set_user_attr("min", (time.time() - t0) / 60)
        print(f"| {t.number} | {b:.2f} | {lvl:.2f} | {rho:.4f} |", flush=True)
        return b

    study = optuna.create_study(direction="minimize", storage=STORE, study_name="rc1", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0))
    if not study.trials:  # trial 0 is plain ridge, no spline and no winsorising, so the search measures its own base
        study.enqueue_trial(dict(alpha=1.0, min_freq=20, knots=0, wins=0.0, degree=3, cols="d2"))

    def enough(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if len(st.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)) >= TRIALS:
            st.stop()  # n_trials counts ATTEMPTS while k below counts COMPLETIONS, so budget the completions

    print("| trial | blend RMSE | ridge alone | agreement |\n|---|---|---|---|", flush=True)
    study.optimize(objective, n_trials=None, timeout=BUDGET_S, callbacks=[enough])

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    k = len(done)
    rows = [f"| {t.number} | {t.value:.2f} | {t.user_attrs['level']:.2f} | {t.user_attrs['agreement']:.4f} |" for t in done]
    lean = min(done, key=lambda t: t.user_attrs["level"])  # the Stage-3 linear baseline, selected on its OWN criterion

    cell = suggest(optuna.trial.FixedTrial(study.best_params))
    p_oof = oof(tra, cell)
    ok = np.isfinite(p_oof) & np.isfinite(base)
    w = blend_weight(p_oof[ok], base[ok], tra.y.values[ok])
    rc_te, sb3_te, y = holdout(tra, tea, cell), sb3_predict(tra, tea), tea.y.values
    e0, ex = sb3_te - y, rc_te - y
    v = paired_delta(e0, floored(w * rc_te + (1 - w) * sb3_te) - y, tea.day.values)
    ow, orm, rho = best_blend(e0, ex)
    shown = {n: (val if n != "cols" else ("d2" if val is COLS_D2 else "d1")) for n, val in cell.items()}

    text = "\n".join([
        "# RC1: ridge regression searched with Optuna, lane A, Jan + Jul 2025 holdout", "",
        f"Regenerate with `python -m taxi.rc1`. Declared in protocol section 4 before the code was written. Optuna TPE, seed 0, "
        f"six knobs, {k} completed trials of {TRIALS} budgeted, {(time.time() - t0) / 3600:.1f} h. The campaign's FIRST linear "
        f"model on lane A: every other lane-A model is a boosted tree or a neural net, so this is both the Stage-3 linear "
        f"baseline the campaign never had and a blend partner from the most distant family it can reach. The objective is the "
        f"pooled BLEND RMSE over all five `CV_FOLDS` month pairs against MC2's cached out-of-fold predictions, never ridge's own "
        f"level, for NN3's reason: what a partner adds is set by level and agreement together.", "",
        "## Trials", "", "| trial | blend RMSE | ridge alone | agreement |", "|---|---|---|---|", *rows, "",
        f"Best cell on the blend objective: `{shown}`.", "",
        "## The Stage-3 linear baseline", "",
        f"The best RIDGE-ALONE trial is selected on a different criterion from the same {k}: trial "
        f"{lean.number} at **{lean.user_attrs['level']:.2f} s** of cross-validated RMSE, against D2's recorded 211.8 s on the "
        f"same five folds. That gap is the answer Stage 4 asks for, that a richer class must beat the simple one, and it is "
        f"reported here whichever way it falls.", "",
        f"Weight on RC1, fitted out of fold on {int(ok.sum()):,} rows: **{w:.4f}**. The holdout is never read to choose it.", "",
        f"## Holdout ({len(tea):,} rows)", "", "| model | RMSE |", "|---|---|",
        f"| SB3 (champion) | {rmse(e0):.2f} |", f"| RC1 alone (diagnostic) | {rmse(ex):.2f} |",
        f"| SB3 + RC1 at w={w:.4f} (challenger) | {rmse(floored(w * rc_te + (1 - w) * sb3_te) - y):.2f} |", "",
        f"RC1 reaches {rmse(ex):.2f} s at agreement **{rho:.4f}** with SB3, against NN3's 233.48 s. The oracle weight puts "
        f"{ow:.3f} on SB3 and reaches {orm:.2f} s, {rmse(e0) - orm:+.2f} s, which upper-bounds any weight an out-of-fold choice "
        f"could have picked.", "",
        verdict_line("SB3 + RC1", "SB3", v, k=max(k, 1)), ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
