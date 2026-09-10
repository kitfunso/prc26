"""The model-class register as DATA, so a missing family fails a test instead of reading as complete.

`docs/COVERAGE.md` section 4 kept this as prose in a table, and that is how ridge went missing for 60 ledger rows: a class
nobody tried produced no row, so the section read as complete precisely because the gap left no trace. Here every row names the
scikit-learn estimators it owns, and `tests/test_model_class_register.py` enumerates the installed library against it. A new
family cannot be silently absent, because the library itself is the checklist.
"""
from __future__ import annotations

import io

from typing import NamedTuple


class Row(NamedTuple):
    family: str
    covers: tuple[str, ...]  # scikit-learn regressor class names this family owns; empty for a family sklearn does not ship
    instance: str  # the campaign lane that instantiated it, or "" while the row stands empty
    status: str


# Families sklearn ships. Every regressor in `all_estimators("regressor")` belongs to exactly one row.
SKLEARN: tuple[Row, ...] = (
    Row("constant / group mean", ("DummyRegressor",),
        "B0 global, B1 per airport, B2 airport x hour x weekday", "TRIED (H1-H4)"),
    Row("ordinary least squares",
        ("LinearRegression", "Lars", "LarsCV", "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV"),
        "B4 per-airport OLS, B3 gap family", "TRIED (H5-H11)"),
    Row("penalised linear",
        ("Ridge", "RidgeCV", "Lasso", "LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC", "ElasticNet", "ElasticNetCV",
         "SGDRegressor", "PassiveAggressiveRegressor"),
        "RC1 ridge (Optuna over 6 knobs); FAM2-en elastic net", "TRIED (RC1, FAM2): en closed at 295.87"),
    Row("Bayesian linear", ("ARDRegression", "BayesianRidge"), "FAM2-brr", "TRIED (FAM2): closed at 295.57"),
    Row("robust / quantile linear", ("HuberRegressor", "QuantileRegressor", "TheilSenRegressor", "RANSACRegressor"),
        "lane B `taxi.l2.huber`; FAM2-hub on lane A", "TRIED (lane B, FAM2): hub closed at 269.42"),
    Row("non-Gaussian GLM", ("PoissonRegressor", "GammaRegressor", "TweedieRegressor"),
        "FAM2-twd, Tweedie on the level", "TRIED (FAM2): closed at 272.42"),
    Row("latent-factor regression", ("PLSRegression", "PLSCanonical", "CCA"), "FAM2-pls",
        "TRIED (FAM2): closed at 295.98"),
    Row("support-vector regression", ("SVR", "NuSVR", "LinearSVR"), "FAM2-svr, the linear case",
        "TRIED (FAM2): closed at 274.98"),
    Row("kernel methods", ("KernelRidge", "GaussianProcessRegressor"),
        "FAM1-nys, an RBF kernel approximated by Nystroem then ridged", "TRIED (FAM1)"),
    Row("k-nearest neighbours", ("KNeighborsRegressor", "RadiusNeighborsRegressor"), "FAM1-knn", "TRIED (FAM1)"),
    Row("single decision tree", ("DecisionTreeRegressor", "ExtraTreeRegressor"),
        "none, and none is planned: it is a random forest of one, so FAM1-rf covers the class", "subsumed"),
    Row("bagged trees", ("RandomForestRegressor", "BaggingRegressor"), "FAM1-rf", "TRIED (FAM1)"),
    Row("randomised trees", ("ExtraTreesRegressor",), "FAM1-et", "TRIED (FAM1)"),
    Row("gradient-boosted trees", ("GradientBoostingRegressor", "HistGradientBoostingRegressor"),
        "M1, M2, D1, D2 LightGBM; MC1, MC2, SB1-SB3 CatBoost; X1 XGBoost", "TRIED, and the champion"),
    Row("adaptive boosting", ("AdaBoostRegressor",), "FAM2-ada", "TRIED (FAM2): closed at 428.12"),
    Row("feed-forward neural net", ("MLPRegressor",), "NN1, NN2, NN3", "TRIED (H57-H59)"),
    Row("isotonic / monotone", ("IsotonicRegression",), "MO1-mono, M1's lane-A cell under `monotone_constraints`",
        "TRIED (MO1): closed, +0.55 s worse out of fold than the same cell free"),
    Row("ensembles and stacking", ("StackingRegressor", "VotingRegressor"),
        "E1, ST1, BF, the SB seed bags", "TRIED (H21, H52)"),
    # Wrappers around another estimator, and the multi-output-only penalised fits. Neither is a model class, and the target
    # here is one scalar column, so these are carried to keep the enumeration total rather than to be built.
    Row("not a model class: meta-wrapper or multi-output only",
        ("MultiOutputRegressor", "RegressorChain", "TransformedTargetRegressor", "MultiTaskElasticNet",
         "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV"),
        "", "N/A"),
)

# Families with no scikit-learn estimator, so the enumeration above cannot reach them. They are listed by hand and an entry
# here is a claim the reviewer must check, not a claim the test can.
OUTSIDE: tuple[Row, ...] = (
    Row("generalised additive (spline basis, penalised)", (),
        "RC1's `knots` arms ARE an additive spline fit under an L2 penalty", "TRIED inside RC1"),
    Row("prior-fitted transformer", (), "M3 and TP2 lane A, LP1 lane B", "TRIED (H12, H55)"),
    Row("piecewise-linear trees", (), "FAM2-lt one fixed cell, then LT2's cell searched over 40 TPE trials",
        "TRIED (FAM2 H62, LT2 H64): cleared both screens, lost the holdout at +0.9 s against a 2 s floor"),
    Row("explainable boosting (GA2M)", (), "FAM3-ebm, one fixed cell of `interpret` 0.7.8",
        "TRIED (FAM3): closed at 237.10, +0.13 s of blend against a 1.0 s gate"),
)

ROWS: tuple[Row, ...] = SKLEARN + OUTSIDE


def empty() -> tuple[Row, ...]:
    return tuple(r for r in ROWS if r.status.startswith("EMPTY"))


HEAD = "| model class | instance | status |\n|---|---|---|"


def table() -> str:
    return "\n".join([HEAD] + [f"| {r.family} | {r.instance or '(none)'} | {r.status} |" for r in ROWS])


def sync_doc(path: str = "docs/COVERAGE.md") -> str:
    """Swap the doc's table for the generated one. Builds the whole file in memory before opening it for write, because
    an `open(path, "w")` in the same expression as its content truncates the doc when that expression raises."""
    doc = io.open(path, encoding="utf-8").read()
    i = doc.index(HEAD)
    out = doc[:i] + table() + doc[doc.index("\n\n", i):]
    assert len(out) > len(table()) + 1000, "refusing to write a doc that lost its prose"
    io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    return f"{path}: register table synced, {len(ROWS)} families"


if __name__ == "__main__":
    print(table())
    print(sync_doc())
    print(f"\n{len(ROWS)} families, {len(empty())} standing EMPTY, "
          f"{sum(len(r.covers) for r in ROWS)} scikit-learn regressors classified.")
