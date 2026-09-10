"""L5: lane-B airline mixture. LIRF's no-type rows get P(copy) per row, times z_sched, plus (1 - P) times the band's non-copy mean; others keep L4b."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.l3 import COPY, all_rows, by_airport
from taxi.l4 import CELLS as L4_CELLS, EDGES, MIN_BAND, band, by_band, per_band, touched
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 2
PSEUDO, MIN_AIRLINE = 10.0, 15
NB = len(EDGES) + 1
Prob = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
Pred = Callable[[pd.DataFrame, pd.DataFrame, Any], np.ndarray]
Champ = tuple[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]]


def is_copy(t: pd.DataFrame, copy_s: float = COPY) -> np.ndarray:
    return np.abs(t.y.values - t.z_sched.values) <= copy_s


def champion(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return per_band(tr, te, L4_CELLS["L4b"])


def shrunk(t: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L5a: P(copy) per airline and band, shrunk toward the band share with PSEUDO rows; an unseen airline gets the band share."""
    c, tb, zb = is_copy(t), band(t.z_sched.values), band(te.z_sched.values)
    p_band = pd.Series(c).groupby(tb).mean().reindex(range(NB)).fillna(c.mean()).values[zb]
    g = pd.DataFrame({"al": t.airline.values, "b": tb, "c": c}).groupby(["al", "b"]).c.agg(["sum", "size"])
    s = g.reindex(pd.MultiIndex.from_arrays([te.airline.values, zb]))
    return (s["sum"].fillna(0).values + PSEUDO * p_band) / (s["size"].fillna(0).values + PSEUDO)


def _stand1(df: pd.DataFrame) -> pd.Series:
    return df.stand.astype(object).fillna("").str[:1]  # stand is categorical in load(); object first so a missing stand becomes ""


def _design(df: pd.DataFrame, airlines: list[str], stands: list[str]) -> np.ndarray:
    al = np.where(df.airline.isin(airlines), df.airline, "other")
    parts = (pd.Categorical(al, categories=[*airlines, "other"]), pd.Categorical(band(df.z_sched.values), categories=range(NB)),
             pd.Categorical(df.hour.values, categories=range(24)), pd.Categorical(_stand1(df), categories=stands))
    return np.column_stack([pd.get_dummies(p).values.astype(float) for p in parts])


def logistic(t: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L5b: L2 logistic regression of the copy flag on airline, band, hour of takeoff and the stand's first character."""
    n = t.airline.value_counts()
    airlines, stands = sorted(n[n >= MIN_AIRLINE].index), sorted(_stand1(t).unique())
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(_design(t, airlines, stands), is_copy(t))
    return clf.predict_proba(_design(te, airlines, stands))[:, 1]


def per_row(tr: pd.DataFrame, te: pd.DataFrame, prob: Prob) -> np.ndarray:
    """L4b everywhere, then each touched row gets P(copy) x z_sched + (1 - P(copy)) x its band's non-copy mean."""
    out, m = champion(tr, te), touched(tr, te)
    for apt in np.unique(te.apt.values[m]):
        t = tr[(tr.apt == apt) & (tr.notyp == 1)].reset_index(drop=True)
        ix = np.flatnonzero(m & (te.apt.values == apt))
        sub, c, tb = te.iloc[ix], is_copy(t), band(t.z_sched.values)
        mu_all = float(t.y.values[~c].mean()) if (~c).any() else float(t.y.mean())
        mu = np.array([float(t.y.values[(tb == b) & ~c].mean()) if int((tb == b).sum()) >= MIN_BAND and ((tb == b) & ~c).any() else mu_all for b in range(NB)])
        p = prob(t, sub)
        out[ix] = p * sub.z_sched.values + (1 - p) * mu[band(sub.z_sched.values)]
    return out


CELLS: dict[str, Prob] = {"L5a": shrunk, "L5b": logistic}


def cv(trb: pd.DataFrame, cells: dict[str, Any], pred: Pred, champ: Champ = ("L4b", champion)) -> tuple[str, list[str], dict[str, float]]:
    """Month-pair CV of the named champion and every cell, all floored; returns the best cell, the table lines and every CV RMSE."""
    names = [champ[0], *cells]
    lines = ["| fold | " + " | ".join(f"{k} RMSE" for k in names) + " |", "|---|" + "---|" * len(names)]
    sq, n = {k: 0.0 for k in names}, 0
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        tr, te, row = trb[~m].reset_index(drop=True), trb[m].reset_index(drop=True), []
        for k in names:
            p = champ[1](tr, te) if k == champ[0] else pred(tr, te, cells[k])
            e = floored(p) - te.y.values
            sq[k], row = sq[k] + float(e @ e), row + [f"{rmse(e):.1f}"]
        n += len(te)
        lines.append(f"| {fold} | " + " | ".join(row) + " |")
        print(lines[-1], flush=True)
    cvr = {k: float(np.sqrt(sq[k] / n)) for k in names}
    best = min(cells, key=cvr.get)
    lines.append("\nCV RMSE over the five folds (every cell floored at 0 s): " + ", ".join(f"{k} {v:.1f} s" for k, v in cvr.items()) + f"; best declared cell {best}.")
    print(lines[-1], flush=True)
    return best, lines, cvr


def judge(cells: dict[str, Any], pred: Pred, title: str, blurb: str, path: str, champ: Champ = ("L4b", champion), k: int = K,
          same_as: str | None = None) -> None:
    """CV over the declared cells, then the CV winner alone against the floored champion on the lane-B holdout (not touched when the winner
    is same_as, or lost the CV outright); writes the report to path.

    A cell CV already rejected cannot decide anything on the holdout, so reading it there spends a read on a foregone verdict: L12 spent
    one before this gate existed.
    """
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb = (d[~d.has_fpl].reset_index(drop=True) for d in (tr, te))
    best, lines, cvr = cv(trb, cells, pred, champ)
    name = champ[0]
    if best == same_as or cvr[best] >= cvr[name]:
        why = (f"The CV winner {best} is {name} itself" if best == same_as else
               f"The CV winner {best} reads {cvr[best]:.1f} s against {name}'s {cvr[name]:.1f} s, so no declared cell beat the champion")
        text = "\n".join([f"# {title}, Jan+Jul 2025 development holdout", "", f"Train months {TRAIN_MONTHS}. {blurb} k = {k}.", "",
                          "## Month-pair CV (lane-B rows of the fold)", ""] + lines + ["", "## Holdout: not touched", "",
                          f"{why}; by the declaration no cell meets the holdout and {name} stays."]) + "\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return
    p0, p1 = floored(champ[1](trb, teb)), floored(pred(trb, teb, cells[best]))
    e0, e1 = p0 - teb.y.values, p1 - teb.y.values
    v, m = paired_delta(e0, e1, teb.day.values, lane="B"), touched(trb, teb)
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {e.min():.0f} | {e.max():.0f} |"
    report = [f"# {title}, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}. Touched rows as L4 ({sorted(teb.apt[m].unique())}); every other row keeps L4b. {blurb} Every cell and the "
              f"champion carry the floor at 0 s (H22). k = {k}; the CV winner alone meets the holdout.", "",
              "## Month-pair CV (lane-B rows of the fold)", ""] + lines + ["",
              f"## Holdout verdict vs champion {name} with the floor ({len(teb)} rows; {best} differs from it on {int((p1 != p0).sum())} rows)", "",
              "| model | RMSE | Jan | Jul | min error | max error |", "|---|---|---|---|---|---|", row(f"{name} floored", e0), row(f"{best} floored", e1), "",
              verdict_line(best, name, v, k=k), "", "## By airport (reported)", ""] + by_airport(teb, {name: e0, best: e1}) + ["",
              f"## Touched rows by band (reported, {int(m.sum())} rows)", ""] + by_band(teb, m, {name: e0, best: e1}) + ["",
              f"All rows with {name} (reported, never decides): {all_rows(te, e0)}; with {best}: {all_rows(te, e1)}."]
    text = "\n".join(report) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main() -> None:
    judge(CELLS, per_row, "L5: lane-B airline mixture",
          f"Regenerate with `python -m taxi.l5`. Prediction = P(copy) x z_sched + (1 - P(copy)) x the band's non-copy mean (copy = |y - z_sched| <= "
          f"{COPY:.0f} s). L5a: P(copy) per airline and band shrunk toward the band share with {PSEUDO:.0f} pseudo-rows. L5b: L2 logistic regression on "
          f"airline (at least {MIN_AIRLINE} training rows, else other), band, hour of takeoff and the stand's first character.", "docs/L5.md")


if __name__ == "__main__":
    main()
