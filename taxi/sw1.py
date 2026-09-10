"""SW1: Romano-Wolf step-down over the lane-A challengers that share the holdout (protocol Stage 2, Stage 6 item 2).

Every verdict in this campaign prices the cells inside ONE lane and then applies a flat 2 s floor. Neither instrument sees
that 60 lanes have met the same 339,046 rows. Step-down controls the family-wise error rate across lanes and, unlike an
N x alpha bound, does not over-punish lanes whose errors are almost the same vector. It names which lanes survive.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from taxi.champ import lane_a
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse

REPS, SEED, ALPHA = 2000, 0, 0.05
NN2_W = 0.4160  # NN2's recorded out-of-fold blend weight; never re-fitted here
NN3_W = 0.5189  # NN3's own, from `docs/NN3.md`; each blend lane keeps the weight its verdict used
LT2_W = 0.5307  # LT2's, from `docs/LT2.md`


def cached(name: str, ids: np.ndarray) -> np.ndarray:
    d = pd.read_parquet(f"data/cache/{name}.parquet")
    return pd.DataFrame({"id": ids}).merge(d[["id", "pred"]], on="id", how="left").pred.values


RC1_W = 0.013593  # SB3+RC1's, from ledger S7; the weight the v17 file shipped at

# THE catalogue of cached lane-A holdout vectors: lane name -> (cache key, blend weight against SB3, or None when the cached
# vector is already a level). `taxi.bf` and `taxi.ms1` read it too. It was typed in all three until 2026-09-10, and BF's copy
# was the one nobody updated: NN3 and LT2 were absent from the stack bound `docs/CEIL.md` quotes.
LANES = {"MC2": ("mc2_holdout", None), "D1": ("d1_holdout", None), "M3": ("m3_holdout", None),
         "W1-A": ("w1a_holdout", None), "SB3+RC1": ("rc1_holdout", RC1_W), "SB3+NN2": ("nn2_holdout", NN2_W),
         "SB3+NN3": ("nn3_holdout", NN3_W), "SB3+LT2": ("lt2_holdout", LT2_W)}


def on_disk(names: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The catalogue entries whose vector is actually written, in registry order. Empty `names` means every entry."""
    want = names or tuple(LANES)
    return tuple(n for n in want if os.path.exists(f"data/cache/{LANES[n][0]}.parquet"))


def raw(name: str, ids: np.ndarray) -> np.ndarray:
    """The cached vector as written: for a blend lane that is the PARTNER alone, which is what `taxi.bf` blends itself."""
    return floored(cached(LANES[name][0], ids))


def lane(name: str, ids: np.ndarray, sb3: np.ndarray) -> np.ndarray:
    """The lane as its own verdict read it: the cached level, or the partner blended onto SB3 at the weight that shipped."""
    v, w = raw(name, ids), LANES[name][1]
    return v if w is None else floored(w * v + (1 - w) * sb3)


def family(tea: pd.DataFrame, sb3: np.ndarray) -> dict[str, np.ndarray]:
    """The cached lane-A challengers, each floored the way its own verdict floored it. Absent caches are skipped, and a
    bigger family only makes step-down STRICTER, so a lane arriving here can never manufacture a survivor."""
    ids = tea.id.values
    out = {n: lane(n, ids, sb3) for n in on_disk()}
    if all(os.path.exists(f"data/cache/sb1_seed{i}_holdout.parquet") for i in (1, 2, 3, 4)):
        seeds = [cached("mc2_holdout", ids)] + [cached(f"sb1_seed{i}_holdout", ids) for i in (1, 2, 3, 4)]
        out["SB1"] = floored(np.mean(seeds, axis=0))
    return out


def boot(errs: dict[str, np.ndarray], e0: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One shared resample of whole days for every lane, so the joint dependence across lanes is kept."""
    _, grp = np.unique(days, return_inverse=True)
    nb = int(grp.max()) + 1
    n = np.bincount(grp, minlength=nb).astype(float)
    pick = np.random.default_rng(SEED).integers(0, nb, size=(REPS, nb))
    tot = n[pick].sum(axis=1)
    sse = lambda e: np.bincount(grp, weights=e * e, minlength=nb)[pick].sum(axis=1)  # noqa: E731
    r0 = np.sqrt(sse(e0) / tot)
    return np.column_stack([r0 - np.sqrt(sse(e) / tot) for e in errs.values()]), np.array([rmse(e0) - rmse(e) for e in errs.values()])


def stepdown(d: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Romano-Wolf: reject the largest studentised statistic against the max of the CENTRED null over lanes still active."""
    sd = draws.std(axis=0, ddof=1)
    t = d / sd
    centred = (draws - d) / sd
    p, active, prev = np.zeros(len(d)), np.ones(len(d), dtype=bool), 0.0
    while active.any():
        mx = centred[:, active].max(axis=1)
        step = np.maximum(np.array([float(np.mean(mx >= ti)) for ti in t]), prev)  # monotone: a later step never lowers a p
        rej = active & (step <= ALPHA)
        if not rej.any():
            p[active] = step[active]
            break
        p[rej], prev, active = step[rej], float(step[rej].max()), active & ~rej
    return t, p


def main(dst: str = "docs/SW1.md") -> None:
    tra, tea = lane_a()
    y, days = tea.y.values, tea.day.values
    sb3 = sb3_predict(tra, tea)
    e0 = sb3 - y
    fam = family(tea, sb3)
    errs = {k: v - y for k, v in fam.items()}
    draws, d = boot(errs, e0, days)
    t, p = stepdown(d, draws)

    order = np.argsort(-d)
    names = list(fam)
    rows = []
    for i in order:
        v = paired_delta(e0, errs[names[i]], days)
        rows.append(f"| {names[i]} | {rmse(errs[names[i]]):.2f} | {d[i]:+.2f} | [{v.lo:+.2f}, {v.hi:+.2f}] | {t[i]:+.2f} | "
                    f"{p[i]:.4f} | {'SURVIVES' if p[i] <= ALPHA else 'no'} |")
    surv = [names[i] for i in order if p[i] <= ALPHA]

    out = ["# SW1: Romano-Wolf step-down over the lane-A family, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.sw1`. Declared in protocol section 4 before the run. Champion SB3 floored at "
           f"{rmse(e0):.4f} s on {len(y):,} rows; {len(fam)} challengers, each tested against it on identical rows. "
           f"{REPS:,} day-block resamples, seed {SEED}, ONE index set shared by every lane so the dependence between lanes is "
           "preserved; statistics are studentised by their own bootstrap SD and the null is the centred distribution. "
           f"Family-wise error rate {ALPHA:.0%}. This lane promotes nothing: `taxi.verdict.FLOOR` is still the promotion rule.", "",
           "## The family", "",
           "| lane | holdout RMSE | improvement on SB3 | unadjusted 95% | studentised t | RW p | at 5% FWER |",
           "|---|---|---|---|---|---|---|", *rows, "",
           ("Survives step-down: " + ", ".join(surv) + "." if surv else "Nothing survives step-down."), "",
           "## What this does and does not say", "",
           f"The family is the {len(fam)} lanes whose per-row holdout errors are cached on disk, not the 60 in the ledger, "
           "because the other lanes' vectors were never written. A larger family makes step-down STRICTER, so surviving here is necessary "
           "and not sufficient. What it does settle is the direction the campaign has never tested: the 2 s floor is a constant "
           "chosen once, while step-down is a measurement of what the shared holdout can support. A lane that survives at 5% "
           "FWER is not a multiple-comparisons artifact, whatever the floor says about it.", ""]
    text = "\n".join(out)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
