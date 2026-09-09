"""Every declared search as data, because a declaration that lives only in prose cannot be checked.

O1, O2 and NN3 were each declared correctly in protocol section 4 and each stopped short of its declared k without
anything going red: O1 killed by hand at 8 of 40, O2 ended by a 10 h clock that 40 trials could never fit inside,
NN3 counting attempts where its own report counts completions. `tests/test_search_completion.py` reads this file
and the study databases so that cannot happen silently again.

A fourth shape was found on 2026-09-10 and closed here: the declared rate itself can be wrong. `declared_s` was a
hand-copied mean over COMPLETE trials, which cannot see the wall clock burned by trials that never complete, so O2's
real 73.4 min a trial read as 40 and rule A called an infeasible search feasible. The rate is now READ from the study.
"""
from __future__ import annotations

import importlib
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime

SHORT_DOC, SHORT_MARK = "docs/NOTDONE-PLAN.md", "## Short searches"
MIN_TRIALS = 3  # fewer completions than this and one slow trial sets the rate, so the study has no rate yet


@dataclass(frozen=True)
class Search:
    name: str
    module: str
    db: str
    study: str
    k: int
    declared_s: float  # the rate that justified the clock, frozen at declaration; `measured_s` is what the study did
    clock: str
    doc: str


# declared_s is what was on the record when each clock was set: O1 32.75 min over 8, O2 40.0 over 4,
# NN3 6.6 over 30, all read 2026-09-08 and rounded up. RC1 predates its own study, so its rate is a timed full-size
# trial in the SLOWEST corner of its space (knots=8, 6.2 min; knots=0 reads 2.3), recorded before the search launched. LT2's 21 min is the same shape: its slowest declared corner timed at 100 trees
# on 2026-09-09 and scaled to the 3000 the tie `n_estimators = 60 / lr` reaches at the low end of the rate, before its first trial.
SEARCHES = (
    Search("O1", "taxi.o1", "data/o1.db", "o1", 40, 33 * 60, "TIMEOUT", "docs/O1.md"),
    Search("O2", "taxi.o2", "data/cache/o2.db", "o2", 40, 40 * 60, "TIMEOUT", "docs/O2.md"),
    Search("NN3", "taxi.nn3", "data/cache/nn3.db", "nn3", 40, 7 * 60, "BUDGET_S", "docs/NN3.md"),
    Search("RC1", "taxi.rc1", "data/cache/rc1.db", "rc1", 40, 7 * 60, "BUDGET_S", "docs/RC1.md"),
    Search("LT2", "taxi.lt2", "data/cache/lt2.db", "lt2", 40, 21 * 60, "TIMEOUT", "docs/LT2.md"),
)


def clock_s(s: Search) -> float:
    """The wall clock read off the module itself, so the budget and the test can never drift apart."""
    return float(getattr(importlib.import_module(s.module), s.clock))


def achieved(s: Search) -> int:
    """COMPLETE trials only: a pruned trial is an attempt, and the reports all count completions."""
    if not os.path.exists(s.db):
        return 0
    con = sqlite3.connect(f"file:{s.db}?mode=ro", uri=True)
    try:
        return int(con.execute("SELECT count(*) FROM trials t JOIN studies s ON s.study_id = t.study_id "
                               "WHERE s.study_name = ? AND t.state = 'COMPLETE'", (s.study,)).fetchone()[0])
    finally:
        con.close()


def measured_s(s: Search) -> float | None:
    """Wall seconds per COMPLETED trial, read off the study: first start to last completion over the completion count.

    Not the mean of the trials' own durations. A trial that fails or is pruned still burns clock and still crowds out a
    completion, so the number that decides whether k fits inside the wall clock has to count that time. O2 reads 36.5 min
    by own-duration and 73.4 by this, which is the whole difference between feasible and not.
    """
    if not os.path.exists(s.db):
        return None
    con = sqlite3.connect(f"file:{s.db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT t.state, t.datetime_start, t.datetime_complete FROM trials t "
                           "JOIN studies st ON st.study_id = t.study_id WHERE st.study_name = ?", (s.study,)).fetchall()
    finally:
        con.close()
    starts = [datetime.fromisoformat(a) for _, a, _ in rows if a]
    ends = [datetime.fromisoformat(b) for _, _, b in rows if b]
    n = sum(1 for st, _, _ in rows if st == "COMPLETE")
    if n < MIN_TRIALS or not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() / n


def rate_s(s: Search) -> tuple[float, str]:
    """The rate feasibility is judged on: the study's own when it has one, the declared rate while it does not."""
    m = measured_s(s)
    return (m, "measured") if m is not None else (s.declared_s, "declared")


def reap(study: "optuna.Study") -> int:
    """Mark trials left RUNNING by a killed process FAIL, so the report counts them and TPE stops re-suggesting the same cell.

    Safe only because `scripts/gpu_queue.sh` runs one process per study at a time: a RUNNING trial at startup is always a corpse.
    """
    import optuna

    n = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
            n += 1
    return n


def short() -> tuple[Search, ...]:
    return tuple(s for s in SEARCHES if achieved(s) < s.k)


def remaining(module: str) -> int:
    """Trials a declared search still owes; -1 for a module this file does not declare, so a queue runs that one once."""
    for s in SEARCHES:
        if s.module == module:
            return max(0, s.k - achieved(s))
    return -1


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "remaining":
        print(remaining(sys.argv[2]))
        return
    print("| search | k achieved | k declared | declared min/trial | measured min/trial | need (h) | clock (h) | feasible |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SEARCHES:
        r, src = rate_s(s)
        need, clk, m = s.k * r, clock_s(s), measured_s(s)
        print(f"| {s.name} | {achieved(s)} | {s.k} | {s.declared_s / 60:.0f} | "
              f"{'-' if m is None else format(m / 60, '.0f')} | {need / 3600:.1f} | {clk / 3600:.0f} | "
              f"{'yes' if need <= clk else 'NO'} ({src}) |")


if __name__ == "__main__":
    main()
