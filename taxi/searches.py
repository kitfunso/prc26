"""Every declared search as data, because a declaration that lives only in prose cannot be checked.

O1, O2 and NN3 were each declared correctly in protocol section 4 and each stopped short of its declared k without
anything going red: O1 killed by hand at 8 of 40, O2 ended by a 10 h clock that 40 trials could never fit inside,
NN3 counting attempts where its own report counts completions. `tests/test_search_completion.py` reads this file
and the study databases so that cannot happen silently again.
"""
from __future__ import annotations

import importlib
import os
import sqlite3
import sys
from dataclasses import dataclass

SHORT_DOC, SHORT_MARK = "docs/NOTDONE-PLAN.md", "## Short searches"


@dataclass(frozen=True)
class Search:
    name: str
    module: str
    db: str
    study: str
    k: int
    trial_s: float
    clock: str
    doc: str


# trial_s is MEASURED from each study's own completed trials, never an estimate: O1 32.75 min over 8, O2 40.0 over 4,
# NN3 6.6 over 30, all read 2026-09-08 and rounded up. RC1 predates its own study, so its rate is a timed full-size
# trial in the SLOWEST corner of its space (knots=8, 6.2 min; knots=0 reads 2.3), recorded before the search launched.
SEARCHES = (
    Search("O1", "taxi.o1", "data/o1.db", "o1", 40, 33 * 60, "TIMEOUT", "docs/O1.md"),
    Search("O2", "taxi.o2", "data/cache/o2.db", "o2", 40, 40 * 60, "TIMEOUT", "docs/O2.md"),
    Search("NN3", "taxi.nn3", "data/cache/nn3.db", "nn3", 40, 7 * 60, "BUDGET_S", "docs/NN3.md"),
    Search("RC1", "taxi.rc1", "data/cache/rc1.db", "rc1", 40, 7 * 60, "BUDGET_S", "docs/RC1.md"),
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
    print("| search | k achieved | k declared | measured min/trial | need (h) | declared clock (h) | feasible |")
    print("|---|---|---|---|---|---|---|")
    for s in SEARCHES:
        need, clk = s.k * s.trial_s, clock_s(s)
        print(f"| {s.name} | {achieved(s)} | {s.k} | {s.trial_s / 60:.0f} | {need / 3600:.1f} | {clk / 3600:.0f} | "
              f"{'yes' if need <= clk else 'NO'} |")


if __name__ == "__main__":
    main()
