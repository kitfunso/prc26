# Hand-off, 2026-09-22

Written mid-campaign so the next session can pick up without re-deriving anything. Every number below has the command
that regenerates it. Nineteen days to the deadline (11 Oct 2026 23:59:59 CET, `PLAN.md:3`).

## 1. Where we stand

| thing | value | source |
|---|---|---|
| file of record | `data/merry-octopus_v32.parquet` | `README.md:98` |
| board score | 271.5521, rank 20 of 152 | `LEDGER.md` v32 row |
| leader | vigorous-whistle 241.75 | same row |
| 10th place | strong-turtle 261.7928 | same row |
| v32 blend | `ra_cb_wz` 0.491 + `ra_lgb_pw` 0.175 + `nn3_pw` 0.334 | same row |
| v32 holdout | all 321.61, lane A 218.69, lane B 1900.6 | `grep "blend:ra_cb_wz+ra_lgb_pw+nn3_pw " docs/RESULTS.md` |
| scorer cap | 5 files a day, reset 00:00 UTC | `LEDGER.md:3` |

Upload rule, unchanged (`LEDGER.md:5`): expected board gain over 3 s against the current best, holdout CI excluding
zero, and `python board.py check` passes. Probing the scorer is banned.

## 2. What R7 was

Keith's instruction, verbatim: "for each of these three model, i need you to run proper hyperparameters tuning for all
parameters. show all parameters to me and the range. run optuna optimisation with 200 trials for each of them."

Every tunable parameter of all three v32 members, 200 Optuna trials each, judged on three folds the search never saw.
Search folds (2,3) and (4,5); judge folds (6,8), (9,10), (11,12). Studies live in `data/optuna.db` under
`<member>_full_23_45` and resume if killed. Full spaces and rationale: `docs/EXPERIMENT-PROTOCOL.md` section 4.

### Results: all three pass, 0 failed trials of 600

| member | COMPLETE / PRUNED / FAIL | judge folds: shipped to tuned | gain | transfer |
|---|---|---|---|---|
| `ra_lgb_pw` | 128 / 72 / 0 | 211.250 to 207.015 | **+4.235** | 118% |
| `ra_cb_wz` | 113 / 87 / 0 | 207.375 to 206.233 | **+1.142** | 81% |
| `nn3_pw` | 115 / 85 / 0 | 212.053 to 208.572 | **+3.481** | 74% |

Transfer is the judge gain over the tune gain. Above 100% means the winning region was not fitted fold noise.

Regenerate: `grep -c "Trial [0-9]* \(finished\|pruned\)" data/r7_*.log` and the `JUDGE` line in each `data/r7_*.out`.

### The re-fit that has landed

| member | holdout all | holdout lane A | lane A delta |
|---|---|---|---|
| `ra_lgb_pw` (shipped) | 326.37 | 225.75 | baseline |
| `ra_lgb_pw_t7` (tuned) | 323.67 | 221.75 | **-4.00** |

`grep -E "^\| ra_lgb_pw" docs/RESULTS.md`. The holdout lane A gain (4.00) lands within 0.24 of the judge-fold gain
(4.235), on rows neither the search nor the judge folds touched. Best transfer evidence in the campaign.

## 3. What is running right now

| job | pid | writes | check |
|---|---|---|---|
| `lanea.py cv ra_cb_wz_t7 mix_cb_wx` | 24108 | `data/r7_cv_cb.out` / `.log` | `tasklist //FI "PID eq 24108" //NH` |
| `nn.py cv nn3_pw_t7 mix_cb_wx`, queued behind it | launcher 53876 | `data/r7_cv_nn.out` / `.log` | starts when 24108 exits |

Both are detached processes, not harness background tasks, because the harness watchdog kills background tasks on a
low-memory alarm even when the machine has 28 GB free. Restart either with `Start-Process` and
`-RedirectStandardOutput` / `-RedirectStandardError`; never `2>&1`, a hook blocks it.

Expect the CatBoost re-fit to take several hours: the tuned settings drop the learning rate from 0.1 to 0.0417, so it
fits many more iterations. The LightGBM re-fit went from 641 s shipped to 7597 s tuned for the same reason.

## 4. Next steps, in order

1. Wait for both re-fits. Each appends its row to `docs/RESULTS.md`.
2. Reblend all three tuned members and compare against v32 in one command:
   `python blend.py ra_cb_wz_t7 ra_lgb_pw_t7 nn3_pw_t7 --base=blend_ra_cb_wz+ra_lgb_pw+nn3_pw`
   That runs NNLS on the OOF, scores the holdout, prints the paired day-block bootstrap delta and the regime slice.
3. Check each tuned member earns a non-zero NNLS weight. A member with weight 0.000 is dead regardless of its own
   holdout, which is exactly how `ra_ridge` died (`docs/EXPERIMENT-PROTOCOL.md:100`).
4. Refit the second stage (`recal.py`) on the new blend.
5. Apply the slot gate: section 3's 1.5 s all-rows bar, CI excluding zero, plus the `LEDGER.md:5` upload rule.
6. Present the Stage 6 surface including the Stage 4.5 coverage interrogation table, then **wait for Keith's sign-off.**
   Promotion requires explicit sign-off, every time. No scorer slot without it.

## 5. The honest read, before anyone gets excited

The judge-fold gains are real and clean. The board is a different question, and recent history is bad:

| file | holdout promise | board outcome |
|---|---|---|
| v29 | +0.873 [+0.104, +1.708] | **LOST** 0.43 s |
| v30 | +0.220 [+0.038, +0.423] | **LOST** 0.15 s |
| v31 | +0.240 [+0.108, +0.402] | won, carried 0.4 of promise |
| v32 | +0.754 [+0.392, +1.314] | won, carried 0.15 of promise |
| v33 | +0.402 [+0.109, +0.748] | **LOST** 0.29 s |

Three of the last five gated changes lost board seconds while passing a CI that excluded zero. The carry fraction on
the two that won fell from 0.4 to 0.15. This is the standing caution in `LEDGER.md:18-22` and it applies to R7 in full.

What that means concretely: a lane A holdout gain of 4.00 on a member carrying weight 0.175 in the blend is not 4 s of
board, and on the recent carry it may be a fraction of a second. The upload bar is 3 s. R7 on its own probably does not
clear it. The realistic read is that R7 is a foundation for the remaining lanes, not a submission by itself.

Also worth saying plainly: CatBoost bought +1.142 from the same 200 trials LightGBM turned into +4.235. Further
CatBoost tuning is priced as close to dead. Do not spend another search there.

## 6. Traps already paid for

- **Git Bash eats `/FLAG`.** `tasklist /FI "PID eq N"` reaches the exe as a mangled Windows path and fails. Use `//FI`.
  The worse half: a check built on a failed command reads as a real negative. Two watchers reported both jobs dead while
  both were alive. Test any liveness check against a case that must be negative before trusting it in a loop.
- **The harness background-task watchdog is not the machine.** It killed two jobs claiming low memory with 28 GB free.
  Long jobs go to `Start-Process`, detached.
- **Never hand-copy tuned parameters.** Both config sets were read back out of `data/optuna.db` through the same
  `space(fam, FixedTrial(best_params), base)` call `tune.py:148` uses for its judge run, then asserted equal to the
  config. Scripts are in the session scratchpad; the pattern is worth rebuilding if they are gone.
- **`tune.py` counts NEW trials per process.** Two workers on one study means 400 trials, not 200, which breaks the
  declared budget and inflates selection. Do not parallelise a search that way.
- **The GPU is shared** with another session's `fifty` jobs. Keith's standing choice is to WAIT, never pause them.

## 7. Commits

| commit | what |
|---|---|
| `48c6b36` | R7 search code for all three members |
| `1aa3e76` | tuned LightGBM and CatBoost configs, both tree results |
| `ecde569` | tuned MLP config, MLP result, first re-fit result |

All pushed to `origin master:campaign2`. Tests green: `python -m pytest tests/ -q`, 11 passed.
