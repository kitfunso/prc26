# Lane A second stage (`recal.py`)

**Question.** In lane A the recorded off-block is often a plan stamp (AOBT, EOBT, LOBT, IOBT or the schedule), so the
label y equals one of `z_aobt` ... `z_sched` on many rows. The lane A blend models c = y - `z_aobt` and hedges between
stamps. Does the blend's distance to each stamp carry information the blend did not use, and does a flight's own copy
history add to that?

**Design.** LightGBM (300 trees, learning rate 0.05, 63 leaves, at least 200 rows per leaf, row and column subsample
0.8) on e = y - p, where p is the v26/v27 blend 0.5223 `ra_cb_wz` + 0.1564 `ra_lgb_wz` + 0.3213 `nn3_wz` (OOF months
from the fold models, Jan and Jul from the holdout models). Columns: p, the airport, and z - p for `z_aobt`, `z_sched`,
`z_eobt`, `z_lobt` and `z_iobt`. It is fitted leave-one-month-group-out over six groups with the holdout pair as one
of them, so the holdout prediction comes from a stage trained on months 2 to 12 only. The prediction is
max(p + stage, 0). The parameters were fixed before the first run and not tuned.

**Result** (`python recal.py cv`): lane A holdout 220.43 to 218.60, +1.83 [+1.17, +2.57] (paired day-block
bootstrap); over the 12 months +1.04 [+0.65, +1.39].

Holdout lane A SSE by label class, M s^2 (classes taken in order: AOBT copy is |y - `z_aobt`| < 60 s, then EOBT, then
schedule; session scratchpad `mech_a.py`, log `mech_a.log`):

| class | blend | second stage |
|---|---|---|
| AOBT copy | 1,051 | 1,148 |
| EOBT copy | 1,322 | 1,272 |
| schedule copy | 1,732 | 1,552 |
| other | 12,369 | 12,229 |

The gain sits on schedule-copy rows (180M) and on the rest (140M), and AOBT-copy rows lose 97M. That fits a stage that
pulls predictions toward a stamp where the blend already leaned to it, at a cost on the rows that turn out to be AOBT
copies.

**Copy history adds nothing on the holdout.** The same stage plus 12 history columns, for each of schedule, EOBT, LOBT
and IOBT the share of lane A rows whose label copies that stamp by airport and airline, by airport and flight number,
and the log count (`laneb._hist_rates`, rows with the stamp 30+ min before takeoff, months 3+ away): lane A holdout
218.68 against 218.60, -0.08 [-0.52, +0.30]; 12 months +0.24 [+0.10, +0.43]. On rows with the stamp 30+ min before
takeoff, lane A copies are rare: schedule 0.013, EOBT 0.020, LOBT 0.020, IOBT 0.020. Not added.

**Linear leaves lose.** The same stage with linear leaves (`linear_tree=True`, `linear_lambda=1.0`), on the idea that a
leaf can fit the distance to a copied stamp as a slope where a constant leaf only reaches its average: lane A holdout
218.60 to 221.45, -2.85 [-8.34, +0.14]; 12 months -0.68 [-1.85, -0.06]. Holdout SSE on schedule-copy rows goes from
2,312M to 2,670M (classes taken in the order AOBT, schedule, EOBT, LOBT, IOBT), and no other class moves by more than
61M (session scratchpad `recal_lin.py`, log `recal_lin.log`). Not used.

**Degenerate AOBT stamps.** At six airports a few `z_aobt` minutes hold mostly rows whose recorded off-block sits 10+
min before the AOBT stamp (at least half of 30+ OOF rows in the airport and minute): 0 to 10 min at EDDF, EDDM, EGLL,
LEBL and LEMD, -2 to 0 min at LTFM (session scratchpad `mech_e.py`, log `mech_e.log`). There the label lies within 5
min of the EOBT on 42% of rows, the schedule on 41%, the LOBT on 38% and the IOBT on 49%. After the second stage these
rows are 1.1% of lane A in the OOF months with 2.4% of its SSE, bias -4.2 s, and 4,022 holdout rows with 3.2% (518M),
bias +7.5 s. The error there is spread, not a level: a perfect call on every such holdout row would take lane A from
218.60 to 215.07. No change.

**Regime context adds nothing** (session scratchpad `rc2.py`, log `rc2.log`). The same stage given where and when a
row sits (airline, stand, runway and hour, the airport-day and +-2 h stamp signatures of `docs/LANEA_CLF.md` section
4, the 20 window-state and 7 day-level columns; 48 columns): lane A holdout 218.60 to 219.53, -0.94 [-1.42, -0.47];
12 months -1.19 [-1.44, -0.81], worse in 11 of 12 months. With the occupancy, weather, arrival and plan-gap columns
as well (87 columns): 218.86, -0.26 [-0.78, +0.22]; 12 months -0.74 [-1.00, -0.36]. Stand and airline take 38 to 40%
of the split gain in both, and the loss sits on rows that copy no stamp (123M of the first variant's 139M). The first
stage already fits these columns, so a stage fitted to its out-of-fold residual finds month-specific noise in them.
The LIRF rows 30+ min after the schedule (59,604 train rows, holdout SSE 2,978M) show the same thing from the other
side (scratchpad `lirf_a.py`): the stage's mean sits within 30 s of the label mean in every slice by inbound block
against the schedule, EOBT and AOBT against the schedule, `occ` and the previous same-stand arrival's mode, and the
schedule-copy share moves with that arrival's mode (0.17 against 0.04) and the airline (ITY 0.013, THY 0.20), both
already first-stage columns. Not used.

**Full file** (session scratchpad `gate_a.py`): with v27's lane B, holdout 307.18 against v27's 308.47, +1.29 [+0.76,
+2.03]; with the copy-history lane B of `docs/LANEB.md` section 6, 306.76, +1.71 [+0.75, +2.88]. The second is v28.

**Schedule-copy head (v29, lost on the board).** The 1,302 LIRF schedule-copy rows hold 1,295M of the second stage's
holdout SSE (session scratchpad `lae_d.py`, `lae2_d.py`); a perfect call there takes LIRF lane A from 368.4 to 293.6.
Copy history is predictive of them: `laneb._hist_rates` scored on lane A rows (`copyhead.hist_a`) gives corr(h_flt,
copy) 0.546 on the delayed LIRF rows, and a binary LightGBM on the history, the stamps and identity columns
(`copyhead.py`) reaches AUC 0.959 to 0.966 by month group. Three ways to use it (`lah.py`, `lsh.py`, `rcp.py` in the
session scratchpad): as first-stage `ra_lgb_wz` columns, +0.18 [-0.09, +0.51] lane A, diluted by losses on physical
rows; as a direct correction v28 + w P (z_sched - v28) with w fitted on the OOF months (0.232), +0.035 [-0.188,
+0.204], because the stage already moves these rows; as a stage column, lane A holdout 218.60 to 218.28, +0.314
[+0.060, +0.558], 12 months +0.374 [+0.261, +0.476], full file +0.220 [+0.038, +0.423]. The third shipped in v29 with
the lane B change of `docs/LANEB.md` section 11 (together +0.873 [+0.104, +1.708]) and the board went from 271.783
to 272.2175. Rank LIRF flight-number coverage is 0.454 against 0.973 on the holdout (`docs/LANEB.md` section 6), so
the head meets rows it has no history for; which of the two v29 changes lost is not measured.

**Caveat.** The stage learns on OOF predictions from models fitted on 8 months and is checked on holdout predictions
from models fitted on 10. The rank predictions come from models fitted on all 12, so there the stage meets a slightly
stronger blend than anywhere it was checked. It moves lane A further there: 42.9 s rms (mean +4.45 s) on the rank set
against 35.7 s rms (mean +3.11 s) on the holdout.
