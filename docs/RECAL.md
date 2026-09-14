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

**Full file** (session scratchpad `gate_a.py`): with v27's lane B, holdout 307.18 against v27's 308.47, +1.29 [+0.76,
+2.03]; with the copy-history lane B of `docs/LANEB.md` section 6, 306.76, +1.71 [+0.75, +2.88]. The second is v28.

**Caveat.** The stage learns on OOF predictions from models fitted on 8 months and is checked on holdout predictions
from models fitted on 10. The rank predictions come from models fitted on all 12, so there the stage meets a slightly
stronger blend than anywhere it was checked. It moves lane A further there: 42.9 s rms (mean +4.45 s) on the rank set
against 35.7 s rms (mean +3.11 s) on the holdout.
