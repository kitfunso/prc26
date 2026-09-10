"""CEIL: the best board score this campaign can still reach, and where each tier of it ranks (diagnostic, never a verdict).

Every gain below was measured by an earlier run and is quoted from that run's own document. Nothing here is fitted, declared or
promoted; it converts what is already known into the one unit the prize is decided in.
"""
from __future__ import annotations

import json

from typing import NamedTuple

BOARD = "data/board.json"
# Every scored file, KEYED, because six unlabelled pairs put v15 after v16 and made `ANCHORS[1]` a puzzle. The v15/v16 pair
# moves lane A alone, so PER_A is measured off it (CAL3) rather than differentiated out of the pooled slope; v17 joins the line
# but cannot calibrate PER_A (ledger S7 says why). A file with no board result of its own does not belong here.
SCORED = {"S1": (365.24, 325.3982), "v16": (321.1269, 296.2745), "v15": (322.5796, 298.0671),
          "v17": (321.1716, 296.2490), "v18": (319.6704, 294.9917), "v19": (319.9404, 295.5028),
          "v20": (320.4961, 295.2957)}
ANCHORS = tuple(SCORED.values())
BEST = min(SCORED, key=lambda k: SCORED[k][1])  # the file the board ranks us on, and the one `taxi.pb` must price against
LANE_N, LANE_RMSE = (339046, 5373), (230.0951, 1808.1729)  # v16's own lanes: the point the per-lane rates differentiate at


class Pair(NamedTuple):
    """One CONTROLLED lane-A pair: two scored files whose lane B is byte-identical, so the board delta is lane A alone."""

    label: str
    parent: str
    child: str
    base: str  # the lane `parent` ships on lane A, named as `taxi.ms1.lanes` names it
    chal: str
    hold: float  # holdout lane-A seconds, measured off the cached per-row vectors and quoted from `docs/MS1.md`

    @property
    def board(self) -> float:
        return SCORED[self.parent][1] - SCORED[self.child][1]

    @property
    def carry(self) -> float:
        return self.board / self.hold


# Both halves used to be typed. The v17/v20 row read 0.9585 because its 229.20 was never measured anywhere: it was SB3's
# 230.0951 minus the one-decimal "+0.9 s" `docs/LT2.md` published, and the measured blend is 229.1635. Board deltas are now
# COMPUTED off SCORED and holdout gains quoted from `docs/MS1.md`, which reads them off the vectors; the test greps that table.
PAIRS = (Pair("v15/v16, MC2 to SB3", "v15", "v16", "MC2", "SB3", 2.0551),
         Pair("v17/v18, SB3+RC1 to SB3+NN3", "v17", "v18", "SB3+RC1", "SB3+NN3", 2.1328),
         Pair("v17/v19, SB3+RC1 to SB3+NN2", "v17", "v19", "SB3+RC1", "SB3+NN2", 1.7437),
         Pair("v17/v20, SB3+RC1 to SB3+LT2", "v17", "v20", "SB3+RC1", "SB3+LT2", 0.9949))
BY_CHILD = {p.child: p for p in PAIRS}
CAL3 = PAIRS[0]  # the first controlled pair, and the one PER_A is still measured off


def line() -> tuple[float, float]:
    """Intercept and slope least-squares through every scored anchor."""
    n = len(ANCHORS)
    mx, my = sum(x for x, _ in ANCHORS) / n, sum(y for _, y in ANCHORS) / n
    sxy = sum((x - mx) * (y - my) for x, y in ANCHORS)
    sxx = sum((x - mx) ** 2 for x, _ in ANCHORS)
    b = sxy / sxx
    return my - b * mx, b


def per_lane() -> tuple[float, float]:
    """Lane A is MEASURED off the controlled CAL3 pair; lane B is still differentiated out of the pooled slope.

    The pooled slope predicted 0.4657 for lane A and the controlled pair returned 0.8723, so the derivation understates by
    1.87x. Lane B has no controlled pair, so its rate carries that same unquantified error and is a floor, not a number.
    """
    _, b = line()
    n, comp = sum(LANE_N), SCORED["v16"][0]
    return CAL3.carry, b * LANE_N[1] * LANE_RMSE[1] / (n * comp)


V16 = SCORED["v16"][1]  # MEASURED, not projected: the bucket's merry-octopus_v16.parquet_result.json
V17 = SCORED["v17"][1]  # v16 stays the base every lane-A gain stacks on, so V17 is kept for the files that already cite it
V18 = SCORED["v18"][1]  # our BEST scored file since 2026-09-10, and the one the board ranks us on
V19 = SCORED["v19"][1]  # NN2: better than v20 on the holdout, worse on the board. The inversion the page above is about
PER_A, PER_B = per_lane()

# (tier, name, lane, holdout seconds, source, quote, what it rests on). QUOTE is the verbatim string the source doc must still
# contain: `tests/test_ceil_sources.py` greps for it, so a re-run of any source that moves its number fails instead of leaving
# this hand-copied table quietly two lanes stale, which is how the BF row sat at +0.36 after BF itself had published +1.88.
ITEMS = [
    ("built", "the searched net blended onto SB3 at its out-of-fold weight", "A", 2.07, "docs/SW1.md",
     "| SB3+NN3 | 228.03 | +2.07 |",
     "BUILT and measured on the untouched holdout, weight 0.5189 fitted out of fold on 1,723,531 rows. It clears the 2 s "
     "floor at +2.07 s with a lower bound of +1.1 s, and NC1 retired the k = 40 bound that was held against it. The floor "
     "still sits INSIDE its interval, so the margin is 0.07 s and the decision is Keith's"),
    ("reachable", "stand and runway geometry, the share that survives the calendar", "A", 0.597, "docs/GEO.md",
     "**+0.597 s of lane A**",
     "capped at +3.295 s with the congestion interaction; 26% of it survives, and no lane is written"),
    ("reachable", "a better fit over the 49 columns already read", "A", 0.10, "docs/EXH.md",
     "worth about 0.10 s of lane A",
     "what an exhaustive search of the existing feature set could still find"),
    ("oracle", "every other cached lane-A model stacked on top of that blend", "A", 0.59, "docs/BF.md",
     "adds +0.59 s on top of the best pair",
     "BF's own increment over the best pair, so the net above is counted once; the stack weights are read off the holdout's "
     "own answers, so no out-of-fold stack reaches even this"),
    ("oracle", "perfect copy membership at the nine untouched airports", "B", 25.4, "docs/LBC.md",
     "| near copies, the nine untouched airports | 746 | 3.18% | +25.4 |",
     "the real classifier for exactly this was run as L11 (H51) and returned -5.7 s"),
    ("oracle", "a perfect call on the six day-shifted rows", "B", 510.7, "docs/DE.md",
     "**+510.7 s**",
     "the label is not visible in 2026 and every blind rule tried has lost"),
]
CLOSED = [("lane-B level removal, no model change", 1.17, "docs/BL.md section 2",
           "v16 runs 208.6 s low on the average 2026 lane-B row, but that level was measured by probes P1 and P3s/P5s. "
           "`docs/RULES.md` quotes the ranking page naming attempts to learn from the ranking process unfair, and the same "
           "objection that closes F38 closes this. Real, measured, and not ours to spend.")]
TIERS = {"built": "run, measured on the untouched holdout, and held back only by the promotion floor",
         "reachable": "measured headroom with a route, none of it yet run",
         "oracle": "ceilings with the answers in hand, no route to any of them"}


def rank(score: float, board: list[dict]) -> int:
    return 1 + sum(1 for t in board if t["score"] < score)


def main(out: str = "docs/CEIL.md") -> None:
    b = json.load(open(BOARD, encoding="utf-8"))
    board, us, lead = b["board"], b["us"], b["leader"]
    gap = V16 - lead["score"]

    rows, running, tiers_at = [], V16, {}
    for tier, name, lane, sec, src, _quote, note in ITEMS:
        gain = sec if lane == "board" else sec * (PER_A if lane == "A" else PER_B)
        running -= gain
        tiers_at[tier] = running
        rows.append(f"| {tier} | {name} | {lane} | {'quoted as board' if lane == 'board' else format(sec, ',.2f')} | "
                    f"{gain:.2f} | {running:.2f} | {rank(running, board):d} | `{src}` |")

    tier_rows = []
    at = V16
    for t, desc in TIERS.items():
        at = tiers_at[t]
        tier_rows.append(f"| {t} ({desc}) | {V16 - at:.2f} | {at:.2f} | {rank(at, board)} | "
                         f"{'yes' if at < lead['score'] else 'no'} |")

    near = [t for t in board if abs(t["score"] - V16) < 12][:14]
    ladder = [f"| {t['rank']} | {t['team']} | {t['score']:.4f} | {t['subs']} |" for t in near]

    txt = ["# CEIL: the best board score still reachable, and where it ranks", "",
           f"Regenerate with `python -m taxi.ceil`. Arithmetic only: no fit, no cell declared, no verdict. Leaderboard snapshot "
           f"`{BOARD}` read {b['fetched_utc']} UTC, {b['submissions']} submissions across {b['teams']} scored teams. We are rank "
           f"{us['rank']} at {us['score']:.4f} on {us['subs']} submissions; the leader is {lead['score']:.4f}. Our candidate file v16 "
           f"was uploaded and scored: **{V16:.4f}**, rank **{rank(V16, board)}**, {gap:.2f} s behind the lead. That is a MEASURED "
           f"number, so nothing on this page rests on a projection band any more. It also killed the old one: v16 was projected "
           f"at 286.12 across a 281.4 to 288.0 family and scored {V16:.4f}, above the whole band, because every board number in "
           f"this repo was fitted from the single S1 anchor. {len(ANCHORS)} scored anchors now fix the line, board = {line()[0]:.4f} + "
           f"{line()[1]:.6f} x holdout composite, and the per-lane rates below are differentiated from it at v16 instead of "
           f"assumed: one holdout second of lane A is {PER_A:.4f} s of board, one of lane B is {PER_B:.4f} s. The campaign "
           f"assumed 0.633 and 0.0776, so every gain priced before today was overstated by about 27%.", "",
           "## The holdout and the ranking set disagree about which model is better", "",
           f"{len(PAIRS)} CONTROLLED lane-A pairs have now been scored: two files whose lane B is byte-identical, so the board "
           f"delta is lane A alone and both ends are exact arithmetic on fixed rows, with no measurement noise in the ratio.", "",
           "| pair | holdout s of lane A | board s | implied rate |", "|---|---|---|---|",
           *[f"| {p.label} | {p.hold:.4f} | {p.board:.4f} | {p.carry:.4f} |" for p in PAIRS], "",
           f"The rate spans {min(p.carry for p in PAIRS):.2f} to {max(p.carry for p in PAIRS):.2f}, and the three files "
           f"uploaded on 2026-09-10 are the sharp version of what that means. v19 beats v20 on the holdout by "
           f"{BY_CHILD['v19'].hold - BY_CHILD['v20'].hold:.2f} s of lane A and LOSES to it on the board by "
           f"{V19 - SCORED['v20'][1]:.2f} s, {V19:.4f} against {SCORED['v20'][1]:.4f}. That is an "
           f"ORDER inversion, not a scale error, and no single rate can produce it: a rate rescales differences, it cannot "
           f"reverse their sign. So a projection of the form board = V17 - gain x PER_A is not merely imprecise for a model it "
           f"has not seen, it can rank two models the wrong way round, which is the one job this page is asked to do.", "",
           f"Everything below still multiplies by PER_A = {PER_A:.4f} and is now a DIAGNOSTIC in the protocol's sense: it cannot "
           f"decide which model to build or upload. Two measured rows say why. The searched net projected "
           f"{V16 - 2.07 * PER_A:.2f} and scored {V18:.4f}. Its own sibling NN2 projected 295.46 and scored {V19:.4f}, close on "
           f"the point and wrong on the rank.", "",
           f"The cause is PARTLY identified and `docs/MS1.md` holds the measurement. Splitting each pair's holdout gain by "
           f"month ranks the {len(PAIRS)} controlled pairs perfectly: the lane that put the least of its gain in January, 11%, "
           f"kept the least on the board, {min(p.carry for p in PAIRS):.2f}, and the lane that put the most in January, 43%, "
           f"kept {max(p.carry for p in PAIRS):.2f}, with no exception across all four (rank correlation +1.00, exact "
           f"permutation p = 0.042). The magnitudes do not follow: two lanes that split at 35% and 33% carried at "
           f"{BY_CHILD['v16'].carry:.2f} and {BY_CHILD['v18'].carry:.2f}. So month share is a proxy for the mechanism and not "
           f"the mechanism, and four points against a hypothesis written after the inversion was seen is a tie-break rule, not "
           f"a model. It is enough to prefer a candidate whose gain spreads across the holdout year over one that concentrates "
           f"it in July.", "",
           "## Every measured gain, applied in order", "",
           "| tier | gain | lane | holdout s | board s | board after | rank | source |",
           "|---|---|---|---|---|---|---|---|", *rows, "",
           "## What each tier buys", "",
           "| tier | board s | board after | rank | clears the lead |", "|---|---|---|---|---|", *tier_rows, "",
           "## The board around us", "",
           "| rank | team | score | submissions |", "|---|---|---|---|", *ladder, "",
           "## Measured, real, and closed on the rules", "",
           "| gain | board s | source |", "|---|---|---|",
           *[f"| {n} | {g:.2f} | `{src}` |" for n, g, src, _ in CLOSED], "",
           *[note for _, _, _, note in CLOSED], "",
           "## Reading", "",
           f"The first row is not headroom. It is a file that could be written today: SB3 with NN3 blended in at the weight "
           f"fitted out of fold takes v16 from {V16:.2f} to {tiers_at['built']:.2f}, {V16 - tiers_at['built']:.2f} s of board and "
           f"rank {rank(V16, board)} to {rank(tiers_at['built'], board)}. State the blocker exactly, because the campaign has "
           f"stated it loosely: at +2.07 s the lane CLEARS the 2 s floor. It is held by the k = 40 optimism bound that "
           f"`docs/NN3.md` adds on top, for a hurdle of 3.0 s.", "",
           "That bound is now retired, by measurement rather than by argument. `docs/NC1.md` took NN3's winning cell as a "
           "FIXED k = 1 spec and read it on months 4, 5, 9 and 10, which `taxi.nn3.score` never ranked a cell on: against one "
           "base the blend gains +3.80 s on the months that chose the cell and +3.32 s on the months that did not, so the "
           "selection advantage is about 0.48 s, not the 1.0 s the bound assumed. The holdout never took part in choosing the "
           "cell at all, so the +2.07 needed no such correction in the first place. `docs/SW1.md` closes the other "
           "multiplicity, the campaign meeting one holdout 60 times: the lane survives Romano-Wolf step-down at 5% "
           "family-wise error over the eight cached lanes.", "",
           "So every objection this campaign has instrumented is now answered, and `taxi.verdict` passes the lane on its own "
           "rule: +2.07 s against a 2 s floor with a lower bound of +1.1 s. Say the caveat as loudly as the result. The "
           "margin is 0.07 s, the floor sits INSIDE the interval [+1.1, +2.9], so these rows cannot rule out a true gain "
           "below the floor, and the 1.81 s of board above is a projection off the measured per-lane rate, not a scored "
           "file. This is a Stage 6 promotion decision and it is Keith's alone. Nothing here builds a file or uploads one.", "",
           f"Spending the reachable rows on top of it reaches {tiers_at['reachable']:.2f}, rank "
           f"{rank(tiers_at['reachable'], board)}; every oracle as well reaches {tiers_at['oracle']:.2f}, and that column includes "
           f"two that a real model has already been built for and lost. The lead is {gap:.2f} s away, so first place is inside "
           f"the oracle column and outside every route this campaign has found. `docs/BL.md` section 5 states it from the other "
           f"end: first place on lane A alone needs holdout lane A at {LANE_RMSE[0] - gap / PER_A:.1f} s against SB3's "
           f"{LANE_RMSE[0]:.2f}, a gain of {gap / PER_A:.1f} s, against "
           f"{sum(x[3] for x in ITEMS if x[2] == 'A'):.2f} s of lane-A seconds on this whole page.", "",
           f"The honest reading is unchanged in direction and smaller in size than it was. The remaining distance is not in the "
           f"tuning and not in the features: `docs/COVERAGE.md` records that lane A's target is `AOBT_3_flt - BLOCK_TIME`, two "
           f"clocks disagreeing rather than taxi physics, which is why GEO and EXH both cap under a second. Closing a "
           f"{gap:.0f} s gap needs a different reading of the "
           f"target, not a better fit to this one.", ""]
    text = "\n".join(txt)
    open(out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
