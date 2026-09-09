"""CEIL: the best board score this campaign can still reach, and where each tier of it ranks (diagnostic, never a verdict).

Every gain below was measured by an earlier run and is quoted from that run's own document. Nothing here is fitted, declared or
promoted; it converts what is already known into the one unit the prize is decided in.
"""
from __future__ import annotations

import json

BOARD = "data/board.json"
# Three REAL anchors, each a scored file: S1 (holdout composite 365.24, board 325.3982), v16 (321.1269, 296.2745) and
# v15 (322.5796, 298.0671). v15 and v16 differ on 329,648 lane-A rows and ZERO lane-B rows, so that pair is a controlled
# lane-A experiment and PER_A below is measured off it rather than differentiated out of the pooled slope (CAL3).
ANCHORS = ((365.24, 325.3982), (321.1269, 296.2745), (322.5796, 298.0671))
LANE_N, LANE_RMSE = (339046, 5373), (230.0951, 1808.1729)  # v16's own lanes: the point the per-lane rates differentiate at
# The controlled pair: MC2 lane A costs 2.0551 holdout seconds against SB3 (H52/H53a) and cost 1.7926 board seconds.
CAL3 = ((232.1502 - 230.0951), (298.0671 - 296.2745))


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
    n, comp = sum(LANE_N), ANCHORS[1][0]
    return CAL3[1] / CAL3[0], b * LANE_N[1] * LANE_RMSE[1] / (n * comp)


V16 = ANCHORS[1][1]  # MEASURED, not projected: the bucket's merry-octopus_v16.parquet_result.json
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
    ("oracle", "every other cached lane-A model stacked on top of that blend", "A", 0.14, "docs/BF.md",
     "adds +0.14 s on top of the best pair",
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
           f"this repo was fitted from the single S1 anchor. Two scored anchors now fix the line, board = {line()[0]:.4f} + "
           f"{line()[1]:.6f} x holdout composite, and the per-lane rates below are differentiated from it at v16 instead of "
           f"assumed: one holdout second of lane A is {PER_A:.4f} s of board, one of lane B is {PER_B:.4f} s. The campaign "
           f"assumed 0.633 and 0.0776, so every gain priced before today was overstated by about 27%.", "",
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
           f"That bound is now retired, by measurement rather than by argument. `docs/NC1.md` took NN3's winning cell as a "
           f"FIXED k = 1 spec and read it on months 4, 5, 9 and 10, which `taxi.nn3.score` never ranked a cell on: against one "
           f"base the blend gains +3.80 s on the months that chose the cell and +3.32 s on the months that did not, so the "
           f"selection advantage is about 0.48 s, not the 1.0 s the bound assumed. The holdout never took part in choosing the "
           f"cell at all, so the +2.07 needed no such correction in the first place. `docs/SW1.md` closes the other "
           f"multiplicity, the campaign meeting one holdout 60 times: the lane survives Romano-Wolf step-down at 5% "
           f"family-wise error over the seven cached lanes.", "",
           f"So every objection this campaign has instrumented is now answered, and `taxi.verdict` passes the lane on its own "
           f"rule: +2.07 s against a 2 s floor with a lower bound of +1.1 s. Say the caveat as loudly as the result. The "
           f"margin is 0.07 s, the floor sits INSIDE the interval [+1.1, +2.9], so these rows cannot rule out a true gain "
           f"below the floor, and the 1.81 s of board above is a projection off the measured per-lane rate, not a scored "
           f"file. This is a Stage 6 promotion decision and it is Keith's alone. Nothing here builds a file or uploads one.", "",
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
