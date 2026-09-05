"""D1: plan-provenance lane. The M2 recipe on W1-A's columns plus the plan columns (taxi.data.PLAN), judged against W1-A on Jan+Jul."""
from __future__ import annotations

from taxi.champ import COLS_D1, champion_holdout, lane_a
from taxi.data import PLAN, TRAIN_MONTHS
from taxi.m1 import chosen_cells
from taxi.m2 import K, Models, is_tail, predict_m2, stages
from taxi.verdict import paired_delta, rmse
from taxi.w1 import COLS_A, cv_a, verdict_line



def plan_gain(models: Models) -> list[str]:
    lines = ["| column | classifier gain share | tail gain share | body gain share |", "|---|---|---|---|"]
    shares = [b.feature_importance("gain") / b.feature_importance("gain").sum() for b in models]
    names = models[0].feature_name()
    for c in PLAN:
        j = names.index(c)
        lines.append(f"| {c} | {shares[0][j]:.3f} | {shares[1][j]:.3f} | {shares[2][j]:.3f} |")
    return lines


def main() -> None:
    tra, tea = lane_a()
    cell, _ = chosen_cells()["A"]
    rounds, lines, _ = cv_a(tra, cell, cols=COLS_D1, name="D1")
    models = stages(tra, cell, rounds=rounds, cols=COLS_D1)
    e1 = predict_m2(models, tea)[0] - tea.y.values
    e0 = champion_holdout("W1-A", tra, tea) - tea.y.values
    v = paired_delta(e0, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"
    report = ["# D1: plan-provenance lane, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.d1`. Lane A = M2 recipe (cell {cell}) on {len(COLS_D1)} columns: "
              f"W1-A's {len(COLS_A)} plus {list(PLAN)}; k = {K}. Champion W1-A refit at its recorded rounds on the same rows.", "",
              "## Month-pair CV", ""] + lines + ["", f"## Holdout verdict vs champion W1-A ({len(tea)} rows; tail rows {int(vt.sum())})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row("W1-A", e0), row("D1", e1), "",
              verdict_line("D1", "W1-A", v), "", "## Gain share of the plan columns in the final fit", ""] + plan_gain(models)
    text = "\n".join(report) + "\n"
    with open("docs/D1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
