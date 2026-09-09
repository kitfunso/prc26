"""NN3 reference: NN2's incumbent cell scored on NN3's own three folds. Diagnostic, never a verdict, and NOT a search cell.

NN3's declared grid cannot express the incumbent (max_epochs 60 and patience 15 are off its ladders), so without this the search
has no reading against the cell it is trying to beat, only against SB3 at the end. Run after `python -m taxi.nn3`.
"""
from __future__ import annotations

from taxi.champ import COLS_D1, lane_a
from taxi.nn1 import CELL, base_oof
from taxi.nn3 import FOLDS, score

REF = {**CELL, "patience": 15, "cols": COLS_D1}  # NN2 = NN1's cell with patience 15; everything else untouched


def main(out: str = "docs/NN3-REF.md") -> None:
    tra, _ = lane_a()
    blend, level, rho = score(REF, tra, base_oof(tra))
    text = "\n".join([
        "# NN3-REF: the incumbent NN cell on NN3's own instrument (diagnostic, never a verdict)", "",
        f"Regenerate with `python -m taxi.nn3ref`. NN2's cell, one seed, the same three folds {FOLDS} and the same pooled "
        "out-of-fold blend against MC2 that every NN3 trial is scored on. It is the reading NN3's trial table needs to be "
        "interpretable: the declared grid cannot express this cell, so it is measured beside the search rather than inside it, "
        "and it does not count toward k.", "",
        "| cell | blend RMSE | net alone | agreement |", "|---|---|---|---|",
        f"| NN2 incumbent | {blend:.2f} | {level:.2f} | {rho:.4f} |", "",
        "Any NN3 trial under this blend RMSE is a cell the search found and the campaign did not have.", ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
