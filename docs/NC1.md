# NC1: NN3's cell measured on the folds its search never scored

Regenerate with `python -m taxi.nc1`. Declared in protocol section 4 before this file was written. No model is fitted here: NN3's cached five-seed out-of-fold vector and MC2's are read from disk, so the only new quantities are one blend weight and one paired interval. k = 1, one fixed spec, so the hurdle is the 2 s floor alone. This lane promotes nothing whatever it finds.

NN3 ranked its 40 cells on folds ((2, 3), (6, 8), (11, 12)); the weight below is fitted on those months and the verdict is read on ((4, 5), (9, 10)), which no cell was ever ranked on. Weight on NN3, fitted on the search months only: **0.5267** (NN3's own published weight, fitted on all ten months, is 0.5189).

| rows | n | MC2 out-of-fold | blend | gain |
|---|---|---|---|---|
| search months (weight fitted here, NOT a verdict) | 1,000,485 | 215.54 | 211.74 | +3.80 |
| held months (THE VERDICT) | 723,046 | 209.11 | 205.79 | +3.32 |

## Verdict

On the 723,046 rows no cell was ranked on, the blend beats MC2 out of fold by **+3.32 s** [+2.87, +3.82], against a 2 s floor and a 0.0 s optimism bound.

**PASS, and the pre-registered prediction was WRONG.** The declaration predicted the gain would come in below the holdout's +2.07 s and named a real risk of failing this gate. It came in at +3.32 s with the whole interval above the floor. That is recorded here because a prediction which only appears when it is right is not a prediction.

The sharper reading is the pair of rows above, which is what this lane was built to see. Against ONE base, the same MC2 out-of-fold vector, the blend gains +3.80 s on the months that chose the cell and +3.32 s on the months that did not. The selection advantage is about 0.48 s, not the 1.0 s the k = 40 bound assumes, and nowhere near enough to explain a +2.07 s holdout gain away. The bound was a reasonable default and it is larger than the thing it was defending against, now that the thing has been measured.

This retires that objection and nothing else. No file is built and no upload is made here, and whether the 2 s floor is the right constant is untouched. The promotion of SB3+NN3 is a Stage 6 decision and Keith's alone.

## What this cannot say

Whether 2 s is the right floor is a decision-rule question and not measurable from inside the campaign. These rows are training months, so the level is not comparable to a holdout RMSE and only the GAIN transfers. And the base here is MC2 out of fold rather than SB3, which is what every out-of-fold blend in this campaign uses because a five-seed bag has no out-of-fold vector.

