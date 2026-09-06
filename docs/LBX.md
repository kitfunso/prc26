# LBX: is lane B out of signal, or was it never fit properly?

Regenerate with `python -m taxi.lbx`. Diagnostic: no cell declared, no champion moves, no verdict decided. One fixed
LightGBM cell throughout (31 leaves, min_data 100, lr 0.05, 400 rounds), so nothing here is a search.

## 1. What the two lost tree attempts shared

Lane B has 17,097 training and 5,373 holdout rows. Rows above 43,200 s are DE's artifact band, a block time
written one day early.

| frame | rows | artifact rows | share of the raw sum of squares |
|---|---|---|---|
| training | 17,097 | 20 | 52.5% |
| holdout | 5,373 | 13 | 67.2% |

20 of 17,097 training rows, 0.12% of them, carry 52.5% of the squared loss a tree minimises. H12 (raw y,
-312.1 s) and L10 (residual label, -757 s) both fit that loss, so both spent their splits on twenty rows.

## 2. Lane B's columns. B4 and every mixture since fit 2 of the 18 populated numeric columns

`month`, `t_off`, `hour`, `wday`, `z_anchor`, `dep_before`, `dep_after`, `arr_anchor`, `arr_off`, `push_anchor`, `occ`, `hol`, `hol_prev`, `hol_next`, `rq`, `rec_taxi` are populated on every lane-B row and no lane-B regressor has read them.
Airport enters the champion as a separate per-airport fit rather than a column, so the trees below carry `apt`, `airline` as
categoricals to give them the same information.

## 3. The question: on ordinary rows, does a tree beat two columns of OLS?

| model | all holdout rows | ordinary rows | artifact rows |
|---|---|---|---|
| L8c (champion, 2 columns) | 1,808.2 | 1037.2 | 30,129 |
| LightGBM, B4 columns, fit on all rows | 3,448.3 | 1944.6 | 57,924 |
| LightGBM, B4 columns, fit on ordinary rows only | 3,735.9 | 1881.2 | 65,646 |
| LightGBM, B4 columns + airport, fit on all rows | 2,345.2 | 1462.2 | 37,307 |
| LightGBM, B4 columns + airport, fit on ordinary rows only | 2,833.9 | 1323.1 | 50,964 |
| LightGBM, all 18 columns, fit on all rows | 3,243.2 | 1823.8 | 54,550 |
| LightGBM, all 18 columns, fit on ordinary rows only | 3,607.4 | 1782.5 | 63,785 |
| LightGBM, all 18 + airport, airline, fit on all rows | 2,567.7 | 1365.4 | 44,231 |
| LightGBM, all 18 + airport, airline, fit on ordinary rows only | 2,934.5 | 1258.9 | 53,904 |
| LightGBM on y minus z_sched, all 18 + airport, airline, fit on all rows (L10's design) | 2,717.3 | 1952.9 | 38,460 |
| LightGBM on y minus z_sched, all 18 + airport, airline, fit on ordinary rows only | 3,063.3 | 2001.0 | 47,197 |

Paired day-block bootstrap against the champion on the 5,360 ordinary rows (positive is better than the champion):

| challenger | delta (s) | 95% interval | |
|---|---|---|---|
| LightGBM, B4 columns, fit on all rows | -907.5 | [-1111.6, -728.1] | lose |
| LightGBM, B4 columns, fit on ordinary rows only | -844.1 | [-1162.6, -569.4] | lose |
| LightGBM, B4 columns + airport, fit on all rows | -425.0 | [-556.0, -281.6] | lose |
| LightGBM, B4 columns + airport, fit on ordinary rows only | -286.0 | [-378.4, -194.9] | lose |
| LightGBM, all 18 columns, fit on all rows | -786.7 | [-970.3, -606.8] | lose |
| LightGBM, all 18 columns, fit on ordinary rows only | -745.3 | [-1040.9, -489.6] | lose |
| LightGBM, all 18 + airport, airline, fit on all rows | -328.3 | [-458.4, -201.8] | lose |
| LightGBM, all 18 + airport, airline, fit on ordinary rows only | -221.8 | [-367.1, -101.6] | lose |
| LightGBM on y minus z_sched, all 18 + airport, airline, fit on all rows (L10's design) | -915.8 | [-1329.3, -537.0] | lose |
| LightGBM on y minus z_sched, all 18 + airport, airline, fit on ordinary rows only | -963.9 | [-1406.5, -511.7] | lose |

## 4. Answer: lane B is out of signal for this model class, not mis-fit

Every one of the 10 specifications loses on the ordinary rows, the closest by -221.8 s with an interval clear of zero. Three
things were tried against the two lost attempts and none of them turned the sign. Removing the artifact rows from the fit
helps but nowhere near enough. Sixteen unread columns help a little. The airport, which the champion carries as a separate
per-airport fit and both earlier trees carried as a categorical, is worth about 500 s on its own and was the largest single
term here, so a global tree without it is not a fair comparison and the first draft of this table was wrong. And the residual
label, which hands the tree the copy branch's identity for free, is the worst row on the page, confirming L10 independently
rather than explaining it away.

The champion's shape is doing the work: a per-airport linear fit can emit `y = z_sched` on a copied row exactly, and a tree
emits piecewise constants over a variable that ranges across thousands of seconds. That is a representation gap, not a
tuning gap, and no amount of columns or rounds closes it.

Consequence for the campaign: with F25 closing lane A's 49 columns and F21 closing the artifact rows, the 6.9% of board
squared error that separates the projected v15 from the leader is not sitting in a lane-B model class either. Do not spend
a declared lane on CatBoost or TabPFN in lane B on the strength of their lane-A wins.

