"""NN1: a neural net as a BLEND PARTNER for SB3, the last model family never fit here (protocol section 4, k = 1).

It fits the correction `c` directly, with no tail/body split: the M2 recipe is shared by every family measured so far, and a
blend partner must not share approximation error. Regenerate with `python -m taxi.nn1`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from taxi.champ import COLS_D1, lane_a
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.m1 import CAT
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

CELL = dict(hidden=(512, 256, 128), dropout=0.1, lr=1e-3, weight_decay=1e-4, batch=4096, max_epochs=60, patience=5)
VAL_FRAC = 0.10
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def _cell(cell: dict | None) -> dict:
    """Every knob below reads through here, so NN3 can search one without a second copy of the net."""
    return CELL if cell is None else {**CELL, **cell}


class Enc:
    """Encoders fitted on the fit rows only; index 0 is reserved so an unseen or missing level is a real embedding, not a crash."""

    def __init__(self, fit: pd.DataFrame, cols: list[str] | None = None) -> None:
        cols = COLS_D1 if cols is None else cols
        self.cats = [c for c in CAT if c in cols]
        self.nums = [c for c in cols if c not in self.cats]
        self.maps = {c: {v: i + 1 for i, v in enumerate(pd.unique(fit[c].dropna().astype(object)))} for c in self.cats}
        self.card = {c: len(self.maps[c]) + 1 for c in self.cats}
        x = fit[self.nums].astype("float32")
        self.mu, self.sd = x.mean(), x.std().replace(0.0, 1.0).fillna(1.0)
        self.miss = [c for c in self.nums if x[c].isna().any()]

    def __call__(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        ci = np.stack([df[c].astype(object).map(self.maps[c]).fillna(0).to_numpy(np.int64) for c in self.cats], 1)
        x = df[self.nums].astype("float32")
        z = ((x - self.mu) / self.sd).to_numpy(np.float32)
        ind = x[self.miss].isna().to_numpy(np.float32) if self.miss else np.zeros((len(df), 0), np.float32)
        return torch.from_numpy(ci), torch.from_numpy(np.nan_to_num(np.hstack([z, ind]), nan=0.0))


def emb_dim(card: int, cap: int = 32) -> int:
    return int(min(cap, max(4, round(card ** 0.5))))


class Net(nn.Module):
    def __init__(self, enc: Enc, n_num: int, cell: dict | None = None) -> None:
        super().__init__()
        c = _cell(cell)
        self.emb = nn.ModuleList([nn.Embedding(enc.card[k], emb_dim(enc.card[k], c.get("emb_cap", 32))) for k in enc.cats])
        d = sum(e.embedding_dim for e in self.emb) + n_num
        act, norm = ACTS[c.get("act", "gelu")], c.get("norm", "batch")
        layers: list[nn.Module] = []
        for h in c["hidden"]:
            layers += [nn.Linear(d, h)]
            layers += [nn.BatchNorm1d(h)] if norm == "batch" else ([nn.LayerNorm(h)] if norm == "layer" else [])
            layers += [act(), nn.Dropout(c["dropout"])]
            d = h
        self.mlp, self.head = nn.Sequential(*layers), nn.Linear(d, 1)

    def forward(self, ci: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
        e = torch.cat([m(ci[:, i]) for i, m in enumerate(self.emb)] + [xn], 1)
        return self.head(self.mlp(e)).squeeze(1)


def split(fit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Early stopping needs a validation set the fit never saw; the last days of the fit window keep it forward in time."""
    days = np.sort(fit.day.unique())
    cut = days[int(len(days) * (1 - VAL_FRAC))]
    return fit.day.values < cut, fit.day.values >= cut


def _batches(n: int, size: int, shuffle: bool, g: torch.Generator | None = None):
    idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


@torch.no_grad()
def _infer(model: Net, ci: torch.Tensor, xn: torch.Tensor) -> np.ndarray:
    model.eval()
    out = [model(ci[b].to(DEV), xn[b].to(DEV)).cpu() for b in _batches(len(ci), 16384, False)]
    return torch.cat(out).numpy()


def train(fit: pd.DataFrame, seed: int = SEED, patience: int | None = None,
          cell: dict | None = None) -> tuple[Net, Enc, float, float, int]:
    """seed, patience and cell are arguments so NN2 can bag and NN3 can search; the defaults reproduce NN1 exactly."""
    cf = _cell(cell)
    torch.manual_seed(seed)
    patience = cf["patience"] if patience is None else patience
    tr_m, va_m = split(fit)
    tr, va = fit[tr_m], fit[va_m]
    enc = Enc(tr, cf.get("cols"))
    ci, xn = enc(tr)
    vci, vxn = enc(va)
    mu, sd = float(tr.c.mean()), float(tr.c.std())  # MSE on raw c would put the tail rows' gradients thousands of times above the body's
    y = torch.from_numpy(((tr.c.values - mu) / sd).astype(np.float32))
    vy = va.c.values
    model = Net(enc, xn.shape[1], cf).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cf["lr"], weight_decay=cf["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cf.get("t_max") or cf["max_epochs"])
    lossf = (nn.functional.mse_loss if cf.get("loss", "mse") == "mse" else
             lambda a, b: nn.functional.huber_loss(a, b, delta=cf.get("huber", 1.0)))
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(1, cf["max_epochs"] + 1):
        model.train()
        for b in _batches(len(ci), cf["batch"], True, g):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(ci[b].to(DEV), xn[b].to(DEV)), y[b].to(DEV))
            loss.backward()
            opt.step()
        sched.step()
        v = rmse(_infer(model, vci, vxn) * sd + mu - vy)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, enc, mu, sd, ep


def predict(model: Net, enc: Enc, rows: pd.DataFrame, mu: float, sd: float) -> np.ndarray:
    ci, xn = enc(rows)
    return _infer(model, ci, xn) * sd + mu + rows.z_aobt.values


def blend_weight(nn_p: np.ndarray, base_p: np.ndarray, y: np.ndarray) -> float:
    """Least squares weight on the NN, fitted out of fold. X1 lost 4.2 s by fixing this at 0.5 instead of measuring it."""
    a, b = nn_p - y, base_p - y
    d = a - b
    w = float(-(b @ d) / (d @ d)) if float(d @ d) > 0 else 0.0
    return float(np.clip(w, 0.0, 1.0))


def cv(tra: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold NN predictions on the training months, the only honest surface the blend weight may be read from."""
    oof = np.full(len(tra), np.nan)
    lines = ["| fold | NN1 RMSE (s) | epochs | fit s |", "|---|---|---|---|"]
    for fold in CV_FOLDS:
        m, t0 = tra.month.isin(fold).values, time.time()
        model, enc, mu, sd, ep = train(tra[~m].reset_index(drop=True))
        va = tra[m].reset_index(drop=True)
        oof[m] = predict(model, enc, va, mu, sd)
        lines.append(f"| {fold} | {rmse(oof[m] - va.y.values):.1f} | {ep} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    return oof, lines


def base_oof(tra: pd.DataFrame) -> np.ndarray:
    d = pd.read_parquet("data/cache/mc2_oof.parquet")
    return tra[["id"]].merge(d[["id", "pred"]], on="id", how="left").pred.to_numpy(float)


def main() -> None:
    from taxi.bf import agreement, best_blend
    tra, tea = lane_a()
    t0 = time.time()
    oof, fold_lines = cv(tra)
    base = base_oof(tra)
    ok = np.isfinite(oof) & np.isfinite(base)
    w = blend_weight(oof[ok], base[ok], tra.y.values[ok])
    print(f"OOF rows {int(ok.sum()):,}; weight on NN1 fitted out of fold: {w:.4f}", flush=True)

    model, enc, mu, sd, ep = train(tra)
    nn_te = floored(predict(model, enc, tea, mu, sd))
    sb3_te = sb3_predict(tra, tea)
    y = tea.y.values
    e0, ex = sb3_te - y, nn_te - y
    blend = floored(w * nn_te + (1 - w) * sb3_te)
    v = paired_delta(e0, blend - y, tea.day.values)
    ow, orm, rho = best_blend(e0, ex)
    wall = (time.time() - t0) / 60

    vt, jan, jul = tea.c.values >= tea.c.quantile(0.9), tea.month.values == 1, tea.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.2f} | {rmse(e[jan]):.2f} | {rmse(e[jul]):.2f} |"
    report = [
        "# NN1: a neural net as a blend partner for SB3, Jan+Jul 2025 development holdout", "",
        f"Regenerate with `python -m taxi.nn1`. Declared in protocol section 4 before the code was written; k = 1, one cell, floor 2 s. "
        f"Cell {CELL}, {len(CATS)} embedded categorical columns and {len(NUMS)} numeric on D1's {len(COLS_D1)} columns, "
        f"fitting the correction `c` directly with no tail/body split. Device {DEV}. CV plus the final fit took {wall:.1f} min.", "",
        "## Out-of-fold folds", "", *fold_lines, "",
        f"Weight on NN1, fitted out of fold against MC2's cached out-of-fold predictions on {int(ok.sum()):,} rows: **{w:.4f}**. "
        f"The holdout is never read to choose it.", "",
        f"## Holdout ({len(tea):,} rows)", "",
        "| model | RMSE | Jan | Jul |", "|---|---|---|---|",
        row("SB3 (champion)", e0), row("NN1 alone (diagnostic)", ex), row(f"SB3 + NN1 at w={w:.4f} (challenger)", blend - y), "",
        f"NN1's holdout error agrees **{rho:.4f}** with SB3's (uncentred correlation, `taxi.bf.agreement`), at {rmse(ex):.2f} s "
        f"against SB3's {rmse(e0):.2f} s. The oracle weight, read off the holdout itself and so unavailable in practice, puts "
        f"{ow:.3f} on SB3 and reaches {orm:.2f} s, a gain of {rmse(e0) - orm:+.2f} s.", "",
        verdict_line("SB3 + NN1", "SB3", v),
    ]
    text = "\n".join(report) + "\n"
    with open("docs/NN1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
