"""Lane-A MLP on the residual c = y - z_aobt, no tail/body split, as a blend partner for the tree models (v1's NN3 cell).
Lane B rows come from a donor. `python nn.py cv NAME DONOR` / `python nn.py fit NAME DONOR`."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import CACHE, rmse
from features import ARR, CAT, DAY, NUM, OCC, PW, WX, WZ, build, with_pw
from lanea import _merge_b
from train import FOLDS, HOLDOUT, RESULTS, report

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAL_FRAC = 0.10
CELL = dict(hidden=(1024, 1024, 1024), dropout=0.2555, lr=1.5e-3, weight_decay=1.65e-4, batch=1024, max_epochs=150, t_max=60,
            patience=5, emb_cap=16, huber=0.405, seed=0)
CONFIGS: dict[str, dict] = {
    "nn3": dict(cell=CELL, cat=CAT, num=NUM + DAY + OCC + WX),
    "nn3_s1": dict(cell={**CELL, "seed": 1}, cat=CAT, num=NUM + DAY + OCC + WX),
    "nn3_arr": dict(cell=CELL, cat=CAT, num=NUM + DAY + OCC + WX + ARR),
    "nn3_wz": dict(cell=CELL, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ),
    "nn3_pw": dict(cell=CELL, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW),
}


class Enc:
    """Fitted on the fit rows only; index 0 is the unseen or missing level, so a new stand in the ranking file is not a crash."""

    def __init__(self, fit: pd.DataFrame, cats: list[str], nums: list[str]) -> None:
        self.cats, self.nums = cats, nums
        self.maps = {c: {v: i + 1 for i, v in enumerate(pd.unique(fit[c].dropna().astype(object)))} for c in cats}
        self.card = {c: len(self.maps[c]) + 1 for c in cats}
        x = fit[nums].astype("float32")
        self.mu, self.sd = x.mean(), x.std().replace(0.0, 1.0).fillna(1.0)
        self.miss = [c for c in nums if x[c].isna().any()]

    def __call__(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        ci = np.stack([df[c].astype(object).map(self.maps[c]).fillna(0).to_numpy(np.int64) for c in self.cats], 1)
        x = df[self.nums].astype("float32")
        z = ((x - self.mu) / self.sd).to_numpy(np.float32)
        ind = x[self.miss].isna().to_numpy(np.float32) if self.miss else np.zeros((len(df), 0), np.float32)
        return torch.from_numpy(ci), torch.from_numpy(np.nan_to_num(np.hstack([z, ind]), nan=0.0))


class Net(nn.Module):
    def __init__(self, enc: Enc, n_num: int, cell: dict) -> None:
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(enc.card[k], int(min(cell["emb_cap"], max(4, round(enc.card[k] ** 0.5))))) for k in enc.cats])
        d, layers = sum(e.embedding_dim for e in self.emb) + n_num, []
        for h in cell["hidden"]:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(cell["dropout"])]
            d = h
        self.mlp, self.head = nn.Sequential(*layers), nn.Linear(d, 1)

    def forward(self, ci: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(torch.cat([m(ci[:, i]) for i, m in enumerate(self.emb)] + [xn], 1))).squeeze(1)


def _batches(n: int, size: int, g: torch.Generator | None = None):
    idx = torch.randperm(n, generator=g) if g is not None else torch.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


@torch.no_grad()
def _infer(model: Net, ci: torch.Tensor, xn: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.cat([model(ci[b].to(DEV), xn[b].to(DEV)).cpu() for b in _batches(len(ci), 16384)]).numpy()


def train(cfg: dict, fit: pd.DataFrame) -> tuple[Net, Enc, float, float, int]:
    """Early stopping on the last 10% of fit days, so the validation window stays forward in time."""
    cell = cfg["cell"]
    torch.manual_seed(cell["seed"])
    days = np.sort(fit["day"].unique()); cut = days[int(len(days) * (1 - VAL_FRAC))]
    tr, va = fit[fit["day"].values < cut], fit[fit["day"].values >= cut]
    enc = Enc(tr, cfg["cat"], cfg["num"])
    ci, xn = enc(tr); vci, vxn = enc(va)
    c = (tr["y"] - tr["z_aobt"]).values
    mu, sd = float(c.mean()), float(c.std())
    y = torch.from_numpy(((c - mu) / sd).astype(np.float32))
    vy = (va["y"] - va["z_aobt"]).values
    model = Net(enc, xn.shape[1], cell).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cell["lr"], weight_decay=cell["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cell["t_max"])
    g = torch.Generator().manual_seed(cell["seed"])
    best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(1, cell["max_epochs"] + 1):
        model.train()
        for b in _batches(len(ci), cell["batch"], g):
            opt.zero_grad(set_to_none=True)
            nn.functional.huber_loss(model(ci[b].to(DEV), xn[b].to(DEV)), y[b].to(DEV), delta=cell["huber"]).backward()
            opt.step()
        sched.step()
        v = rmse(vy, _infer(model, vci, vxn) * sd + mu)
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cell["patience"]:
                break
    model.load_state_dict(state)
    return model, enc, mu, sd, ep


def predict(cfg: dict, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, int]:
    model, enc, mu, sd, ep = train(cfg, tr)
    ci, xn = enc(te)
    return _infer(model, ci, xn) * sd + mu + te["z_aobt"].values, ep


def cv(name: str, donor: str) -> None:
    cfg, f = CONFIGS[name], build()
    if set(PW) & set(cfg["num"]):
        f = with_pw(f)
    f = f[f["split"] == "train"]
    tr, ho = f[~f["month"].isin(HOLDOUT)], f[f["month"].isin(HOLDOUT)]
    ta, ha = tr[tr["has_plan"] == 1], ho[ho["has_plan"] == 1]
    oof, eps, t0 = np.full(len(ta), np.nan), [], time.time()
    for i, months in enumerate(FOLDS):
        va = ta["month"].isin(months).values
        p, ep = predict(cfg, ta[~va], ta[va])
        oof[va] = p; eps.append(ep)
        print(f"fold {i} {months}: {rmse(ta['y'].values[va], p):.2f} at {ep} epochs, {time.time() - t0:.0f}s", flush=True)
    ph, ep = predict(cfg, ta, ha); ph = np.maximum(ph, 0)
    o = _merge_b(pd.DataFrame({"id": ta["id"].values, "oof": oof}), pd.read_parquet(CACHE / f"{donor}_oof.parquet"), tr, "oof")
    h = _merge_b(pd.DataFrame({"id": ha["id"].values, "p": ph}), pd.read_parquet(CACHE / f"{donor}_holdout.parquet"), ho, "p")
    o.to_parquet(CACHE / f"{name}_oof.parquet"); h.to_parquet(CACHE / f"{name}_holdout.parquet")
    o, h = tr.merge(o, on="id"), ho.merge(h, on="id")
    line = f"| {name} (B from {donor}) | {json.dumps(report(o['y'].values, o['oof'].values, o))} | {json.dumps(report(h['y'].values, h['p'].values, h))} | epochs {eps}+{ep} | {time.time() - t0:.0f}s |\n"
    RESULTS.open("a", encoding="utf-8").write(line)
    print(line)


def fit(name: str, donor: str) -> None:
    cfg, f = CONFIGS[name], build()
    if set(PW) & set(cfg["num"]):
        f = with_pw(f)
    tr, rk = f[(f["split"] == "train") & (f["has_plan"] == 1)], f[f["split"] == "rank"]
    ra = rk[rk["has_plan"] == 1]
    p = np.maximum(predict(cfg, tr, ra)[0], 0)
    d = pd.read_parquet(f"data/{donor}_rank.parquet")
    b = d[d["MVT_ID_mvt"].isin(rk["id"][rk["has_plan"] == 0])]
    out = pd.concat([pd.DataFrame({"MVT_ID_mvt": ra["id"].values, "TAXITIME_SEC_mvt": p}), b], ignore_index=True)
    out.to_parquet(f"data/{name}_rank.parquet")
    print(f"data/{name}_rank.parquet", len(out), float(out["TAXITIME_SEC_mvt"].mean()))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2], sys.argv[3]), "fit": lambda: fit(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
