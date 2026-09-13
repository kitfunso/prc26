"""Mechanism study of lane A's physical (class-3) error tail: rows with no known stamp copy where
|y - z_aobt| > 600s. Uses the shipped v25 blend (ra_cb_arr+ra_lgb_arr+nn3_arr) residual r = y - p.
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/phys_tail.py harness|census|tail|mech|cluster|dup|arvt|rank|all`"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from data import CACHE, departures, load_raw, rmse, secs
from features import build
from mix import label
from train import HOLDOUT

pd.set_option("display.width", 250)

NAMES = ("ra_cb_arr", "ra_lgb_arr", "nn3_arr")
TAIL = 600.0
BANDS = [0, 600, 1800, 3600, np.inf]
BAND_LABELS = ["<600", "600-1800", "1800-3600", ">3600"]
CLS_NAMES = {0: "aobt", 1: "eobt", 2: "sched", 3: "physical"}


def _oof_blend(tr: pd.DataFrame) -> pd.DataFrame:
    """blend.py never writes an OOF parquet (holdout only); rebuild its nnls weights in memory to get one."""
    comp = None
    for n in NAMES:
        c = pd.read_parquet(CACHE / f"{n}_oof.parquet").rename(columns={"oof": n})
        comp = c if comp is None else comp.merge(c, on="id")
    t = tr.merge(comp, on="id")
    w, _ = nnls(t[list(NAMES)].values, t["y"].values)
    w = w / w.sum()
    p = np.maximum(t[list(NAMES)].values @ w, 0)
    return pd.DataFrame({"id": t["id"].values, "p": p})


def _load() -> pd.DataFrame:
    """Lane A rows, OOF (reconstructed) + holdout, v25 blend residual r, mix.label class, phys-tail flag."""
    f = build()
    f = f[f["split"] == "train"]
    tr, ho = f[~f["month"].isin(HOLDOUT)], f[f["month"].isin(HOLDOUT)]
    tr = tr.merge(_oof_blend(tr), on="id").assign(part="oof")
    ho = ho.merge(pd.read_parquet(CACHE / "blend_ra_cb_arr+ra_lgb_arr+nn3_arr_holdout.parquet"), on="id").assign(part="holdout")
    d = pd.concat([tr, ho], ignore_index=True)
    d = d[d["has_plan"] == 1].copy()
    d["r"] = d["y"] - d["p"]
    d["c"] = d["y"] - d["z_aobt"]
    d["cls"] = label(d)
    d["tail"] = (d["cls"] == 3) & (d["c"].abs() > TAIL)
    return d


def harness(d: pd.DataFrame) -> None:
    ho, oo = d[d["part"] == "holdout"], d[d["part"] == "oof"]
    print(f"holdout laneA RMSE: {rmse(ho['y'], ho['p']):.2f}  (want 223.65)")
    print(f"OOF laneA RMSE (reconstructed blend): {rmse(oo['y'], oo['p']):.2f}  (want 204.96)")


def census(d: pd.DataFrame) -> None:
    print("=== SSE share by label class ===")
    for part in ("oof", "holdout"):
        p = d[d["part"] == part]
        se, tot = p["r"].values ** 2, (p["r"].values ** 2).sum()
        rows = [{"part": part, "class": nm, "n": int((p["cls"] == k).sum()), "row_share": float((p["cls"] == k).mean()),
                 "sse_share": float(se[(p["cls"] == k).values].sum() / tot)} for k, nm in CLS_NAMES.items()]
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== inside class 3 (physical), by |c| band ===")
    for part in ("oof", "holdout"):
        p = d[d["part"] == part]
        p3 = p[p["cls"] == 3].assign(band=pd.cut(p.loc[p["cls"] == 3, "c"].abs(), BANDS, labels=BAND_LABELS, include_lowest=True))
        se3, se_all = (p3["r"] ** 2).sum(), (p["r"] ** 2).sum()
        g = p3.groupby("band", observed=True)["r"].agg(n="size", sse=lambda s: (s ** 2).sum())
        g["row_share"], g["sse_share_of_phys"], g["sse_share_of_laneA"] = g["n"] / len(p3), g["sse"] / se3, g["sse"] / se_all
        print(f"-- {part} --\n" + g[["n", "row_share", "sse_share_of_phys", "sse_share_of_laneA"]].round(4).to_string())

    print("\n=== composition of the top 1% of rows by r^2 ===")
    for part in ("oof", "holdout"):
        p = d[d["part"] == part]
        top = p[p["r"].abs() >= p["r"].abs().quantile(0.99)]
        comp = top["cls"].map(CLS_NAMES).value_counts(normalize=True)
        print(f"-- {part}: n_top={len(top)} --\n{comp.round(3).to_string()}\nphys-tail share of top1%: {top['tail'].mean():.3f}")


def tail_compare(d: pd.DataFrame) -> None:
    c3 = d[d["cls"] == 3]
    tail, body = c3[c3["tail"]], c3[~c3["tail"]]
    print(f"n phys-tail={len(tail)}, n other class3={len(body)}")

    print("\n=== airport x sign(c), top 10 by n ===")
    for name, g in (("phys-tail", tail), ("other class3", body)):
        t = g.assign(sign=np.sign(g["c"])).groupby(["apt", "sign"], observed=True).size()
        print(f"-- {name} --\n{t.sort_values(ascending=False).head(10).to_string()}")

    print("\n=== hour distribution ===")
    for name, g in (("phys-tail", tail), ("other class3", body)):
        print(f"-- {name} --\n{g['hour'].value_counts(normalize=True).sort_index().round(4).to_string()}")

    se_tot = (tail["r"] ** 2).sum()
    for col in ("stand", "airline"):
        g = tail.groupby(col, observed=True)["r"].agg(n="size", sse=lambda s: (s ** 2).sum())
        g["row_share"], g["sse_share"] = g["n"] / len(tail), g["sse"] / se_tot
        print(f"\n=== top 20 {col} by phys-tail SSE ===")
        print(g.sort_values("sse_share", ascending=False).head(20)[["n", "row_share", "sse_share"]].round(4).to_string())

    print("\n=== marker shares: phys-tail vs other class3 ===")
    print(f"aobt_sec==0: phys-tail {(tail['aobt_sec'] == 0).mean():.4f}, other class3 {(body['aobt_sec'] == 0).mean():.4f}")
    for col in ("aobt_m5", "eobt_m5"):
        print(f"{col}: phys-tail {tail[col].mean():.4f}, other class3 {body[col].mean():.4f}")

    print("\n=== distributions ===")
    for col in ("d_ae", "d_al", "occ", "t_occ", "pa_gap", "pa_missing"):
        print(f"-- {col} --\n" + pd.DataFrame({"phys-tail": tail[col].describe(), "other class3": body[col].describe()}).round(2).to_string())


def which_stamp(d: pd.DataFrame, min_n: int = 20) -> None:
    """Per (apt, runway), median y of the well-behaved class-3 rows: which clock does phys-tail's y track?"""
    c3 = d[d["cls"] == 3]
    body = c3[~c3["tail"]]
    cnt = body.groupby(["apt", "runway"], observed=True).size()
    med = body.groupby(["apt", "runway"], observed=True)["y"].median().rename("med_y").reset_index()
    med = med[med.set_index(["apt", "runway"]).index.map(cnt) >= min_n]

    tail = c3[c3["tail"]].merge(med, on=["apt", "runway"], how="inner")
    y_ok = (tail["y"] - tail["med_y"]).abs() <= 300
    z_ok = (tail["z_aobt"] - tail["med_y"]).abs() <= 300
    print(f"phys-tail rows with a stable (apt,runway) body median (n>={min_n}): {len(tail)} / {c3['tail'].sum()}")
    print(f"share y within 300s of median, z_aobt not: {(y_ok & ~z_ok).mean():.4f}  (off-block stamp looks wrong)")
    print(f"share z_aobt within 300s of median, y not:  {(z_ok & ~y_ok).mean():.4f}  (genuinely long/short taxi)")
    print(f"share both within 300s: {(y_ok & z_ok).mean():.4f}, share neither: {(~y_ok & ~z_ok).mean():.4f}")


def _dispersion(d: pd.DataFrame, group_cols: list[str], parent_cols: list[str], min_n: int = 20) -> tuple[float, int]:
    tab = d.groupby(group_cols, observed=True)["tail"].agg(p_g="mean", n_g="size").reset_index()
    tab = tab[tab["n_g"] >= min_n]
    parent = d.groupby(parent_cols, observed=True)["tail"].mean().rename("p_parent").reset_index()
    tab = tab.merge(parent, on=parent_cols, how="left")
    num = ((tab["p_g"] - tab["p_parent"]) ** 2).sum()
    den = (tab["p_parent"] * (1 - tab["p_parent"]) / tab["n_g"]).sum()
    return float(num / den), len(tab)


def cluster(d: pd.DataFrame) -> None:
    r, n = _dispersion(d, ["apt", "day"], ["apt"])
    print(f"airport-day (parent=airport): dispersion ratio={r:.2f}, n_groups(n>=20)={n}")
    r, n = _dispersion(d, ["apt", "day", "hour"], ["apt", "day"])
    print(f"airport-day-hour (parent=airport-day): dispersion ratio={r:.2f}, n_groups(n>=20)={n}")


def duplicates(d: pd.DataFrame) -> None:
    """Another departure, same FLIGHT_mvt (fallback CALLSIGN_flt), same airport, within +-12h."""
    raw = load_raw()
    dep = departures(raw)
    dep = dep[dep["MVT_ID_mvt"].isin(d["id"])].copy()
    dep["key"] = dep["FLIGHT_mvt"].fillna(dep["CALLSIGN_flt"].astype(str))
    dep["t"] = secs(dep["MVT_TIME_UTC_mvt"], pd.Timestamp(0, tz="UTC"))
    dup = np.zeros(len(dep), dtype=bool)
    for _, ix in dep.groupby(["ADEP_mvt", "key"], observed=True).indices.items():
        if len(ix) < 2:
            continue
        ix = np.asarray(ix)
        order = np.argsort(dep["t"].values[ix])
        ts = dep["t"].values[ix][order]
        lo, hi = np.searchsorted(ts, ts - 43200, "left"), np.searchsorted(ts, ts + 43200, "right")
        dup[ix[order]] = (hi - lo) > 1
    dm = pd.DataFrame({"id": dep["MVT_ID_mvt"].values, "dup": dup})
    d = d.merge(dm, on="id", how="left")
    c3 = d[d["cls"] == 3]
    print(f"duplicate-within-12h share, phys-tail: {c3.loc[c3['tail'], 'dup'].mean():.4f}")
    print(f"duplicate-within-12h share, other class3: {c3.loc[~c3['tail'], 'dup'].mean():.4f}")


def arvt3_check(d: pd.DataFrame) -> None:
    """ARVT_3 is measurement-only (never a model feature): loaded fresh from raw, not merged into features.build."""
    raw = load_raw()
    dep = departures(raw)
    dep = dep[dep["MVT_ID_mvt"].isin(d["id"])]
    t3 = pd.DataFrame({"id": dep["MVT_ID_mvt"].values, "t_arvt3": secs(dep["ARVT_3_flt"], pd.Timestamp(0, tz="UTC")).values})
    d = d.merge(t3, on="id", how="left").dropna(subset=["t_arvt3", "dur"])
    gap = d["t_arvt3"] - d["t_off"]
    print(f"corr(ARVT_3 - takeoff, r): {np.corrcoef(gap, d['r'])[0, 1]:.4f}  (n={len(d)})")

    offblock_rec, offblock_aobt = d["t_off"] - d["y"], d["t_off"] - d["z_aobt"]
    leak = ((d["t_arvt3"] - offblock_rec) - d["dur"]).abs() < 60
    safe = ((d["t_arvt3"] - offblock_aobt) - d["dur"]).abs() < 60
    print(f"share ARVT_3 - recorded_offblock == filed dur +-60s: {leak.mean():.4f}  (label-leak pattern if high; never use)")
    print(f"share ARVT_3 - AOBT_3 == filed dur +-60s:            {safe.mean():.4f}")


def rank_marks(d: pd.DataFrame, candidates: dict[str, pd.Series]) -> pd.DataFrame:
    base_rate = d["tail"].mean()
    se_tail = (d.loc[d["tail"], "r"] ** 2).sum()
    rows = []
    for name, mask in candidates.items():
        mask = mask.reindex(d.index, fill_value=False)
        n = int(mask.sum())
        cov = float((d.loc[mask & d["tail"], "r"] ** 2).sum() / se_tail) if n else 0.0
        lift = float(d.loc[mask, "tail"].mean() / base_rate) if n else np.nan
        rows.append({"mark": name, "n_marked": n, "coverage_of_tail_sse": cov, "lift": lift})
    out = pd.DataFrame(rows).sort_values("coverage_of_tail_sse", ascending=False)
    print(out.round(4).to_string(index=False))
    return out


def dayvar_features(f: pd.DataFrame) -> pd.DataFrame:
    """Day- and hour-level std of the AOBT-EOBT/LOBT gaps, visible stamps only (no y): the phys-tail's 18.4x
    day-level overdispersion vs binomial (cluster()) suggests a per-day AOBT anomaly, not per-flight noise."""
    gday = f.groupby(["split", "apt", "day"], observed=True)
    ghour = f.groupby(["split", "apt", "day", "hour"], observed=True)
    return pd.DataFrame({
        "day_dae_std": gday["d_ae"].transform("std").fillna(0),
        "day_dal_std": gday["d_al"].transform("std").fillna(0),
        "h_dae_std": ghour["d_ae"].transform("std").fillna(0),
        "h_dal_std": ghour["d_al"].transform("std").fillna(0),
    }, index=f.index)


def build_rank_candidates(d: pd.DataFrame) -> pd.DataFrame:
    f_all = build()
    f_all = f_all[f_all["split"] == "train"]
    dv = dayvar_features(f_all).assign(id=f_all["id"].values)
    d = d.merge(dv, on="id", how="left")
    cands = {
        "aobt_sec==0 [existing NUM]": d["aobt_sec"] == 0,
        "eobt_m5 [existing NUM]": d["eobt_m5"] == 1,
        "airline==THY [existing CAT]": d["airline"] == "THY",
        "apt==LTFM [existing CAT]": d["apt"] == "LTFM",
        "day_dae_std top decile [NEW]": d["day_dae_std"] >= d["day_dae_std"].quantile(0.9),
        "day_dal_std top decile [NEW]": d["day_dal_std"] >= d["day_dal_std"].quantile(0.9),
        "h_dae_std top decile [NEW]": d["h_dae_std"] >= d["h_dae_std"].quantile(0.9),
        "h_dal_std top decile [NEW]": d["h_dal_std"] >= d["h_dal_std"].quantile(0.9),
    }
    return rank_marks(d, cands)


def main(cmd: str) -> None:
    d = _load()
    steps = {"harness": harness, "census": census, "tail": tail_compare, "mech": which_stamp,
              "cluster": cluster, "dup": duplicates, "arvt": arvt3_check, "rank": build_rank_candidates}
    if cmd == "all":
        for name, fn in steps.items():
            print(f"\n########## {name} ##########")
            fn(d)
        return
    steps[cmd](d)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
