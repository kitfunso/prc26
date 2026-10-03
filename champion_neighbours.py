"""Original neighbour blend fitted only on each downstream permitted scope."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from campaign import file_hash, prediction_windows
from champion import fit_member, ids_hash, write_record
from champion_budget import budget_guard
from champion_predictions import raw_member_predictions
from features import PW, PW_BLEND


def file_month(frame: pd.DataFrame) -> pd.Series:
    return frame["file"].str.slice(14, 16).astype(int)


def month_groups(frame: pd.DataFrame, fit_ids: pd.Index) -> tuple[tuple[int, ...], ...]:
    months = sorted(file_month(frame[frame.id.isin(fit_ids)]).unique())
    if len(months) < 3:
        raise ValueError("champion cross-fitting requires at least three file months")
    return tuple(tuple(int(month) for month in months[i::3]) for i in range(3))


def with_champion_pw(frame: pd.DataFrame, fit_ids: pd.Index, directory: Path) -> tuple[pd.DataFrame, list[dict]]:
    fit_ids = pd.Index(fit_ids)
    if not fit_ids.is_unique or not fit_ids.isin(frame.id).all():
        raise ValueError("PW source IDs must be unique and present")
    fit = frame.id.isin(fit_ids) & (frame.has_plan == 1)
    prediction = pd.Series(0., index=frame.id.values)
    scopes = []
    groups = (*month_groups(frame, fit_ids), None)
    target = directory / ids_hash(fit_ids)
    for group in groups:
        allowed = fit & ~file_month(frame).isin(group) if group is not None else fit
        query = fit & file_month(frame).isin(group) if group is not None else ~fit & (frame.has_plan == 1)
        if not query.any():
            continue
        train, test = frame[allowed], frame[query].assign(y=np.nan)
        tag = "full" if group is None else "group-" + "-".join(map(str, group))
        logging.info("Original PW donor scope %s: fit %s, query %s", tag, len(train), len(test))
        record = {"group": group, "fit_rows": len(train), "query_rows": len(test),
                  "fit_ids_hash": ids_hash(train.id), "query_ids_hash": ids_hash(test.id), "members": {}}
        p = np.zeros(len(test))
        for name, weight in PW_BLEND.items():
            budget_guard(directory)
            member, detail = fit_member(name, train, test, target / tag / name)
            if group is not None:
                member = raw_member_predictions(name, detail["specification"]["config"], test, target / tag / name)
            p += weight * member
            record["members"][name] = {"weight": weight, "runtime_seconds": detail["runtime_seconds"],
                                        "prediction_floor": group is None,
                                        "prediction_hash": file_hash(target / tag / name / "raw_predictions.parquet")
                                                           if group is not None else detail["artifact_hashes"]["predictions.parquet"]}
        prediction.loc[test.id] = np.maximum(p, 0)
        scopes.append(record)
        write_record(target / f"{tag}.json", record)
    windows = prediction_windows(frame, prediction)
    windows.insert(0, "id", frame.id.values)
    target.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(target / "pw.parquet")
    write_record(target / "pw.json", {"fit_ids_hash": ids_hash(fit_ids), "scopes": scopes,
                                      "output_hash": file_hash(target / "pw.parquet"),
                                      "engine_hash": file_hash(Path(__file__)), "weights": PW_BLEND})
    output = frame.drop(columns=PW, errors="ignore").merge(windows, on="id", validate="one_to_one", sort=False)
    output.attrs = frame.attrs.copy()
    return output, scopes
