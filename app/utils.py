"""Utility helpers for IO and formatting."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import AppConfig


def ensure_directories(*paths: Path) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    """Save dict as pretty JSON."""
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)


def append_prediction_log(path: Path, record: dict) -> pd.DataFrame:
    """Append current prediction record to CSV log (deduplicated by Date)."""
    new_df = pd.DataFrame([record])

    if path.exists():
        old_df = pd.read_csv(path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    if "RunDate" not in merged.columns:
        merged["RunDate"] = pd.NA
    if "TargetDate" not in merged.columns:
        merged["TargetDate"] = pd.NA
    if "Date" in merged.columns:
        merged["RunDate"] = merged["RunDate"].fillna(merged["Date"])
        merged["TargetDate"] = merged["TargetDate"].fillna(merged["Date"])
        merged = merged.drop(columns=["Date"])

    key_col = "TargetDate"
    merged = (
        merged.dropna(subset=[key_col])
        .drop_duplicates(subset=[key_col], keep="last")
        .sort_values(key_col)
    )
    merged.to_csv(path, index=False)
    return merged


def save_text(path: Path, text: str) -> None:
    """Write plain text content."""
    with path.open("w", encoding="utf-8") as fp:
        fp.write(text)


def next_business_day(d: date) -> date:
    """Return next Monday-Friday day."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def previous_business_day(d: date) -> date:
    """Return previous Monday-Friday day."""
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def market_data_cutoff_date(
    tz_name: str,
    close_hour: int,
    close_minute: int,
    now_dt: datetime | None = None,
) -> date:
    """
    Return the latest market date expected to be available in daily data.

    - Before market close -> previous business day
    - After market close -> today (or previous business day if weekend)
    """
    tz = ZoneInfo(tz_name)
    now_local = (now_dt or datetime.now(tz)).astimezone(tz)
    today = now_local.date()
    close_dt = datetime.combine(today, time(close_hour, close_minute), tzinfo=tz)

    if today.weekday() >= 5:
        return previous_business_day(today)
    if now_local >= close_dt:
        return today
    return previous_business_day(today)


def update_live_accuracy_log(
    log_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    labels: pd.Series,
    config: AppConfig,
) -> pd.DataFrame:
    """Backfill quick/final realized outcomes for previous target dates."""
    if log_df.empty or "TargetDate" not in log_df.columns:
        return log_df

    out = log_df.copy()
    for col in [
        "PredClass",
        "QuickActual",
        "QuickError",
        "QuickCorrect",
        "FinalActual",
        "FinalError",
        "FinalCorrect",
    ]:
        if col not in out.columns:
            out[col] = pd.NA

    for idx, row in out.iterrows():
        target = pd.to_datetime(row["TargetDate"]).date()
        pred_class = 1 if str(row.get("Signal", "")) in {"DIP WATCH", "DIP ZONE"} else 0
        out.at[idx, "PredClass"] = pred_class

        ts = pd.Timestamp(target)

        if ts in feat_df.index:
            ret_1d = feat_df.loc[ts, "ret_1d"]
        else:
            ret_1d = pd.NA
        if pd.notna(ret_1d):
            quick_actual = int(float(ret_1d) <= config.quick_eval_drop_threshold)
            quick_err = abs(pred_class - int(quick_actual))
            out.at[idx, "QuickActual"] = int(quick_actual)
            out.at[idx, "QuickError"] = quick_err
            out.at[idx, "QuickCorrect"] = int(quick_err == 0)

        if ts in labels.index:
            lbl = labels.loc[ts]
        else:
            lbl = pd.NA
        if pd.notna(lbl):
            final_actual = int(lbl)
            final_err = abs(pred_class - int(final_actual))
            out.at[idx, "FinalActual"] = int(final_actual)
            out.at[idx, "FinalError"] = final_err
            out.at[idx, "FinalCorrect"] = int(final_err == 0)

    return out
