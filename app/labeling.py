"""Dip-zone labeling and baseline rule engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig


def _future_window_stats(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Build future max/min from next N trading days only."""
    future_cols = [close.shift(-i) for i in range(1, horizon + 1)]
    future_frame = pd.concat(future_cols, axis=1)
    return future_frame.max(axis=1), future_frame.min(axis=1)


def _days_to_threshold(close: pd.Series, horizon: int, rebound_threshold: float) -> pd.Series:
    """
    Earliest day index (1..horizon) when forward return crosses rebound threshold.
    Returns NaN if threshold is not reached.
    """
    future_rets = pd.concat([(close.shift(-i) / close - 1) for i in range(1, horizon + 1)], axis=1)
    future_rets.columns = list(range(1, horizon + 1))
    reached = future_rets >= rebound_threshold
    return reached.idxmax(axis=1).where(reached.any(axis=1), np.nan)


def create_dip_labels(feat_df: pd.DataFrame, config: AppConfig) -> pd.Series:
    """Create target labels without leaking future information into features."""
    close = feat_df["Close"]
    dd20_abs = -feat_df["drawdown_20"]
    dd60_abs = -feat_df["drawdown_60"]
    dip_now = (dd20_abs >= config.drawdown_20_threshold) | (dd60_abs >= config.drawdown_60_threshold)

    horizon = config.rebound_horizon_days

    future_max, future_min = _future_window_stats(close, horizon)
    rebound_max = future_max / close - 1
    future_worst = future_min / close - 1
    days_to_rebound = _days_to_threshold(close, horizon, config.rebound_threshold)
    quick_enough = days_to_rebound <= config.max_rebound_wait_days
    no_deeper_crash = future_worst >= -config.max_additional_drawdown

    if config.label_mode == "local_bottom":
        rebound = future_max / close - 1
        near_bottom = close <= (future_min * (1 + config.local_bottom_tolerance))
        valid = future_max.notna() & future_min.notna()
        label_bool = dip_now & near_bottom & (rebound >= config.rebound_threshold) & quick_enough
    else:
        valid = rebound_max.notna() & future_worst.notna()
        label_bool = dip_now & (rebound_max >= config.rebound_threshold) & no_deeper_crash & quick_enough

    labels = pd.Series(np.where(valid, label_bool.astype(int), np.nan), index=feat_df.index, name="label")
    return labels


def compute_rule_score(row: pd.Series, config: AppConfig) -> tuple[float, int, str]:
    """Simple heuristic score for dip detection."""
    rules = [
        ((-row.get("drawdown_20", np.nan) >= config.drawdown_20_threshold) or (-row.get("drawdown_60", np.nan) >= config.drawdown_60_threshold)),
        (row.get("rsi", np.nan) <= config.rsi_threshold),
        (row.get("Close", np.nan) <= row.get("bb_lower", np.nan) * (1 + config.bollinger_touch_buffer)),
        (row.get("ret_5d", np.nan) < 0),
    ]

    hit_count = int(sum(bool(x) for x in rules if pd.notna(x)))
    score = hit_count / len(rules)

    if score >= config.rule_dip_zone_threshold:
        signal = "DIP ZONE"
    elif score >= config.rule_watch_threshold:
        signal = "DIP WATCH"
    else:
        signal = "NO DIP"

    return score, hit_count, signal


def apply_rule_engine(feat_df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Apply baseline rules for all rows."""
    records = []
    for _, row in feat_df.iterrows():
        score, hits, signal = compute_rule_score(row, config)
        records.append({"rule_score": score, "rule_hits": hits, "rule_signal": signal})

    return pd.DataFrame(records, index=feat_df.index)
