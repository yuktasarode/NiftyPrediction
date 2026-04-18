from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.features import build_features
from app.labeling import create_dip_labels


def test_rebound_label_logic_flags_expected_case() -> None:
    idx = pd.bdate_range("2023-01-02", periods=120)
    close = np.full(120, 100.0)
    close[40:70] = np.linspace(100, 88, 30)   # drawdown
    close[70:95] = np.linspace(88, 98, 25)    # rebound
    close[95:] = 100

    close_s = pd.Series(close, index=idx)
    open_s = close_s.shift(1).fillna(close_s.iloc[0])
    high = np.maximum(open_s, close_s) + 1
    low = np.minimum(open_s, close_s) - 1

    df = pd.DataFrame(
        {
            "Open": open_s,
            "High": high,
            "Low": low,
            "Close": close_s,
            "Adj Close": close_s,
            "Volume": 1_000_000,
        },
        index=idx,
    )

    cfg = AppConfig(
        drawdown_20_threshold=0.05,
        drawdown_60_threshold=0.08,
        rebound_horizon_days=10,
        rebound_threshold=0.03,
    )

    feat = build_features(df, cfg)
    labels = create_dip_labels(feat, cfg)

    assert labels.dropna().sum() > 0
    assert set(labels.dropna().unique()).issubset({0.0, 1.0})


def test_rebound_label_rejects_deeper_future_crash() -> None:
    idx = pd.bdate_range("2023-01-02", periods=120)
    close = np.full(120, 100.0)
    close[70] = 90.0   # dip day candidate
    close[71] = 95.0   # quick rebound
    close[72] = 80.0   # deeper crash within horizon
    close[73:85] = np.linspace(82, 98, 12)
    close[85:] = 100.0

    close_s = pd.Series(close, index=idx)
    open_s = close_s.shift(1).fillna(close_s.iloc[0])
    high = np.maximum(open_s, close_s) + 1
    low = np.minimum(open_s, close_s) - 1

    df = pd.DataFrame(
        {
            "Open": open_s,
            "High": high,
            "Low": low,
            "Close": close_s,
            "Adj Close": close_s,
            "Volume": 1_000_000,
        },
        index=idx,
    )

    strict_cfg = AppConfig(
        drawdown_20_threshold=0.05,
        drawdown_60_threshold=0.08,
        rebound_horizon_days=5,
        rebound_threshold=0.03,
        max_rebound_wait_days=3,
        max_additional_drawdown=0.02,  # strict adverse-move filter
    )
    relaxed_cfg = AppConfig(
        drawdown_20_threshold=0.05,
        drawdown_60_threshold=0.08,
        rebound_horizon_days=5,
        rebound_threshold=0.03,
        max_rebound_wait_days=3,
        max_additional_drawdown=0.20,
    )

    feat = build_features(df, strict_cfg)
    labels_strict = create_dip_labels(feat, strict_cfg)
    labels_relaxed = create_dip_labels(feat, relaxed_cfg)

    target_day = idx[70]
    assert labels_strict.loc[target_day] == 0.0
    assert labels_relaxed.loc[target_day] == 1.0
