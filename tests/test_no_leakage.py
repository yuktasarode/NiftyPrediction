from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.features import build_features


def test_features_do_not_use_future_data() -> None:
    idx = pd.bdate_range("2024-01-01", periods=180)
    rng = np.random.default_rng(21)
    close = pd.Series(20000 + np.cumsum(rng.normal(0, 40, size=len(idx))), index=idx)

    df = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 20,
            "Low": close - 20,
            "Close": close,
            "Adj Close": close,
            "Volume": 10_000_000,
        },
        index=idx,
    )

    cfg = AppConfig()
    feat_1 = build_features(df, cfg)

    cutoff = idx[120]
    df_changed = df.copy()
    df_changed.loc[idx[idx > cutoff], "Close"] += 5000  # modify only future values
    df_changed.loc[idx[idx > cutoff], "Adj Close"] = df_changed.loc[idx[idx > cutoff], "Close"]

    feat_2 = build_features(df_changed, cfg)

    cols_to_check = ["ret_1d", "ret_5d", "ma_20", "drawdown_20", "rsi", "macd", "bb_lower"]
    pd.testing.assert_frame_equal(
        feat_1.loc[idx[idx <= cutoff], cols_to_check],
        feat_2.loc[idx[idx <= cutoff], cols_to_check],
        check_exact=False,
        atol=1e-10,
        rtol=1e-10,
    )
