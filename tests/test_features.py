from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.features import build_features, feature_columns


def make_sample_ohlcv(n: int = 220) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    base = 17000 + np.cumsum(np.random.default_rng(7).normal(0, 35, size=n))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) + np.random.default_rng(8).normal(0, 20, size=n)
    high = np.maximum(open_, close) + np.abs(np.random.default_rng(9).normal(15, 8, size=n))
    low = np.minimum(open_, close) - np.abs(np.random.default_rng(10).normal(15, 8, size=n))
    volume = np.random.default_rng(11).integers(10_000_000, 90_000_000, size=n)

    return pd.DataFrame(
        {
            "Open": open_.values,
            "High": high.values,
            "Low": low.values,
            "Close": close.values,
            "Adj Close": close.values,
            "Volume": volume,
        },
        index=idx,
    )


def test_feature_creation_has_required_columns() -> None:
    cfg = AppConfig()
    df = make_sample_ohlcv()
    feat = build_features(df, cfg)

    cols = feature_columns()
    missing = [c for c in cols if c not in feat.columns]
    assert not missing, f"Missing feature columns: {missing}"

    valid = feat[cols].dropna()
    assert len(valid) > 50
