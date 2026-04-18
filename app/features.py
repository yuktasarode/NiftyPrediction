"""Feature engineering for dip-zone classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def build_features(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Build all model features using only current/past values."""
    feat = df.copy()
    close = feat["Close"]

    # Returns
    feat["ret_1d"] = close.pct_change(1)
    for n in [3, 5, 10, 20]:
        feat[f"ret_{n}d"] = close.pct_change(n)

    # Moving averages + distances
    for w in [5, 10, 20, 50]:
        feat[f"ma_{w}"] = close.rolling(w).mean()
    feat["dist_ma20"] = close / feat["ma_20"] - 1
    feat["dist_ma50"] = close / feat["ma_50"] - 1

    # Rolling volatility
    for w in [5, 10, 20]:
        feat[f"vol_{w}"] = feat["ret_1d"].rolling(w).std()

    # Drawdown from rolling highs
    feat["roll_high_20"] = close.rolling(20).max()
    feat["roll_high_60"] = close.rolling(60).max()
    feat["drawdown_20"] = close / feat["roll_high_20"] - 1
    feat["drawdown_60"] = close / feat["roll_high_60"] - 1

    # Rolling min/max
    feat["roll_min_10"] = close.rolling(10).min()
    feat["roll_max_10"] = close.rolling(10).max()
    feat["roll_min_20"] = close.rolling(20).min()
    feat["roll_max_20"] = close.rolling(20).max()

    # RSI, MACD, Bollinger
    feat["rsi"] = _rsi(close, period=config.rsi_period)
    feat["macd"], feat["macd_signal"] = _macd(close)
    feat["bb_mid"] = close.rolling(config.bollinger_window).mean()
    bb_std = close.rolling(config.bollinger_window).std()
    feat["bb_upper"] = feat["bb_mid"] + config.bollinger_std * bb_std
    feat["bb_lower"] = feat["bb_mid"] - config.bollinger_std * bb_std
    feat["bb_dist_lower"] = close / feat["bb_lower"] - 1

    # Candle features
    body_raw = feat["Close"] - feat["Open"]
    total_range_raw = feat["High"] - feat["Low"]

    feat["candle_body"] = body_raw / feat["Open"].replace(0, np.nan)
    feat["candle_range"] = total_range_raw / feat["Open"].replace(0, np.nan)
    feat["upper_wick"] = (feat["High"] - feat[["Open", "Close"]].max(axis=1)) / feat["Open"].replace(0, np.nan)
    feat["lower_wick"] = (feat[["Open", "Close"]].min(axis=1) - feat["Low"]) / feat["Open"].replace(0, np.nan)
    feat["bullish_candle"] = (feat["Close"] > feat["Open"]).astype(int)
    feat["bearish_candle"] = (feat["Close"] < feat["Open"]).astype(int)

    abs_body = body_raw.abs()
    feat["hammer_like"] = (
        (feat["lower_wick"] > 2 * (abs_body / feat["Open"].replace(0, np.nan)))
        & (feat["upper_wick"] < (abs_body / feat["Open"].replace(0, np.nan)))
    ).astype(int)

    feat["doji_like"] = (
        abs_body <= (0.1 * total_range_raw.replace(0, np.nan))
    ).astype(int)

    feat["close_above_ma50"] = (feat["Close"] > feat["ma_50"]).astype(int)

    return feat


def feature_columns() -> list[str]:
    """Explicit model feature set."""
    return [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_50",
        "dist_ma20",
        "dist_ma50",
        "vol_5",
        "vol_10",
        "vol_20",
        "drawdown_20",
        "drawdown_60",
        "roll_min_10",
        "roll_max_10",
        "roll_min_20",
        "roll_max_20",
        "rsi",
        "macd",
        "macd_signal",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_dist_lower",
        "candle_body",
        "candle_range",
        "upper_wick",
        "lower_wick",
        "bullish_candle",
        "bearish_candle",
        "hammer_like",
        "doji_like",
        "close_above_ma50",
    ]
