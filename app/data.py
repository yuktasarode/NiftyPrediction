"""Data ingestion and local persistence for NIFTY 50."""

from __future__ import annotations

from datetime import timedelta
import warnings

import pandas as pd
import yfinance as yf

from .config import AppConfig
from .utils import market_data_cutoff_date


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output columns and date index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    missing = [col for col in PRICE_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = pd.NA

    df = df[PRICE_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates and missing values safely."""
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=["Close"])
    for col in ["Open", "High", "Low", "Adj Close"]:
        df[col] = df[col].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df


def load_local_data(config: AppConfig) -> pd.DataFrame:
    """Load saved raw CSV if available."""
    if not config.raw_data_path.exists():
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = pd.read_csv(config.raw_data_path, parse_dates=["Date"], index_col="Date")
    return _clean_ohlcv(df)


def _download_data(config: AppConfig, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Download daily data from Yahoo Finance."""
    data = yf.download(
        config.ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    return _clean_ohlcv(_standardize_columns(data))


def update_nifty_data(config: AppConfig, refresh: bool | None = None) -> pd.DataFrame:
    """Refresh local CSV incrementally or fully."""
    use_refresh = config.refresh_data if refresh is None else refresh
    local = load_local_data(config)
    cutoff_date = market_data_cutoff_date(
        tz_name=config.market_timezone,
        close_hour=config.market_close_hour,
        close_minute=config.market_close_minute,
    )

    try:
        if use_refresh:
            end_exclusive = (cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d")
            merged = _download_data(config, config.start_date, end_exclusive)
            if merged.empty:
                raise RuntimeError(
                    f"No data downloaded for ticker {config.ticker} from {config.start_date} to {cutoff_date}."
                )
        elif local.empty:
            end_exclusive = (cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d")
            merged = _download_data(config, config.start_date, end_exclusive)
            if merged.empty:
                raise RuntimeError(
                    f"No data downloaded for ticker {config.ticker} from {config.start_date} to {cutoff_date}."
                )
        else:
            last_date = local.index.max()
            if last_date.date() >= cutoff_date:
                merged = local.copy()
            else:
                incremental_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                end_exclusive = (cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d")
                latest = _download_data(config, incremental_start, end_exclusive)
                if latest.empty:
                    merged = local.copy()
                else:
                    merged = pd.concat([local, latest]).sort_index()
                    merged = merged[~merged.index.duplicated(keep="last")]
    except Exception as exc:
        if local.empty:
            raise
        warnings.warn(
            f"Data download failed ({exc}). Falling back to existing local data at {config.raw_data_path}.",
            RuntimeWarning,
        )
        merged = local.copy()

    merged = _clean_ohlcv(merged)
    merged.to_csv(config.raw_data_path, index=True)
    return merged
