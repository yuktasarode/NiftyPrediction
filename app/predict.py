"""Prediction helpers for latest and historical probabilities."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .config import AppConfig
from .labeling import compute_rule_score
from .utils import next_business_day


def _ml_signal(prob: float, config: AppConfig) -> str:
    if prob >= config.ml_dip_zone_threshold:
        return "DIP ZONE"
    if prob >= config.ml_watch_threshold:
        return "DIP WATCH"
    return "NO DIP"


def _combine_signals(ml_signal: str, rule_signal: str) -> str:
    if "DIP ZONE" in (ml_signal, rule_signal):
        return "DIP ZONE"
    if "DIP WATCH" in (ml_signal, rule_signal):
        return "DIP WATCH"
    return "NO DIP"


def predict_latest(feat_df: pd.DataFrame, model_bundle: dict, config: AppConfig) -> dict:
    """Generate next-trading-day dip probability and recommendation."""
    model = model_bundle["model"]
    cols = model_bundle["feature_columns"]

    latest = feat_df.dropna(subset=cols).iloc[-1]
    X_latest = latest[cols].to_frame().T

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_latest)[0, 1])
    else:
        pred = int(model.predict(X_latest)[0])
        prob = float(pred)

    rule_score, _, rule_signal = compute_rule_score(latest, config)
    ml_signal = _ml_signal(prob, config)
    final_signal = _combine_signals(ml_signal, rule_signal)

    return {
        "RunDate": latest.name.strftime("%Y-%m-%d"),
        "TargetDate": next_business_day(latest.name.date()).strftime("%Y-%m-%d"),
        "NIFTY Close": float(latest["Close"]),
        "Dip Probability": prob,
        "Rule-Based Score": float(rule_score),
        "ML Signal": ml_signal,
        "Rule Signal": rule_signal,
        "Signal": final_signal,
    }


def historical_probabilities(feat_df: pd.DataFrame, model_bundle: dict) -> pd.Series:
    """Predict probabilities over all rows where features are available."""
    model = model_bundle["model"]
    cols = model_bundle["feature_columns"]

    valid = feat_df.dropna(subset=cols)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(valid[cols])[:, 1]
    else:
        probs = model.predict(valid[cols]).astype(float)

    return pd.Series(probs, index=valid.index, name="dip_probability")


def prediction_to_text(pred: dict) -> str:
    """Human-readable summary text."""
    return (
        f"Run Date: {pred['RunDate']}\n"
        f"Target Date: {pred['TargetDate']}\n"
        f"NIFTY Close: {pred['NIFTY Close']:.2f}\n"
        f"Dip Probability: {pred['Dip Probability']:.3f}\n"
        f"Rule-Based Score: {pred['Rule-Based Score']:.3f}\n"
        f"ML Signal: {pred['ML Signal']}\n"
        f"Rule Signal: {pred['Rule Signal']}\n"
        f"Signal: {pred['Signal']}\n"
    )


def predicted_class_from_signal(signal: str) -> int:
    """Map recommendation to binary class for evaluation logging."""
    return int(signal in {"DIP WATCH", "DIP ZONE"})


def quick_actual_from_features(feat_df: pd.DataFrame, target_dt: date, config: AppConfig) -> int | None:
    """
    Quick next-day realized outcome (available at target day close):
    1 if target-day close-to-close return <= threshold, else 0.
    """
    ts = pd.Timestamp(target_dt)
    if ts not in feat_df.index:
        return None
    ret_1d = feat_df.loc[ts, "ret_1d"]
    if pd.isna(ret_1d):
        return None
    return int(float(ret_1d) <= config.quick_eval_drop_threshold)


def final_actual_from_labels(labels: pd.Series, target_dt: date) -> int | None:
    """Final dip-zone truth once label matures (may be unavailable for recent days)."""
    ts = pd.Timestamp(target_dt)
    if ts not in labels.index:
        return None
    val = labels.loc[ts]
    if pd.isna(val):
        return None
    return int(val)
