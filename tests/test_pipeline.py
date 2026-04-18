from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.features import build_features
from app.labeling import create_dip_labels
from app.model import train_and_evaluate
from app.predict import predict_latest


def test_prediction_pipeline_runs() -> None:
    idx = pd.bdate_range("2021-01-01", periods=420)
    rng = np.random.default_rng(42)
    close = pd.Series(15000 + np.cumsum(rng.normal(0, 45, len(idx))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 30,
            "Low": np.minimum(open_, close) - 30,
            "Close": close,
            "Adj Close": close,
            "Volume": 20_000_000,
        },
        index=idx,
    )

    cfg = AppConfig(
        rebound_horizon_days=10,
        cv_splits=4,
        model_type="logistic",
        drawdown_20_threshold=0.01,
        drawdown_60_threshold=0.015,
        rebound_threshold=0.005,
    )

    feat = build_features(df, cfg)
    feat["label"] = create_dip_labels(feat, cfg)

    bundle, fold_metrics, avg_metrics = train_and_evaluate(feat, cfg)
    pred = predict_latest(feat, bundle, cfg)

    assert fold_metrics
    assert 0 <= avg_metrics["f1"] <= 1
    assert "pr_auc" in avg_metrics
    assert "balanced_accuracy" in avg_metrics
    assert "brier" in avg_metrics
    assert 0 <= pred["Dip Probability"] <= 1
    assert pred["Signal"] in {"NO DIP", "DIP WATCH", "DIP ZONE"}
