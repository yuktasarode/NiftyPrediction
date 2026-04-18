from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.config import AppConfig
from app.utils import market_data_cutoff_date, update_live_accuracy_log


def test_market_cutoff_before_and_after_close() -> None:
    cfg = AppConfig()

    before_close = datetime.fromisoformat("2026-04-20T09:00:00+05:30")
    after_close = datetime.fromisoformat("2026-04-20T16:00:00+05:30")

    assert str(market_data_cutoff_date(cfg.market_timezone, cfg.market_close_hour, cfg.market_close_minute, before_close)) == "2026-04-17"
    assert str(market_data_cutoff_date(cfg.market_timezone, cfg.market_close_hour, cfg.market_close_minute, after_close)) == "2026-04-20"


def test_update_live_accuracy_log_populates_quick_and_final() -> None:
    cfg = AppConfig(quick_eval_drop_threshold=-0.001)

    idx = pd.to_datetime(["2026-04-17", "2026-04-20"]) 
    feat = pd.DataFrame({"ret_1d": [0.002, -0.01]}, index=idx)
    labels = pd.Series([0.0, 1.0], index=idx, name="label")

    log_df = pd.DataFrame(
        [
            {
                "RunDate": "2026-04-17",
                "TargetDate": "2026-04-20",
                "Signal": "DIP ZONE",
            }
        ]
    )

    out = update_live_accuracy_log(log_df, feat, labels, cfg)
    row = out.iloc[0]

    assert int(row["PredClass"]) == 1
    assert int(row["QuickActual"]) == 1
    assert int(row["QuickCorrect"]) == 1
    assert int(row["FinalActual"]) == 1
    assert int(row["FinalCorrect"]) == 1
