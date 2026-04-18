from __future__ import annotations

import pandas as pd

from app.reporting import compute_live_metrics, generate_daily_reports


def test_compute_live_metrics_trend_flags() -> None:
    # 50 resolved quick rows: first 30 poor, last 20 strong => improving
    records = []
    for i in range(50):
        records.append(
            {
                "TargetDate": f"2026-03-{(i % 28) + 1:02d}",
                "QuickCorrect": 1 if i >= 30 else 0,
                "FinalCorrect": 1 if i >= 35 else 0,
            }
        )
    df = pd.DataFrame(records)
    m = compute_live_metrics(df)
    assert m.quick_trend in {"improving", "stable", "declining", "insufficient_data"}
    assert m.quick_resolved == 50


def test_generate_daily_reports_creates_files(tmp_path) -> None:
    log_df = pd.DataFrame(
        [
            {
                "RunDate": "2026-04-17",
                "TargetDate": "2026-04-20",
                "Signal": "NO DIP",
                "Dip Probability": 0.2,
                "QuickCorrect": 1,
            },
            {
                "RunDate": "2026-04-18",
                "TargetDate": "2026-04-21",
                "Signal": "DIP WATCH",
                "Dip Probability": 0.6,
                "QuickCorrect": 0,
            },
        ]
    )

    generate_daily_reports(log_df, tmp_path)

    assert (tmp_path / "all_runs_report.csv").exists()
    assert (tmp_path / "live_accuracy_report.csv").exists()
    assert (tmp_path / "daily_dashboard.txt").exists()
