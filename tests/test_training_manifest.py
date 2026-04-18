from __future__ import annotations

import pandas as pd

from app.config import AppConfig
from app.reporting import build_training_manifest, save_training_manifest


def test_training_manifest_files_created(tmp_path) -> None:
    cfg = AppConfig()
    idx = pd.bdate_range("2026-01-01", periods=20)
    feat = pd.DataFrame(
        {
            "Close": range(20),
            "label": [0] * 15 + [1] * 5,
        },
        index=idx,
    )
    bundle = {
        "feature_columns": ["f1", "f2"],
        "metadata": {
            "model_type": "ensemble",
            "train_rows": 20,
            "decision_threshold": 0.5,
            "purge_gap_days": 15,
        },
    }
    avg = {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.44,
        "balanced_accuracy": 0.6,
        "roc_auc": 0.7,
        "pr_auc": 0.3,
        "brier": 0.12,
        "train_f1": 0.55,
        "f1_gap": 0.11,
    }

    manifest = build_training_manifest(feat, bundle, avg, cfg.as_dict())
    hist = tmp_path / "model_training_history.csv"
    latest = tmp_path / "latest_training_manifest.json"
    save_training_manifest(manifest, hist, latest)

    assert hist.exists()
    assert latest.exists()

    hdf = pd.read_csv(hist)
    assert len(hdf) == 1
    assert hdf.loc[0, "ModelType"] == "ensemble"
