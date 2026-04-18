"""Model training, validation, and persistence."""

from __future__ import annotations

from statistics import mean

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import AppConfig
from .features import feature_columns


def _logistic_model(config: AppConfig):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.35,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _random_forest_model(config: AppConfig):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=12,
        min_samples_split=30,
        max_features="sqrt",
        random_state=config.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _extra_trees_model(config: AppConfig):
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        random_state=config.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _hist_gb_model(config: AppConfig):
    return HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=400,
        learning_rate=0.04,
        min_samples_leaf=30,
        l2_regularization=0.8,
        random_state=config.random_state,
    )


def _build_estimator(config: AppConfig):
    if config.model_type == "random_forest":
        return _random_forest_model(config)
    if config.model_type == "extra_trees":
        return _extra_trees_model(config)
    if config.model_type == "hist_gb":
        return _hist_gb_model(config)
    if config.model_type == "ensemble":
        return VotingClassifier(
            estimators=[
                ("logit", _logistic_model(config)),
                ("rf", _random_forest_model(config)),
                ("et", _extra_trees_model(config)),
                ("hgb", _hist_gb_model(config)),
            ],
            voting="soft",
            weights=[2, 2, 2, 1],
            n_jobs=-1,
        )
    return _logistic_model(config)


def _prepare_training_data(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = feature_columns()
    required = cols + ["label"]
    work = feat_df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = work[cols]
    y = work["label"].astype(int)
    return X, y, cols


def _fold_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
    }
    if y_true.nunique() > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
        m["pr_auc"] = average_precision_score(y_true, y_prob)
        m["brier"] = brier_score_loss(y_true, y_prob)
    return m


def train_and_evaluate(feat_df: pd.DataFrame, config: AppConfig) -> tuple[dict, list[dict], dict]:
    """Train classifier with purged time-series CV and final fit on all labeled data."""
    X, y, cols = _prepare_training_data(feat_df)

    if y.nunique() < 2:
        raise RuntimeError("Insufficient class variation in labels. Adjust labeling thresholds.")

    splitter = TimeSeriesSplit(n_splits=config.cv_splits)
    fold_metrics: list[dict] = []
    purge_gap = max(1, int(config.rebound_horizon_days))

    for fold_id, (train_idx_raw, test_idx) in enumerate(splitter.split(X), start=1):
        # Purge the tail of the training set to avoid overlap with test-period forward-label windows.
        first_test_idx = int(test_idx[0])
        train_idx = train_idx_raw[train_idx_raw < (first_test_idx - purge_gap)]
        if len(train_idx) < 100:
            continue

        model = _build_estimator(config)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2:
            continue

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = (y_prob_test >= config.decision_threshold).astype(int)
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_pred_train = (y_prob_train >= config.decision_threshold).astype(int)
        else:
            y_pred_test = model.predict(X_test).astype(int)
            y_prob_test = y_pred_test.astype(float)
            y_pred_train = model.predict(X_train).astype(int)
            y_prob_train = y_pred_train.astype(float)

        test_m = _fold_metrics(y_test, y_pred_test, y_prob_test)
        train_m = _fold_metrics(y_train, y_pred_train, y_prob_train)
        metrics = {
            "fold": fold_id,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "test_positive_rate": float(y_test.mean()),
            "threshold": config.decision_threshold,
            **{k: v for k, v in test_m.items()},
            "train_f1": train_m["f1"],
            "train_pr_auc": train_m["pr_auc"],
            "f1_gap": float(train_m["f1"] - test_m["f1"]),
        }
        fold_metrics.append(metrics)

    if not fold_metrics:
        raise RuntimeError("TimeSeries CV did not produce valid folds. Increase data or tune config.")

    avg_metrics = {
        "precision": mean(m["precision"] for m in fold_metrics),
        "recall": mean(m["recall"] for m in fold_metrics),
        "f1": mean(m["f1"] for m in fold_metrics),
        "balanced_accuracy": mean(m["balanced_accuracy"] for m in fold_metrics),
        "roc_auc": float(np.nanmean([m["roc_auc"] for m in fold_metrics])),
        "pr_auc": float(np.nanmean([m["pr_auc"] for m in fold_metrics])),
        "brier": float(np.nanmean([m["brier"] for m in fold_metrics])),
        "train_f1": mean(m["train_f1"] for m in fold_metrics),
        "f1_gap": mean(m["f1_gap"] for m in fold_metrics),
        "decision_threshold": config.decision_threshold,
        "purge_gap_days": purge_gap,
    }

    final_model = _build_estimator(config)
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "feature_columns": cols,
        "metadata": {
            "model_type": config.model_type,
            "train_rows": int(len(X)),
            "positive_rate": float(y.mean()),
            "decision_threshold": float(config.decision_threshold),
            "purge_gap_days": purge_gap,
        },
    }
    return bundle, fold_metrics, avg_metrics


def save_model_bundle(bundle: dict, path) -> None:
    """Persist trained model bundle."""
    joblib.dump(bundle, path)


def load_model_bundle(path) -> dict:
    """Load persisted model bundle."""
    return joblib.load(path)
