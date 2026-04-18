"""Reporting utilities for consolidated run history and live accuracy trends."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass
class LiveMetrics:
    total_runs: int
    quick_resolved: int
    quick_accuracy: float | None
    quick_last_20: float | None
    quick_prev_20: float | None
    quick_trend: str
    final_resolved: int
    final_accuracy: float | None
    final_last_20: float | None
    final_prev_20: float | None
    final_trend: str


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.astype(float).mean())


def _trend(last_val: float | None, prev_val: float | None, tol: float = 0.02) -> str:
    if last_val is None or prev_val is None:
        return "insufficient_data"
    diff = last_val - prev_val
    if diff > tol:
        return "improving"
    if diff < -tol:
        return "declining"
    return "stable"


def compute_live_metrics(log_df: pd.DataFrame) -> LiveMetrics:
    """Compute cumulative and rolling-window quick/final accuracy metrics."""
    df = log_df.copy()
    if "TargetDate" in df.columns:
        df = df.sort_values("TargetDate")

    quick = df.dropna(subset=["QuickCorrect"]) if "QuickCorrect" in df.columns else df.iloc[0:0]
    final = df.dropna(subset=["FinalCorrect"]) if "FinalCorrect" in df.columns else df.iloc[0:0]

    quick_acc = _safe_mean(quick["QuickCorrect"]) if not quick.empty else None
    final_acc = _safe_mean(final["FinalCorrect"]) if not final.empty else None

    quick_last_20 = _safe_mean(quick["QuickCorrect"].tail(20)) if not quick.empty else None
    quick_prev_20 = _safe_mean(quick["QuickCorrect"].iloc[-40:-20]) if len(quick) >= 40 else None

    final_last_20 = _safe_mean(final["FinalCorrect"].tail(20)) if not final.empty else None
    final_prev_20 = _safe_mean(final["FinalCorrect"].iloc[-40:-20]) if len(final) >= 40 else None

    return LiveMetrics(
        total_runs=int(len(df)),
        quick_resolved=int(len(quick)),
        quick_accuracy=quick_acc,
        quick_last_20=quick_last_20,
        quick_prev_20=quick_prev_20,
        quick_trend=_trend(quick_last_20, quick_prev_20),
        final_resolved=int(len(final)),
        final_accuracy=final_acc,
        final_last_20=final_last_20,
        final_prev_20=final_prev_20,
        final_trend=_trend(final_last_20, final_prev_20),
    )


def _fmt_pct(v: float | None) -> str:
    return "NA" if v is None else f"{100 * v:.2f}%"


def build_dashboard_text(log_df: pd.DataFrame, metrics: LiveMetrics) -> str:
    """Build plain-text dashboard summary for quick review."""
    lines: list[str] = []
    lines.append("NIFTY Dip-Zone Daily Dashboard")
    lines.append("=" * 32)
    lines.append(f"Total Runs: {metrics.total_runs}")
    lines.append(
        f"Quick Accuracy: {_fmt_pct(metrics.quick_accuracy)} "
        f"(resolved={metrics.quick_resolved}, last20={_fmt_pct(metrics.quick_last_20)}, "
        f"prev20={_fmt_pct(metrics.quick_prev_20)}, trend={metrics.quick_trend})"
    )
    lines.append(
        f"Final Accuracy: {_fmt_pct(metrics.final_accuracy)} "
        f"(resolved={metrics.final_resolved}, last20={_fmt_pct(metrics.final_last_20)}, "
        f"prev20={_fmt_pct(metrics.final_prev_20)}, trend={metrics.final_trend})"
    )
    lines.append("")
    lines.append("Latest Runs:")

    display_cols = [
        "RunDate",
        "TargetDate",
        "Signal",
        "Dip Probability",
        "QuickActual",
        "QuickCorrect",
        "FinalActual",
        "FinalCorrect",
    ]
    available_cols = [c for c in display_cols if c in log_df.columns]

    if not available_cols:
        lines.append("No run data available yet.")
        return "\n".join(lines) + "\n"

    tail = log_df.sort_values("TargetDate").tail(15)[available_cols]
    lines.append(tail.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def generate_daily_reports(log_df: pd.DataFrame, output_dir: Path) -> LiveMetrics:
    """Generate consolidated run view + metrics reports in outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = log_df.copy()
    if "TargetDate" in df.columns:
        df = df.sort_values("TargetDate")

    # Full run history in one easy place
    runs_path = output_dir / "all_runs_report.csv"
    df.to_csv(runs_path, index=False)

    # Snapshot metrics report (one row per run day)
    metrics = compute_live_metrics(df)
    snapshot_row = pd.DataFrame(
        [
            {
                "ReportGeneratedAtUTC": pd.Timestamp.now(tz="UTC").isoformat(),
                "LatestRunDate": (df["RunDate"].iloc[-1] if ("RunDate" in df.columns and not df.empty) else None),
                "LatestTargetDate": (df["TargetDate"].iloc[-1] if ("TargetDate" in df.columns and not df.empty) else None),
                "TotalRuns": metrics.total_runs,
                "QuickResolved": metrics.quick_resolved,
                "QuickAccuracy": metrics.quick_accuracy,
                "QuickLast20": metrics.quick_last_20,
                "QuickPrev20": metrics.quick_prev_20,
                "QuickTrend": metrics.quick_trend,
                "FinalResolved": metrics.final_resolved,
                "FinalAccuracy": metrics.final_accuracy,
                "FinalLast20": metrics.final_last_20,
                "FinalPrev20": metrics.final_prev_20,
                "FinalTrend": metrics.final_trend,
            }
        ]
    )
    summary_path = output_dir / "live_accuracy_report.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, snapshot_row], ignore_index=True)
    else:
        combined = snapshot_row
    combined.to_csv(summary_path, index=False)

    dashboard_path = output_dir / "daily_dashboard.txt"
    dashboard_path.write_text(build_dashboard_text(df, metrics), encoding="utf-8")
    return metrics


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except Exception:
        return None


def build_training_manifest(
    feat_df: pd.DataFrame,
    model_bundle: dict,
    avg_metrics: dict,
    config_dict: dict,
) -> dict:
    """Create a rich, auditable training manifest."""
    df = feat_df.copy()
    labeled = df["label"].notna() if "label" in df.columns else pd.Series(False, index=df.index)
    used = df[labeled]

    date_min = df.index.min().date().isoformat() if len(df) else None
    date_max = df.index.max().date().isoformat() if len(df) else None
    train_min = used.index.min().date().isoformat() if len(used) else None
    train_max = used.index.max().date().isoformat() if len(used) else None

    label_non_null = int(labeled.sum()) if "label" in df.columns else 0
    label_positive = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    label_positive_rate = (label_positive / label_non_null) if label_non_null else None

    meta = model_bundle.get("metadata", {})
    manifest = {
        "TrainedAtUTC": pd.Timestamp.now(tz="UTC").isoformat(),
        "ModelType": meta.get("model_type"),
        "DecisionThreshold": _safe_float(meta.get("decision_threshold")),
        "PurgeGapDays": meta.get("purge_gap_days"),
        "FeatureCount": len(model_bundle.get("feature_columns", [])),
        "TotalFeatureRows": int(len(df)),
        "TrainRowsUsed": int(meta.get("train_rows", 0)),
        "DataStartDate": date_min,
        "DataEndDate": date_max,
        "TrainWindowStartDate": train_min,
        "TrainWindowEndDate": train_max,
        "LabelRowsNonNull": label_non_null,
        "LabelPositiveRows": label_positive,
        "LabelPositiveRate": _safe_float(label_positive_rate),
        "CVSplits": config_dict.get("cv_splits"),
        "ReboundHorizonDays": config_dict.get("rebound_horizon_days"),
        "AvgPrecision": _safe_float(avg_metrics.get("precision")),
        "AvgRecall": _safe_float(avg_metrics.get("recall")),
        "AvgF1": _safe_float(avg_metrics.get("f1")),
        "AvgBalancedAccuracy": _safe_float(avg_metrics.get("balanced_accuracy")),
        "AvgROCAUC": _safe_float(avg_metrics.get("roc_auc")),
        "AvgPRAUC": _safe_float(avg_metrics.get("pr_auc")),
        "AvgBrier": _safe_float(avg_metrics.get("brier")),
        "AvgTrainF1": _safe_float(avg_metrics.get("train_f1")),
        "AvgF1Gap": _safe_float(avg_metrics.get("f1_gap")),
        "ConfigSnapshot": json.dumps(config_dict, default=str, sort_keys=True),
    }
    return manifest


def save_training_manifest(manifest: dict, history_path: Path, latest_json_path: Path) -> None:
    """Persist training manifest as append-only CSV history + latest JSON snapshot."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    latest_json_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([manifest])
    if history_path.exists():
        old = pd.read_csv(history_path)
        history = pd.concat([old, row_df], ignore_index=True)
    else:
        history = row_df
    history.to_csv(history_path, index=False)

    latest_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
