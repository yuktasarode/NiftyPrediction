"""Daily runner for NIFTY dip-zone prediction."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".mplconfig"))

from app.config import AppConfig
from app.data import update_nifty_data
from app.features import build_features
from app.labeling import apply_rule_engine, create_dip_labels
from app.model import load_model_bundle, save_model_bundle, train_and_evaluate
from app.plotting import plot_buy_signals, plot_price_with_labels, plot_probability
from app.predict import historical_probabilities, predict_latest, prediction_to_text
from app.reporting import build_training_manifest, generate_daily_reports, save_training_manifest
from app.utils import append_prediction_log, ensure_directories, save_json, save_text, update_live_accuracy_log


def main() -> None:
    config = AppConfig()
    ensure_directories(config.data_dir, config.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(config.output_dir / ".mplconfig"))
    ensure_directories(config.output_dir / ".mplconfig")

    # 1) Data
    raw = update_nifty_data(config)

    # 2) Features + labels + rule baseline
    feat = build_features(raw, config)
    feat["label"] = create_dip_labels(feat, config)
    rules = apply_rule_engine(feat, config)
    feat = feat.join(rules)
    feat.to_csv(config.features_path, index=True)

    # 3) Train/load
    if config.retrain or (not config.model_path.exists()):
        try:
            bundle, fold_metrics, avg_metrics = train_and_evaluate(feat, config)
            save_model_bundle(bundle, config.model_path)
            training_manifest = build_training_manifest(feat, bundle, avg_metrics, config.as_dict())
            save_training_manifest(
                training_manifest,
                history_path=config.training_history_path,
                latest_json_path=config.latest_training_manifest_path,
            )

            print("\nValidation metrics by fold:")
            for fm in fold_metrics:
                print(
                    f"Fold {fm['fold']}: "
                    f"precision={fm['precision']:.3f}, recall={fm['recall']:.3f}, "
                    f"f1={fm['f1']:.3f}, bal_acc={fm['balanced_accuracy']:.3f}, "
                    f"roc_auc={fm['roc_auc']:.3f}, pr_auc={fm['pr_auc']:.3f}, brier={fm['brier']:.3f}, "
                    f"train_f1={fm['train_f1']:.3f}, f1_gap={fm['f1_gap']:.3f}"
                )

            print("Average metrics:")
            print(
                f"precision={avg_metrics['precision']:.3f}, recall={avg_metrics['recall']:.3f}, "
                f"f1={avg_metrics['f1']:.3f}, bal_acc={avg_metrics['balanced_accuracy']:.3f}, "
                f"roc_auc={avg_metrics['roc_auc']:.3f}, pr_auc={avg_metrics['pr_auc']:.3f}, "
                f"brier={avg_metrics['brier']:.3f}, train_f1={avg_metrics['train_f1']:.3f}, "
                f"f1_gap={avg_metrics['f1_gap']:.3f}, purge_gap={avg_metrics['purge_gap_days']}, "
                f"threshold={avg_metrics['decision_threshold']:.2f}"
            )
            print(
                "Training manifest saved: "
                "outputs/latest_training_manifest.json | outputs/model_training_history.csv"
            )
        except RuntimeError as exc:
            if config.model_path.exists():
                print(f"Training skipped: {exc}")
                print(f"Using existing model at {config.model_path}")
                bundle = load_model_bundle(config.model_path)
            else:
                raise
    else:
        bundle = load_model_bundle(config.model_path)

    # 4) Predict latest
    pred = predict_latest(feat, bundle, config)
    pred_text = prediction_to_text(pred)
    print("\n" + pred_text)

    # 5) Persist prediction summary and log
    labels = feat["label"]
    save_json(config.summary_json_path, pred)
    save_text(config.summary_txt_path, pred_text)

    log_df = append_prediction_log(config.prediction_log_path, pred)
    log_df = update_live_accuracy_log(log_df, feat, labels, config)
    log_df.to_csv(config.prediction_log_path, index=False)

    quick_resolved = log_df.dropna(subset=["QuickCorrect"]) if "QuickCorrect" in log_df.columns else log_df.iloc[0:0]
    final_resolved = log_df.dropna(subset=["FinalCorrect"]) if "FinalCorrect" in log_df.columns else log_df.iloc[0:0]
    if not quick_resolved.empty:
        quick_acc = float(quick_resolved["QuickCorrect"].astype(float).mean())
        print(f"Quick live accuracy (resolved): {quick_acc:.3f} over {len(quick_resolved)} predictions")
    if not final_resolved.empty:
        final_acc = float(final_resolved["FinalCorrect"].astype(float).mean())
        print(f"Final dip-label accuracy (resolved): {final_acc:.3f} over {len(final_resolved)} predictions")

    metrics = generate_daily_reports(log_df, config.output_dir)
    print(
        "Dashboard updated: "
        f"outputs/daily_dashboard.txt | outputs/all_runs_report.csv | outputs/live_accuracy_report.csv"
    )
    print(
        f"Trend: quick={metrics.quick_trend} (last20={metrics.quick_last_20}), "
        f"final={metrics.final_trend} (last20={metrics.final_last_20})"
    )

    # 6) Plots
    prob_series = historical_probabilities(feat, bundle)
    plot_price_with_labels(feat, labels, config.output_dir / "price_with_dip_labels.png")
    plot_buy_signals(feat, prob_series, config.output_dir / "buy_signals.png", threshold=config.ml_dip_zone_threshold)
    plot_probability(prob_series, config.output_dir / "dip_probability_over_time.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("run_daily failed:")
        print(str(exc))
        print(traceback.format_exc())
        raise
