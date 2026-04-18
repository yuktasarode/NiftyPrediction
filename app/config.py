"""Project configuration for dip-zone prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Central configuration for data, labels, model, and outputs."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Data
    ticker: str = "^NSEI"
    start_date: str = "2008-01-01"
    refresh_data: bool = False
    market_timezone: str = "Asia/Kolkata"
    market_close_hour: int = 15
    market_close_minute: int = 45

    # Labeling (rebound mode)
    label_mode: str = "rebound"  # rebound | local_bottom
    drawdown_20_threshold: float = 0.05
    drawdown_60_threshold: float = 0.08
    rebound_horizon_days: int = 15
    rebound_threshold: float = 0.03
    max_additional_drawdown: float = 0.04
    max_rebound_wait_days: int = 10

    # Alternative local-bottom mode
    local_bottom_tolerance: float = 0.01

    # Technical indicators
    rsi_period: int = 14
    rsi_threshold: float = 35.0
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    bollinger_touch_buffer: float = 0.01

    # Model
    model_type: str = "ensemble"  # logistic | random_forest | extra_trees | hist_gb | ensemble
    retrain: bool = True
    cv_splits: int = 5
    random_state: int = 42
    decision_threshold: float = 0.50
    calibrate_probabilities: bool = False

    # Signal thresholds
    ml_watch_threshold: float = 0.45
    ml_dip_zone_threshold: float = 0.65
    rule_watch_threshold: float = 0.50
    rule_dip_zone_threshold: float = 0.75

    # Live evaluation (next-day quick check + mature label check)
    quick_eval_drop_threshold: float = -0.003
    prediction_log_key: str = "TargetDate"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / "nsei_raw.csv"

    @property
    def features_path(self) -> Path:
        return self.data_dir / "nsei_features.csv"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "dip_zone_model.joblib"

    @property
    def prediction_log_path(self) -> Path:
        return self.output_dir / "prediction_log.csv"

    @property
    def summary_json_path(self) -> Path:
        return self.output_dir / "latest_prediction.json"

    @property
    def summary_txt_path(self) -> Path:
        return self.output_dir / "latest_prediction.txt"

    def as_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        data = asdict(self)
        data["base_dir"] = str(self.base_dir)
        return data
