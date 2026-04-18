# NIFTY 50 Dip-Zone Predictor (Local First)

This project predicts whether **today is a realistic buy-the-dip zone** for NIFTY 50 (`^NSEI`) using:

- A configurable **rule-based baseline detector**
- A configurable **ML classifier** (Logistic Regression or Random Forest)
- Labels defined by **drawdown + forward rebound behavior**, not exact bottom timing
- Next-trading-day prediction workflow with rolling live accuracy logs

## Important Note

This is a research tool and not guaranteed financial advice. Markets are uncertain. Use risk management and independent judgment.

## Project Structure

```text
niftyPrediction/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── labeling.py
│   ├── model.py
│   ├── predict.py
│   ├── plotting.py
│   └── utils.py
├── data/
├── outputs/
├── tests/
│   ├── test_features.py
│   ├── test_labeling.py
│   ├── test_no_leakage.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── run_daily.py
```

## Dip-Zone Label Logic

Default `rebound` mode labels day `t` as `1` if:

1. Day `t` is in meaningful drawdown:
- drawdown from 20-day high >= `drawdown_20_threshold`, or
- drawdown from 60-day high >= `drawdown_60_threshold`

2. Future rebound is strong enough:
- `max(Close(t+1...t+horizon)) / Close(t) - 1 >= rebound_threshold`
- rebound must arrive within `max_rebound_wait_days`
- path should avoid large additional crash: `min(Close(t+1...t+horizon)) / Close(t) - 1 >= -max_additional_drawdown`

Alternative `local_bottom` mode is also supported via config.

## Setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Daily After Market Close

```bash
source .venv/bin/activate
python run_daily.py
```

What it does:

1. Downloads/updates `^NSEI` OHLCV from Yahoo Finance (`yfinance`) incrementally
2. Saves raw data to `data/nsei_raw.csv`
3. Builds features and saves to `data/nsei_features.csv`
4. Trains (or loads) model
5. Predicts for the **next trading day** (Target Date)
6. Updates rolling live accuracy in `outputs/prediction_log.csv`:
- quick outcome (next-day dip/drop proxy, available immediately after target close)
- final outcome (mature dip-zone label once forward horizon is available)
7. Saves:
- `outputs/latest_prediction.json`
- `outputs/latest_prediction.txt`
- `outputs/price_with_dip_labels.png`
- `outputs/buy_signals.png`
- `outputs/dip_probability_over_time.png`

## Configuration

Tune values in `app/config.py`:

- `ticker`
- `start_date`
- `drawdown_20_threshold`
- `drawdown_60_threshold`
- `rebound_threshold`
- `rebound_horizon_days`
- `max_rebound_wait_days`
- `max_additional_drawdown`
- `label_mode`
- `rsi_threshold`
- `bollinger_window`
- `bollinger_std`
- `model_type` (`logistic` or `random_forest`)
- `model_type` (`logistic`, `random_forest`, `extra_trees`, `hist_gb`, `ensemble`)
- `retrain`
- `decision_threshold`
- `quick_eval_drop_threshold`

## Evaluation

Uses `TimeSeriesSplit` (no random split) and prints per-fold + average:

- precision
- recall
- f1
- balanced accuracy
- ROC AUC (when probabilities are available)
- PR AUC
- Brier score
- train-vs-test F1 gap (overfit check)
- purged gap based on forward label horizon (reduces label overlap leakage)

## Rule-Based Baseline

Rules include:

- drawdown condition
- RSI oversold condition
- close near/below lower Bollinger Band
- negative recent 5-day return

Outputs rule score and rule signal (`NO DIP`, `DIP WATCH`, `DIP ZONE`) alongside ML output.

## Run Tests

```bash
source .venv/bin/activate
pytest -q
```

## Example Terminal Output

```text
Validation metrics by fold:
Fold 1: precision=0.556, recall=0.385, f1=0.455, roc_auc=0.677
Fold 2: precision=0.612, recall=0.441, f1=0.513, roc_auc=0.702
Fold 3: precision=0.591, recall=0.474, f1=0.526, roc_auc=0.714
Fold 4: precision=0.635, recall=0.500, f1=0.559, roc_auc=0.731
Fold 5: precision=0.600, recall=0.488, f1=0.538, roc_auc=0.720
Average metrics:
precision=0.599, recall=0.458, f1=0.518, roc_auc=0.709

Date: 2026-04-17
NIFTY Close: 22514.65
Dip Probability: 0.673
Rule-Based Score: 0.750
ML Signal: DIP ZONE
Rule Signal: DIP ZONE
Signal: DIP ZONE
```

## Limitations

- Labels are heuristic and dependent on threshold choices.
- Model may overfit to historical regimes.
- In-sample probability chart is not an out-of-sample trading backtest.
- Transaction costs, slippage, and execution constraints are not modeled.
- Yahoo daily data can occasionally arrive late/with revisions.
