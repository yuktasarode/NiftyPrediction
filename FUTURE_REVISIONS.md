# Future Revisions

## 1) 7-Day Model Review Report

### Goal
Create an automatic 7-day model review that helps decide whether the current model should be adjusted.

### Why
We now log predictions, outcomes, and training manifests. Next step is to connect these into a compact review loop.

### Scope
- Build a report file: `outputs/model_review_7d.csv`
- Build a readable summary file: `outputs/model_review_7d.txt`
- Review window: last 7 resolved target days (rolling)
- Join data from:
  - `outputs/prediction_log.csv`
  - `outputs/latest_training_manifest.json` and/or `outputs/model_training_history.csv`

### Metrics to include
- 7-day quick accuracy
- 7-day final accuracy (if available)
- Precision / Recall / F1 over the 7-day resolved window
- Count of DIP vs NO DIP predictions
- Count of false positives / false negatives
- Average predicted dip probability

### Diagnostics to include
- Was performance better/worse than previous 7-day window?
- Drift hints:
  - change in signal frequency
  - change in average probability
  - change in realized class balance
- Active model metadata snapshot used during this window:
  - model type
  - train window
  - train rows
  - label positive rate

### Output behavior
- Generated automatically at end of daily run
- If <7 resolved rows are available, generate partial report and mark as `insufficient_data`

### Acceptance criteria
- Running `./run_daily.sh` updates both 7-day review files
- Report clearly states window dates and sample size
- Review can be used to decide whether to tune thresholds/features/model
