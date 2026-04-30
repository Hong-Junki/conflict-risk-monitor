# Conflict Forecasting Pipeline

This project is set up for the guide's option A submission flow:

1. inspect the processed data
2. train a few baseline models on `train.parquet`
3. choose the model using `val.parquet`
4. retrain the chosen model on `train + val`
5. create `predictions.csv` for `test.parquet`

## Recommended starting point

- Main target: `y_escalation`
- Main model: `LightGBM`
- Comparison models: `LogisticRegression`, `XGBoost`
- Selection metric: validation `PR-AUC`

## Files

- `scripts/eda.py`: quick data inspection and target summary
- `scripts/train_models.py`: train and compare models on train/val
- `scripts/make_submission.py`: retrain best config on train+val and create submission csv
- `scripts/make_risk_scores.py`: create guide-style 0-100 risk scores from predictions
- `scripts/evaluate_submission.py`: evaluate a prediction csv against the local test labels
- `scripts/build_site.py`: build a Korean world-map prototype from risk scores
- `src/config.py`: shared paths and column rules
- `src/data_utils.py`: dataset loading and preprocessing
- `src/metrics.py`: evaluation metrics used in the guide

## Quick start

```bash
python scripts/eda.py
python scripts/train_models.py --dataset se --date-mode ordinal
python scripts/make_submission.py --dataset se --run-name my_first_run
python scripts/evaluate_submission.py --pred artifacts/submissions/predictions__xgboost__my_first_run.csv
python scripts/make_risk_scores.py --pred artifacts/submissions/predictions__xgboost__my_first_run.csv
python scripts/build_site.py
```

Outputs are written to `artifacts/`.

To run with the extra `macis_se_score` feature:

```bash
python scripts/train_models.py --dataset se
python scripts/make_submission.py --dataset se --run-name my_se_run
```

## Expected output

The final file will be:

```text
artifacts/submissions/predictions__lightgbm__my_first_run.csv
```

It will contain:

```csv
date,country,y_prob
2024-07-01,UKR,0.1234
...
```

## How to read the results

- `artifacts/model_comparison.csv`
  - Compare validation metrics across models.
  - Focus on `pr_auc` first.
  - Then check `p_at_5pct` and `persistence_gain`.

- `artifacts/best_model_config.json`
  - Shows which model won on validation and which features were used.

- `artifacts/submissions/...csv`
  - This is your guide-compatible submission file.

- `artifacts/submissions/risk_scores__...csv`
  - This is the guide-style 0-100 score file for interpretation.
  - The option A submission file remains the `predictions__...csv` file.

- `site/index.html`
  - This is a Korean static prototype for viewing country-level risk scores on a world map.
  - Open it directly in a browser.

## GitHub Pages deployment

`scripts/build_site.py` writes the static prototype to both `site/index.html` and `docs/index.html`.
Use `docs/` as the GitHub Pages publishing folder.

```bash
python scripts/build_site.py
git add README.md report.md requirements.txt scripts src docs .gitignore
git commit -m "Add conflict risk monitor prototype"
git push
```

On GitHub:

1. Open the repository settings.
2. Go to Pages.
3. Choose `Deploy from a branch`.
4. Select branch `main` and folder `/docs`.
5. Save.

## Decision rule

- If `lightgbm` has the best `pr_auc`, keep it as your submission model.
- If `xgboost` is slightly better on `pr_auc` but much worse on `p_at_5pct`, compare both.
- If all models are weak, prioritize the one with positive `persistence_gain` and stable `p_at_5pct`.
- If `se` beats `base`, submit the `se` run.
