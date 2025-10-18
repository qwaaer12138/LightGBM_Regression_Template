# LightGBM Regression Template

This repository provides a light-weight template for training a LightGBM regressor
with k-fold cross-validation and running inference on a held-out test set.

## Project Structure

```
├── scripts/
│   └── run_pipeline.py   # CLI entry point for the full pipeline
├── src/
│   ├── data_utils.py     # Dataset loading and preprocessing helpers
│   ├── infer.py          # Inference helpers
│   └── train.py          # Cross-validation training loop
├── outputs/              # Prediction artifacts (created automatically)
├── logs/                 # Training logs (created automatically)
└── README.md
```

## Requirements

Install dependencies with pip:

```bash
pip install lightgbm pandas scikit-learn
```

## Usage

Run the complete pipeline with the CLI script. Provide paths to the training and test
CSV files, the target column name, and any optional configuration parameters.

```bash
python scripts/run_pipeline.py \
    --train-path /path/to/df_train.csv \
    --test-path /path/to/df_test.csv \
    --target-column target
```

Key command-line arguments:

- `--drop-columns`: columns to exclude from both datasets before training
- `--output-dir`: directory where predictions (`predictions.csv`) and cross-validation
  summary (`cv_results.json`) will be saved (default: `outputs/`)
- `--log-dir`: directory where timestamped log files will be created (default: `logs/`)
- `--n-splits`: number of cross-validation folds (default: `5`)
- `--learning-rate`, `--num-leaves`, `--max-depth`, `--n-estimators`, `--subsample`,
  `--colsample-bytree`, `--reg-alpha`, `--reg-lambda`: LightGBM hyperparameters
- `--early-stopping-rounds`: early stopping patience (default: `100`)
- `--log-level`: logging verbosity (default: `INFO`)

The script prints fold-by-fold MAE metrics to the console, persists the same logs to a
file under `logs/`, stores a JSON summary of cross-validation metrics under the
specified `output` directory, and writes test-set predictions to
`outputs/predictions.csv`.

## Example: Run on the UCI Boston Housing dataset

To try the pipeline end-to-end on a permissive public dataset, use the helper script in
`examples/`. It downloads the [Boston Housing](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
dataset from the UCI Machine Learning Repository (public domain), creates a reproducible
train/test split, and executes the LightGBM pipeline.

```bash
python examples/run_on_public_dataset.py
```

The script prints progress to the console and writes artifacts to `examples/data/` and
`examples/outputs/boston_housing/`:

- `examples/data/` &mdash; cached raw dataset, train split, test split, and hold-out targets.
- `examples/outputs/boston_housing/cv_results.json` &mdash; cross-validation MAE summary from the pipeline.
- `examples/outputs/boston_housing/predictions.csv` &mdash; pipeline predictions for the hold-out set.
- `examples/outputs/boston_housing/holdout_metrics.json` &mdash; MAE measured against the withheld targets.
- `examples/outputs/boston_housing/holdout_predictions.csv` &mdash; combined table with target and prediction.
- `examples/outputs/boston_housing/logs/` &mdash; timestamped training logs.

You can re-run the script with `--force-download` to refresh the dataset or with
`--output-dir`/`--data-dir` to customize artifact locations.
