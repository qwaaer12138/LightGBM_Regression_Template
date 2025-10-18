# Public Dataset Example

This directory contains a reproducible example that trains the LightGBM regression
pipeline on the **Boston Housing** dataset from the UCI Machine Learning Repository.

## Dataset provenance

- **Source:** Harrison Jr. & Rubinfeld, "Hedonic housing prices and the demand for clean air".
  Available via the UCI Machine Learning Repository.
- **URL:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv
- **License:** The UCI dataset card states there are *no known copyright restrictions*; the
  data are derived from public domain U.S. Census Service materials.

The example script downloads the CSV directly from the URL above. If you prefer to retrieve the
file manually, save it to `examples/data/boston_housing.csv` before running the script.

## Contents

- `run_on_public_dataset.py` &mdash; downloads the dataset, prepares train/test splits, executes
  the pipeline, and records evaluation artifacts (cross-validation metrics, hold-out MAE, and
  predictions).
- `data/` &mdash; created automatically to store the raw dataset and derived train/test files.
- `outputs/` &mdash; created automatically to store pipeline artifacts for the example run.

Refer to the top-level `README.md` for end-to-end execution instructions.
