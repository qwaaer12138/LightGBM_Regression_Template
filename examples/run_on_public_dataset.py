"""Run the LightGBM pipeline on the UCI Boston Housing dataset."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import os
from pathlib import Path
from typing import Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
DATASET_NAME = "Boston Housing"
TARGET_COLUMN = "MEDV"
COLUMN_NAMES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
RAW_FILENAME = "boston_housing.csv"
TRAIN_FILENAME = "boston_housing_train.csv"
TEST_FILENAME = "boston_housing_test.csv"
TEST_TARGETS_FILENAME = "boston_housing_test_targets.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Boston Housing dataset, prepare a train/test split, run the "
            "LightGBM regression pipeline, and record evaluation artifacts."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "examples/data",
        help="Directory where intermediate CSV files will be stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "examples/outputs/boston_housing",
        help="Directory to store pipeline outputs and evaluation metrics.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to allocate to the hold-out test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split and LightGBM pipeline.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the raw dataset even if it already exists locally.",
    )
    return parser.parse_args()


def download_dataset(data_dir: Path, force_download: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_FILENAME

    if raw_path.exists() and not force_download:
        print(f"Reusing existing dataset at {raw_path}")
        return raw_path

    print(f"Downloading {DATASET_NAME} dataset from {DATA_URL}")
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request) as response:
            data = response.read()
    except URLError as exc:
        raise RuntimeError(
            "Failed to download the Boston Housing dataset. Verify network access "
            "or download the CSV manually as described in examples/README.md."
        ) from exc

    df = pd.read_csv(io.BytesIO(data), header=None, names=COLUMN_NAMES)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to {raw_path}")
    return raw_path


def prepare_train_test(
    raw_path: Path, test_size: float, random_state: int
) -> Tuple[Path, Path, Path]:
    df = pd.read_csv(raw_path)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_path = raw_path.parent / TRAIN_FILENAME
    test_features_path = raw_path.parent / TEST_FILENAME
    test_targets_path = raw_path.parent / TEST_TARGETS_FILENAME

    train_df.to_csv(train_path, index=False)

    test_targets = test_df[[TARGET_COLUMN]].copy()
    test_targets.to_csv(test_targets_path, index=False)

    test_features = test_df.drop(columns=[TARGET_COLUMN])
    test_features.to_csv(test_features_path, index=False)

    print(f"Prepared train/test splits under {raw_path.parent}")
    return train_path, test_features_path, test_targets_path


def run_pipeline(
    train_path: Path, test_path: Path, output_dir: Path, random_state: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "scripts/run_pipeline.py",
        "--train-path",
        str(train_path),
        "--test-path",
        str(test_path),
        "--target-column",
        TARGET_COLUMN,
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(output_dir / "logs"),
        "--random-state",
        str(random_state),
    ]

    print("Running training pipeline:\n  " + " ".join(cmd))
    env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def record_holdout_metrics(
    test_targets_path: Path, predictions_path: Path, output_dir: Path
) -> Path:
    targets = pd.read_csv(test_targets_path)
    predictions = pd.read_csv(predictions_path)

    if len(targets) != len(predictions):
        raise ValueError(
            "Predictions and hold-out targets have mismatched row counts: "
            f"{len(predictions)} vs {len(targets)}"
        )

    mae = mean_absolute_error(targets[TARGET_COLUMN], predictions["prediction"])

    metrics = {
        "dataset": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "holdout_mae": mae,
        "num_holdout_rows": len(targets),
    }

    metrics_path = output_dir / "holdout_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    combined = targets.copy()
    combined["prediction"] = predictions["prediction"]
    combined_path = output_dir / "holdout_predictions.csv"
    combined.to_csv(combined_path, index=False)

    print(f"Recorded hold-out MAE to {metrics_path}")
    print(f"Wrote predictions with ground truth to {combined_path}")

    return metrics_path


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    raw_path = download_dataset(data_dir, force_download=args.force_download)
    train_path, test_path, test_targets_path = prepare_train_test(
        raw_path, test_size=args.test_size, random_state=args.random_state
    )

    run_pipeline(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        random_state=args.random_state,
    )

    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Expected predictions.csv from the training pipeline is missing at "
            f"{predictions_path}"
        )

    record_holdout_metrics(
        test_targets_path=test_targets_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
    )

    cv_results_path = output_dir / "cv_results.json"
    if cv_results_path.exists():
        print(f"Cross-validation results available at {cv_results_path}")
    else:
        print(
            "Warning: cv_results.json was not generated by the pipeline run; "
            "check pipeline logs for errors."
        )


if __name__ == "__main__":
    main()
