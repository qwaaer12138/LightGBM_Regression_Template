import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("lightgbm")

from lightgbm_regression import generate_predictions, train_regression_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(__file__).resolve().parent / "data" / "boston_housing.csv"
SEED = 2025

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.2,
    "num_leaves": 15,
    "max_depth": -1,
    "n_estimators": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
}


def _read_predictions(path: Path) -> np.ndarray:
    predictions_df = pd.read_csv(path)
    return predictions_df["prediction"].to_numpy()


def test_cli_and_library_predictions_are_consistent() -> None:
    df = pd.read_csv(DATA_PATH)

    with TemporaryDirectory() as output_dir, TemporaryDirectory() as log_dir:
        output_path = Path(output_dir)
        log_path = Path(log_dir)
        script_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        command = [
            sys.executable,
            str(script_path),
            "--train-path",
            str(DATA_PATH),
            "--test-path",
            str(DATA_PATH),
            "--target-column",
            "medv",
            "--output-dir",
            str(output_path),
            "--log-dir",
            str(log_path),
            "--n-splits",
            "3",
            "--random-state",
            str(SEED),
            "--learning-rate",
            "0.2",
            "--num-leaves",
            "15",
            "--max-depth",
            "-1",
            "--n-estimators",
            "50",
            "--subsample",
            "0.8",
            "--colsample-bytree",
            "0.8",
            "--early-stopping-rounds",
            "10",
        ]

        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                "CLI pipeline failed"
                f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        cli_predictions = _read_predictions(output_path / "predictions.csv")

    training_result = train_regression_model(
        df,
        target_column="medv",
        test_data=df,
        n_splits=3,
        random_state=SEED,
        lgbm_params=LGBM_PARAMS,
        early_stopping_rounds=10,
    )
    library_predictions = generate_predictions(
        model=training_result.best_model,
        data=df,
        feature_columns=training_result.feature_columns,
    )

    np.testing.assert_allclose(
        cli_predictions,
        library_predictions["prediction"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )


def test_train_regression_model_is_reproducible() -> None:
    df = pd.read_csv(DATA_PATH)

    first_result = train_regression_model(
        df,
        target_column="medv",
        test_data=df,
        n_splits=3,
        random_state=SEED,
        lgbm_params=LGBM_PARAMS,
        early_stopping_rounds=10,
    )
    second_result = train_regression_model(
        df,
        target_column="medv",
        test_data=df,
        n_splits=3,
        random_state=SEED,
        lgbm_params=LGBM_PARAMS,
        early_stopping_rounds=10,
    )

    first_predictions = generate_predictions(
        model=first_result.best_model,
        data=df,
        feature_columns=first_result.feature_columns,
    )["prediction"].to_numpy()
    second_predictions = generate_predictions(
        model=second_result.best_model,
        data=df,
        feature_columns=second_result.feature_columns,
    )["prediction"].to_numpy()

    np.testing.assert_allclose(first_predictions, second_predictions, rtol=1e-8, atol=1e-8)
