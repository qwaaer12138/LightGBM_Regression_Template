"""CLI entry point for the LightGBM regression pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.data_utils import DatasetBundle, load_datasets, split_features_and_target
from src.infer import generate_predictions
from src.train import TrainingConfig, train_with_cv


def configure_logging(log_dir: Path, log_level: str = "INFO") -> Path:
    """Configure logging to stream to stdout and a timestamped file."""

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"training_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LightGBM training pipeline")
    parser.add_argument("--train-path", required=True, help="Path to df_train CSV file")
    parser.add_argument("--test-path", required=True, help="Path to df_test CSV file")
    parser.add_argument(
        "--target-column",
        required=True,
        help="Target column name present in df_train",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        help="Optional columns to drop before training (e.g., identifiers)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store inference outputs",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to store training logs",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-validation",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Number of leaves for LightGBM",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Maximum tree depth for LightGBM",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1000,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio of the training instance",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Subsample ratio of columns when constructing each tree",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0,
        help="L1 regularization term",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=0.0,
        help="L2 regularization term",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=100,
        help="Early stopping rounds for LightGBM",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )

    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    params: Dict[str, Any] = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.random_state,
    }

    return TrainingConfig(
        n_splits=args.n_splits,
        random_state=args.random_state,
        lgbm_params=params,
        early_stopping_rounds=args.early_stopping_rounds,
    )


def log_cross_validation_summary(
    dataset: DatasetBundle,
    training_result,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    summary = {
        "folds": [
            {
                "fold": result.fold,
                "train_mae": result.train_mae,
                "valid_mae": result.valid_mae,
                "best_iteration": result.best_iteration,
            }
            for result in training_result.fold_results
        ],
        "best_fold": training_result.best_fold,
        "feature_columns": training_result.feature_columns,
        "target_column": dataset.target_column,
    }
    summary_path = output_dir / "cv_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Cross-validation summary saved to %s", summary_path)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    log_path = configure_logging(log_dir, log_level=args.log_level)
    logger = logging.getLogger("pipeline")
    logger.info("Log file created at %s", log_path)

    dataset = load_datasets(
        train_path=args.train_path,
        test_path=args.test_path,
        target_column=args.target_column,
        drop_columns=args.drop_columns,
    )
    features, target = split_features_and_target(dataset)

    training_config = build_training_config(args)

    logger.info(
        "Starting cross-validation with %d folds", training_config.n_splits
    )
    training_result = train_with_cv(
        features=features,
        target=target,
        config=training_config,
        logger=logger,
    )

    log_cross_validation_summary(dataset, training_result, output_dir, logger)

    predictions_path = output_dir / "predictions.csv"
    generate_predictions(
        model=training_result.best_model,
        data=dataset.test,
        feature_columns=training_result.feature_columns,
        output_path=predictions_path,
    )
    logger.info("Predictions saved to %s", predictions_path)


if __name__ == "__main__":
    main()
