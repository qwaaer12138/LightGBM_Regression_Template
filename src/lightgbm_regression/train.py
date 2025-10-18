"""LightGBM training utilities."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from ._typing import PathLike


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training."""

    n_splits: int = 5
    random_state: int = 42
    lgbm_params: Dict[str, object] = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
        }
    )
    early_stopping_rounds: Optional[int] = 100


@dataclass(slots=True)
class FoldResult:
    """Metrics recorded for a single cross-validation fold."""

    fold: int
    train_mae: float
    valid_mae: float
    best_iteration: int


@dataclass(slots=True)
class TrainingResult:
    """Aggregate training outputs produced by :func:`train_with_cv`."""

    best_model: lgb.LGBMRegressor
    best_fold: int
    fold_results: List[FoldResult]
    feature_columns: List[str]


DataFrameLike = pd.DataFrame | PathLike


def _create_model(params: Dict[str, object]) -> lgb.LGBMRegressor:
    """Instantiate an :class:`~lightgbm.LGBMRegressor` with defaults."""

    model_params = params.copy()
    model_params.setdefault("verbosity", -1)
    return lgb.LGBMRegressor(**model_params)


def train_with_cv(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig,
    logger: logging.Logger | None = None,
) -> TrainingResult:
    """Train a LightGBM regressor with cross-validation."""

    logger = logger or logging.getLogger(__name__)
    kfold = KFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_state
    )

    fold_results: List[FoldResult] = []
    best_model: Optional[lgb.LGBMRegressor] = None
    best_fold = -1
    best_mae = float("inf")

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(features), start=1):
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

        model = _create_model(config.lgbm_params)
        callbacks = [lgb.log_evaluation(period=50)]
        if config.early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=config.early_stopping_rounds)
            )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=callbacks,
        )

        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        train_mae = mean_absolute_error(y_train, train_pred)
        valid_mae = mean_absolute_error(y_valid, valid_pred)

        best_iteration = getattr(model, "best_iteration_", model.n_estimators)

        result = FoldResult(
            fold=fold,
            train_mae=train_mae,
            valid_mae=valid_mae,
            best_iteration=best_iteration,
        )
        fold_results.append(result)
        logger.info(
            "Fold %d - train MAE: %.5f | valid MAE: %.5f | best iteration: %s",
            fold,
            train_mae,
            valid_mae,
            best_iteration,
        )

        if valid_mae < best_mae:
            best_mae = valid_mae
            best_fold = fold
            best_model = copy.deepcopy(model)

    if best_model is None:
        raise RuntimeError("Training did not produce any model. Check your data.")

    logger.info("Best fold: %d with validation MAE %.5f", best_fold, best_mae)

    return TrainingResult(
        best_model=best_model,
        best_fold=best_fold,
        fold_results=fold_results,
        feature_columns=list(features.columns),
    )


def _ensure_dataframe(data: DataFrameLike, *, copy_df: bool = True) -> pd.DataFrame:
    """Return a dataframe from an in-memory object or a file path."""

    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True) if copy_df else data

    if isinstance(data, (str, Path)):
        return pd.read_csv(data)

    raise TypeError(
        "Data must be a pandas.DataFrame or a path to a CSV file. "
        f"Received type: {type(data)!r}."
    )


def _prepare_training_config(
    config: TrainingConfig | None,
    *,
    n_splits: int | None = None,
    random_state: int | None = None,
    lgbm_params: Dict[str, object] | None = None,
    early_stopping_rounds: int | None = None,
) -> TrainingConfig:
    """Create a :class:`TrainingConfig` instance with runtime overrides."""

    cfg = copy.deepcopy(config) if config is not None else TrainingConfig()

    if n_splits is not None:
        cfg.n_splits = n_splits
    if random_state is not None:
        cfg.random_state = random_state
    if early_stopping_rounds is not None:
        cfg.early_stopping_rounds = early_stopping_rounds

    if lgbm_params:
        cfg.lgbm_params.update(lgbm_params)

    return cfg


def train_regression_model(
    train_data: DataFrameLike,
    target_column: str,
    *,
    drop_columns: Sequence[str] | None = None,
    test_data: DataFrameLike | None = None,
    config: TrainingConfig | None = None,
    n_splits: int | None = None,
    random_state: int | None = None,
    lgbm_params: Dict[str, object] | None = None,
    early_stopping_rounds: int | None = None,
    logger: logging.Logger | None = None,
) -> TrainingResult:
    """Train a LightGBM regression model from CSV paths or dataframes."""

    train_df = _ensure_dataframe(train_data)

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the training data."
        )

    drop_columns = list(drop_columns or [])

    missing_drop_cols = [col for col in drop_columns if col not in train_df.columns]
    if missing_drop_cols:
        raise ValueError(
            "The following drop columns are not present in the training data: "
            + ", ".join(sorted(missing_drop_cols))
        )

    if test_data is not None:
        test_df = _ensure_dataframe(test_data)

        missing_in_test = [col for col in drop_columns if col not in test_df.columns]
        if missing_in_test:
            raise ValueError(
                "The following drop columns are not present in the test data: "
                + ", ".join(sorted(missing_in_test))
            )
    else:
        test_df = None

    feature_columns = [
        column
        for column in train_df.columns
        if column not in drop_columns and column != target_column
    ]

    if test_df is not None:
        missing_features = set(feature_columns) - set(test_df.columns)
        if missing_features:
            raise ValueError(
                "The following feature columns are missing in the test data: "
                + ", ".join(sorted(missing_features))
            )

    features = train_df[feature_columns]
    target = train_df[target_column]

    cfg = _prepare_training_config(
        config,
        n_splits=n_splits,
        random_state=random_state,
        lgbm_params=lgbm_params,
        early_stopping_rounds=early_stopping_rounds,
    )

    return train_with_cv(features, target, cfg, logger)
