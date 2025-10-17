"""LightGBM training with cross-validation."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


@dataclass
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


@dataclass
class FoldResult:
    fold: int
    train_mae: float
    valid_mae: float
    best_iteration: int


@dataclass
class TrainingResult:
    best_model: lgb.LGBMRegressor
    best_fold: int
    fold_results: List[FoldResult]
    feature_columns: List[str]


def _create_model(params: Dict[str, object]) -> lgb.LGBMRegressor:
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
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=[
                lgb.log_evaluation(period=50),
                lgb.early_stopping(stopping_rounds=config.early_stopping_rounds)
                if config.early_stopping_rounds
                else None,
            ],
        )

        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        train_mae = mean_absolute_error(y_train, train_pred)
        valid_mae = mean_absolute_error(y_valid, valid_pred)

        best_iteration = getattr(model, "best_iteration_", model.n_estimators)

        result = FoldResult(
            fold=fold, train_mae=train_mae, valid_mae=valid_mae, best_iteration=best_iteration
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
