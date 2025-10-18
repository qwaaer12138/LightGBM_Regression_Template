from pathlib import Path

import lightgbm as lgb
import pandas as pd

from lightgbm_regression import train_regression_model


def test_train_regression_model_on_boston_dataset():
    data_path = Path(__file__).resolve().parent / "data" / "boston_housing.csv"
    df = pd.read_csv(data_path)

    result = train_regression_model(
        df,
        target_column="medv",
        n_splits=3,
        random_state=0,
        lgbm_params={
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.2,
            "num_leaves": 15,
            "max_depth": -1,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 0,
        },
        early_stopping_rounds=10,
    )

    assert result.best_fold in {1, 2, 3}
    assert len(result.fold_results) == 3
    assert set(result.feature_columns) == set(df.columns) - {"medv"}
    assert isinstance(result.best_model, lgb.LGBMRegressor)
    for fold_result in result.fold_results:
        assert fold_result.train_mae > 0
        assert fold_result.valid_mae > 0
        assert fold_result.best_iteration > 0
