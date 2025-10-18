"""Utilities for loading and preparing tabular datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from ._typing import PathLike


@dataclass(slots=True)
class DatasetBundle:
    """Container for train/test datasets and metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: List[str]
    target_column: str


def load_datasets(
    train_path: PathLike,
    test_path: PathLike,
    target_column: str,
    drop_columns: Sequence[str] | None = None,
) -> DatasetBundle:
    """Load training and test data from disk."""

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training data."
        )

    drop_columns = list(drop_columns or [])

    for column in drop_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Drop column '{column}' not found in training data columns."
            )
        if column not in test_df.columns:
            raise ValueError(
                f"Drop column '{column}' not found in test data columns."
            )

    train_features = train_df.drop(columns=[target_column] + drop_columns)
    test_features = test_df.drop(columns=drop_columns)

    missing_in_test = set(train_features.columns) - set(test_features.columns)
    if missing_in_test:
        raise ValueError(
            "The following feature columns are missing in the test data: "
            + ", ".join(sorted(missing_in_test))
        )

    test_features = test_features[train_features.columns]

    return DatasetBundle(
        train=train_df,
        test=test_df,
        feature_columns=list(train_features.columns),
        target_column=target_column,
    )


def split_features_and_target(
    dataset: DatasetBundle,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target from the training dataset."""

    features = dataset.train[dataset.feature_columns]
    target = dataset.train[dataset.target_column]
    return features, target
