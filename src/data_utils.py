"""Utilities for loading and preparing tabular datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass
class DatasetBundle:
    """Container for train/test datasets and metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: List[str]
    target_column: str


def load_datasets(
    train_path: str | Path,
    test_path: str | Path,
    target_column: str,
    drop_columns: List[str] | None = None,
) -> DatasetBundle:
    """Load training and test data from disk.

    Parameters
    ----------
    train_path: str or Path
        File path to the training dataset containing the target column.
    test_path: str or Path
        File path to the test dataset without the target column.
    target_column: str
        Name of the target column in the training dataset.
    drop_columns: List[str], optional
        Additional columns to drop from both datasets (e.g., identifiers).

    Returns
    -------
    DatasetBundle
        Dataclass that includes training and test dataframes with aligned
        feature columns and the target column name.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in training data."
        )

    drop_columns = drop_columns or []

    for column in drop_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Drop column '{column}' not found in training data columns."
            )
        if column not in test_df.columns:
            raise ValueError(
                f"Drop column '{column}' not found in test data columns."
            )

    # Ensure columns are sorted consistently after dropping identifiers
    train_features = train_df.drop(columns=[target_column] + drop_columns)
    test_features = test_df.drop(columns=drop_columns)

    missing_in_test = set(train_features.columns) - set(test_features.columns)
    if missing_in_test:
        raise ValueError(
            "The following feature columns are missing in the test data: "
            + ", ".join(sorted(missing_in_test))
        )

    # Align column order
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
