"""Model inference utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def generate_predictions(
    model,
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    output_path: str | Path | None = None,
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Generate predictions and optionally persist them to disk."""

    missing_columns = set(feature_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(
            "The following required feature columns are missing from the input data: "
            + ", ".join(sorted(missing_columns))
        )

    features = data[list(feature_columns)]
    predictions = model.predict(features)
    result_df = data.copy()
    result_df[prediction_column] = predictions

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df[[prediction_column]].to_csv(output_path, index=False)

    return result_df
