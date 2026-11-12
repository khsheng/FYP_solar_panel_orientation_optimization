"""Train XGBoost regressors for DNI and DHI using matched NSRDB/CLP data."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "combined"
    / "NSRDB_CLP_2016_2020_combined.csv"
)
MODEL_DIR = Path(__file__).resolve().parent / "models"
TARGET_COLUMNS = ("DNI", "DHI")
EXCLUDED_FEATURE_COLUMNS = {"Datetime", "DNI", "DHI", "GHI"}
TEST_FRACTION = 0.2
NUM_BOOST_ROUND = 300


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load matched data and separate numeric features from DNI/DHI labels."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = {"Datetime", *TARGET_COLUMNS}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {DATA_PATH.name}: "
            f"{sorted(missing_columns)}"
        )

    timestamps = pd.to_datetime(data["Datetime"], errors="raise", utc=True)
    if timestamps.isna().any():
        raise ValueError("Training dataset contains missing Datetime values.")
    if timestamps.duplicated().any():
        raise ValueError("Training dataset contains duplicate timestamps.")

    data = data.assign(_parsed_datetime=timestamps).sort_values(
        "_parsed_datetime", kind="stable"
    )
    for target in TARGET_COLUMNS:
        data[target] = pd.to_numeric(data[target], errors="raise")

    feature_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED_FEATURE_COLUMNS and column != "_parsed_datetime"
    ]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding labels and GHI.")

    features = data[feature_columns]
    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "XGBoost features must be numeric; "
            f"found non-numeric columns: {non_numeric_columns}"
        )
    if np.isinf(features.to_numpy(dtype=np.float64, na_value=np.nan)).any():
        raise ValueError("Training features contain infinite values.")

    labels = data.loc[:, list(TARGET_COLUMNS)]
    if not np.isfinite(labels.to_numpy(dtype=np.float64)).all():
        raise ValueError("Training labels contain missing or infinite DNI/DHI values.")

    return features, labels, data["_parsed_datetime"]


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, float, float]:
    """Return mean absolute error, root mean squared error, and R-squared."""
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    total_sum_of_squares = float(np.sum(np.square(actual - np.mean(actual))))
    r_squared = (
        float(1 - np.sum(np.square(errors)) / total_sum_of_squares)
        if total_sum_of_squares
        else float("nan")
    )
    return mae, rmse, r_squared


def train_models() -> None:
    """Train one regressor per label, evaluate on the latest 20% of timestamps."""
    features, labels, timestamps = load_training_data()
    split_index = int(len(features) * (1 - TEST_FRACTION))
    if split_index < 1 or split_index >= len(features):
        raise ValueError(
            f"Need at least two rows to split the dataset; found {len(features)}."
        )

    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {DATA_PATH}")
    print(
        f"Rows: {len(features):,} "
        f"({len(train_features):,} train, {len(test_features):,} test)"
    )
    print(f"Features ({len(features.columns)}): {', '.join(features.columns)}")
    print(f"Test period: {timestamps.iloc[split_index]} to {timestamps.iloc[-1]}")

    parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    for target in TARGET_COLUMNS:
        train_matrix = xgb.DMatrix(
            train_features,
            label=labels[target].iloc[:split_index],
            feature_names=features.columns.tolist(),
        )
        test_matrix = xgb.DMatrix(
            test_features,
            label=labels[target].iloc[split_index:],
            feature_names=features.columns.tolist(),
        )
        model = xgb.train(
            parameters,
            train_matrix,
            num_boost_round=NUM_BOOST_ROUND,
        )

        predictions = model.predict(test_matrix)
        actual = labels[target].iloc[split_index:].to_numpy(dtype=np.float64)
        mae, rmse, r_squared = regression_metrics(actual, predictions)

        model_path = MODEL_DIR / f"{target.lower()}_xgb.json"
        model.save_model(model_path)
        print(
            f"{target}: MAE={mae:.3f}, RMSE={rmse:.3f}, "
            f"R2={r_squared:.3f}; saved {model_path}"
        )


if __name__ == "__main__":
    train_models()