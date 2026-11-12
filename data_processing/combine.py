"""Join processed weather and experiment datasets on Malaysia-local datetime."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
TIMEZONE = "Asia/Kuala_Lumpur"


def load_processed_csv(filename: str, datetime_column: str) -> pd.DataFrame:
    """Load a processed CSV and normalize its datetime key to Malaysia time."""
    path = PROCESSED_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    data = pd.read_csv(path)
    if datetime_column not in data.columns:
        raise ValueError(
            f"Expected datetime column {datetime_column!r} in {path.name}; "
            f"found {data.columns.tolist()}"
        )

    timestamps = pd.to_datetime(data.pop(datetime_column), errors="raise", utc=True)
    data.insert(0, "Datetime", timestamps.dt.tz_convert(TIMEZONE))
    if data["Datetime"].duplicated().any():
        duplicate_count = int(data["Datetime"].duplicated(keep=False).sum())
        raise ValueError(
            f"{path.name} has {duplicate_count} rows with duplicate timestamps; "
            "cannot perform a one-to-one join."
        )
    return data.sort_values("Datetime", kind="stable").reset_index(drop=True)


def combine_on_datetime(
    left_name: str,
    left_datetime_column: str,
    right_name: str,
    right_datetime_column: str,
    output_filename: str,
    report_filename: str,
) -> None:
    """Write an exact inner join and report timestamps absent from either input."""
    left = load_processed_csv(left_name, left_datetime_column)
    right = load_processed_csv(right_name, right_datetime_column)

    left_timestamps = set(left["Datetime"])
    right_timestamps = set(right["Datetime"])
    matched_timestamps = left_timestamps & right_timestamps
    unmatched_left = sorted(left_timestamps - right_timestamps)
    unmatched_right = sorted(right_timestamps - left_timestamps)

    combined = left.merge(
        right,
        on="Datetime",
        how="inner",
        validate="one_to_one",
        sort=True,
    )
    output_path = PROCESSED_DIR / "combined" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    unmatched_report = pd.DataFrame(
        [
            {"Datetime": timestamp, "dataset_without_match": left_name}
            for timestamp in unmatched_left
        ]
        + [
            {"Datetime": timestamp, "dataset_without_match": right_name}
            for timestamp in unmatched_right
        ],
        columns=["Datetime", "dataset_without_match"],
    )
    report_path = PROCESSED_DIR / "combined" / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_report.to_csv(report_path, index=False)

    print(f"\n{left_name} + {right_name}")
    print(f"  Rows in each input: {len(left):,} + {len(right):,}")
    print(f"  Matched timestamps: {len(matched_timestamps):,}")
    print(f"  Unmatched {left_name}: {len(unmatched_left):,}")
    print(f"  Unmatched {right_name}: {len(unmatched_right):,}")
    print(f"  Combined data: {output_path}")
    print(f"  Unmatched timestamp report: {report_path}")
    for dataset_name, timestamps in (
        (left_name, unmatched_left),
        (right_name, unmatched_right),
    ):
        if timestamps:
            print(
                f"  {dataset_name} unmatched range: "
                f"{timestamps[0]} to {timestamps[-1]}"
            )
        else:
            print(f"  {dataset_name} unmatched range: none")


def main() -> None:
    combine_on_datetime(
        left_name="NSRDB_processed.csv",
        left_datetime_column="Datetime",
        right_name="CLP_2016_2020_processed.csv",
        right_datetime_column="malaysia_datetime",
        output_filename="NSRDB_CLP_2016_2020_combined.csv",
        report_filename="NSRDB_CLP_2016_2020_unmatched_timestamps.csv",
    )
    combine_on_datetime(
        left_name="experiment_processed.csv",
        left_datetime_column="datetime_malaysia",
        right_name="CLP_2026_processed.csv",
        right_datetime_column="malaysia_datetime",
        output_filename="experiment_CLP_2026_combined.csv",
        report_filename="experiment_CLP_2026_unmatched_timestamps.csv",
    )




if __name__ == "__main__":
    main()