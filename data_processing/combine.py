"""Join processed weather, satellite, and experiment datasets by datetime."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
TIMEZONE = "Asia/Kuala_Lumpur"
DatasetSource = tuple[str, str, str]


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
    sources: tuple[DatasetSource, ...],
    output_filename: str,
    report_filename: str,
) -> None:
    """Write an exact inner join and report timestamps not shared by all inputs."""
    if len(sources) < 2:
        raise ValueError("At least two datasets are required for a datetime join.")

    loaded_sources = [
        (filename, tag, load_processed_csv(filename, datetime_column))
        for filename, datetime_column, tag in sources
    ]
    timestamp_sets = [
        set(data["Datetime"]) for _, _, data in loaded_sources
    ]
    matched_timestamps = set.intersection(*timestamp_sets)

    combined = loaded_sources[0][2]
    for _, tag, data in loaded_sources[1:]:
        combined = combined.merge(
            data,
            on="Datetime",
            how="inner",
            validate="one_to_one",
            sort=True,
            suffixes=("", f"_{tag}"),
        )

    output_path = PROCESSED_DIR / "combined" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    unmatched_rows = []
    for index, (filename, _, _) in enumerate(loaded_sources):
        other_timestamp_sets = [
            timestamps
            for other_index, timestamps in enumerate(timestamp_sets)
            if other_index != index
        ]
        timestamps_missing_from_others = sorted(
            timestamp_sets[index] - set.intersection(*other_timestamp_sets)
        )
        unmatched_rows.extend(
            {
                "Datetime": timestamp,
                "dataset_without_match": filename,
            }
            for timestamp in timestamps_missing_from_others
        )

    unmatched_report = pd.DataFrame(
        unmatched_rows,
        columns=["Datetime", "dataset_without_match"],
    )
    report_path = PROCESSED_DIR / "combined" / report_filename
    unmatched_report.to_csv(report_path, index=False)

    print(f"\n{' + '.join(filename for filename, _, _ in loaded_sources)}")
    for (filename, _, data), timestamps in zip(
        loaded_sources, timestamp_sets, strict=True
    ):
        print(f"  Rows in {filename}: {len(data):,}")
        print(
            f"  Timestamps not shared by all other inputs: "
            f"{len(timestamps - matched_timestamps):,}"
        )
    print(f"  Matched timestamps: {len(matched_timestamps):,}")
    print(f"  Combined data: {output_path}")
    print(f"  Unmatched timestamp report: {report_path}")


def main() -> None:
    combine_on_datetime(
        sources=(
            ("NSRDB_processed.csv", "Datetime", "NSRDB"),
            ("CLP_2016_2020_processed.csv", "malaysia_datetime", "CLP"),
            ("ARP_2016_2020_processed.csv", "malaysia_datetime", "ARP"),
        ),
        output_filename="NSRDB_CLP_2016_2020_combined.csv",
        report_filename="NSRDB_CLP_2016_2020_unmatched_timestamps.csv",
    )
    combine_on_datetime(
        sources=(
            ("experiment_processed.csv", "datetime_malaysia", "experiment"),
            ("CLP_2026_processed.csv", "malaysia_datetime", "CLP"),
            ("ARP_2026_processed.csv", "malaysia_datetime", "ARP"),
        ),
        output_filename="experiment_CLP_2026_combined.csv",
        report_filename="experiment_CLP_2026_unmatched_timestamps.csv",
    )


if __name__ == "__main__":
    main()
