import csv
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "row" / "NSRDB"
YEARS = range(2016, 2021)
INPUT_FILES = [
    DATA_DIR / "yearly_csv" / f"{year}-2412285-fixed_tilt.csv"
    for year in YEARS
]
OUTPUT_FILE = DATA_DIR / "NSRDB_2016_2020_combined.csv"
MALAYSIA_DATETIME_COLUMN = "Malaysia Datetime"


def combine_nsrdb_files() -> None:
    missing_files = [path for path in INPUT_FILES if not path.is_file()]
    if missing_files:
        missing_names = ", ".join(path.name for path in missing_files)
        raise FileNotFoundError(f"Missing NSRDB input file(s): {missing_names}")

    with INPUT_FILES[0].open(
        "r", newline="", encoding="utf-8-sig"
    ) as first_file:
        first_reader = csv.reader(first_file)
        metadata_header = next(first_reader, None)
        metadata_values = next(first_reader, None)
        data_header = next(first_reader, None)

    if metadata_header is None or metadata_values is None or data_header is None:
        raise ValueError(f"NSRDB file has incomplete metadata: {INPUT_FILES[0]}")

    if len(data_header) < 5 or data_header[:5] != [
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
    ]:
        raise ValueError(f"Unexpected time columns in {INPUT_FILES[0]}")

    output_header = [*data_header, MALAYSIA_DATETIME_COLUMN]
    timestamp_indexes = range(5)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=DATA_DIR,
            prefix=".NSRDB_2016_2020_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file)
            writer.writerow(metadata_header)
            writer.writerow(metadata_values)
            writer.writerow(output_header)

            previous_malaysia_time: datetime | None = None
            row_count = 0

            for input_file in INPUT_FILES:
                with input_file.open(
                    "r", newline="", encoding="utf-8-sig"
                ) as source:
                    reader = csv.reader(source)
                    source_metadata_header = next(reader, None)
                    source_metadata_values = next(reader, None)
                    source_data_header = next(reader, None)
                    if (
                        source_metadata_header != metadata_header
                        or source_metadata_values != metadata_values
                        or source_data_header != data_header
                    ):
                        raise ValueError(
                            f"NSRDB metadata or columns differ in {input_file}"
                        )

                    for line_number, row in enumerate(reader, start=4):
                        if len(row) != len(data_header):
                            raise ValueError(
                                f"Expected {len(data_header)} columns in "
                                f"{input_file} at line {line_number}, "
                                f"found {len(row)}"
                            )

                        try:
                            source_time = datetime(
                                *(int(row[index]) for index in timestamp_indexes)
                            )
                        except ValueError as error:
                            raise ValueError(
                                f"Invalid timestamp in {input_file} "
                                f"at line {line_number}"
                            ) from error

                        malaysia_time = source_time + timedelta(hours=8)
                        if (
                            previous_malaysia_time is not None
                            and malaysia_time < previous_malaysia_time
                        ):
                            raise ValueError(
                                f"Input rows are not chronological by Malaysia "
                                f"time at {input_file}:{line_number}"
                            )

                        writer.writerow([*row, malaysia_time.isoformat(sep=" ")])
                        previous_malaysia_time = malaysia_time
                        row_count += 1

        os.replace(temporary_path, OUTPUT_FILE)
        temporary_path = None
        print(f"Wrote {row_count} rows to {OUTPUT_FILE}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    combine_nsrdb_files()