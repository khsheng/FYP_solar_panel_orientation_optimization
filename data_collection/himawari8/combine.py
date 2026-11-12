import csv
import os
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "HIMAWARI_DATA_ROOT",
        str(PROJECT_ROOT / "dataset" / "row" / "himawari8"),
    )
).expanduser()

if not DATA_ROOT.is_absolute():
    DATA_ROOT = SCRIPT_DIR / DATA_ROOT

CLP_DIR = DATA_ROOT.resolve() / "CLP"
INPUT_DIR = CLP_DIR / "monthly_csv"
SORT_COLUMN = "malaysia_datetime"
OUTPUT_GROUPS = {
    "CLP_2016_2020.csv": set(range(2016, 2021)),
    "CLP_2026.csv": {2026},
}


def combine_monthly_csv() -> None:
    csv_files = sorted(INPUT_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    header: list[str] | None = None
    sort_index: int | None = None
    records: dict[str, list[tuple[datetime, list[str]]]] = {
        filename: [] for filename in OUTPUT_GROUPS
    }

    for csv_file in csv_files:
        with csv_file.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source)
            file_header = next(reader, None)
            if file_header is None:
                raise ValueError(f"CSV file is empty: {csv_file}")

            if header is None:
                header = file_header
                if SORT_COLUMN not in header:
                    raise ValueError(
                        f"Missing {SORT_COLUMN!r} column in {csv_file}"
                    )
                sort_index = header.index(SORT_COLUMN)
            elif file_header != header:
                raise ValueError(f"CSV columns do not match in {csv_file}")

            if sort_index is None:
                raise RuntimeError("No sort column index was found")

            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"Expected {len(header)} columns in {csv_file} "
                        f"at line {line_number}, found {len(row)}"
                    )

                try:
                    malaysia_time = datetime.fromisoformat(row[sort_index])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid {SORT_COLUMN} in {csv_file} "
                        f"at line {line_number}: {row[sort_index]!r}"
                    ) from error

                for filename, years in OUTPUT_GROUPS.items():
                    if malaysia_time.year in years:
                        records[filename].append((malaysia_time, row))

    if header is None or sort_index is None:
        raise RuntimeError("No CSV header was read")

    CLP_DIR.mkdir(parents=True, exist_ok=True)
    for filename, group_records in records.items():
        group_records.sort(key=lambda record: record[0])
        output_file = CLP_DIR / filename
        with output_file.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(header)
            writer.writerows(row for _, row in group_records)

        print(
            f"Combined {len(group_records)} rows from {len(csv_files)} files "
            f"into {output_file}"
        )


if __name__ == "__main__":
    combine_monthly_csv()