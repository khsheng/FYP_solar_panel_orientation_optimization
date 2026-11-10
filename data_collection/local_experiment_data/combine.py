import json
import pandas as pd
from pathlib import Path


# =========================================================
# 1. CONFIGURATION
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "row" / "local_experiment_data"
INPUT_DIR = DATA_DIR / "daily_ndjson"

OUTPUT_FILE = DATA_DIR / "local_experiment_data_combined.csv"


# =========================================================
# 2. LOAD ALL NDJSON FILES
# =========================================================

all_data = []

files = sorted(INPUT_DIR.glob("data_*.ndjson"))

print("=" * 70)
print("COMBINING LOCAL EXPERIMENT DATA")
print("=" * 70)

print(f"\nFiles found: {len(files)}")

for file_path in files:

    file_count = 0

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                all_data.append(record)
                file_count += 1

            except json.JSONDecodeError:
                print(
                    f"Invalid JSON skipped: "
                    f"{file_path.name}"
                )

    print(
        f"{file_path.name}: "
        f"{file_count:,} records"
    )


# =========================================================
# 3. CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(all_data)


# =========================================================
# 4. SORT BY TIMESTAMP
# =========================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


# =========================================================
# 5. SAVE AS CSV
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 6. PRINT SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("COMBINATION SUMMARY")
print("=" * 70)

print(f"\nTotal files combined: {len(files):,}")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")

print("\nColumns:")
print(df.columns.tolist())

if "timestamp" in df.columns:

    print("\nTimestamp range:")

    print(
        f"Start: {df['timestamp'].min()}"
    )

    print(
        f"End:   {df['timestamp'].max()}"
    )

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nCombination completed.")