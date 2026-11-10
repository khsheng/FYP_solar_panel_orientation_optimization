import argparse
import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import shutil


# =========================================================
# 1. CONFIGURATION
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
# Override with HIMAWARI_DATA_ROOT; relative paths resolve from this directory.
DATA_ROOT = Path(
    os.environ.get(
        "HIMAWARI_DATA_ROOT",
        str(PROJECT_ROOT / "dataset" / "row" / "himawari8")
    )
).expanduser()

if not DATA_ROOT.is_absolute():
    DATA_ROOT = SCRIPT_DIR / DATA_ROOT

DATA_ROOT = DATA_ROOT.resolve()

YEAR = 2020

MONTHS = [
    1
]

TARGET_LAT = 2.4318056
TARGET_LON = 102.6813056

VARIABLES_CLP = [
    "Hour",
    "CLOT",
    "CLTT",
    "CLTH",
    "CLER_23",
    "CLTYPE",
    "QA"
]

VARIABLES_ARP = [
    "Hour",
    "AOT",
    "AOT_uncertainty",
    "QA"
]

PRODUCT_VARIABLES = {
    "CLP": VARIABLES_CLP,
    "ARP": VARIABLES_ARP
}

SMALL_NC_FILE_SIZE_BYTES = 10 * 1024


def normalize_product(product):

    product = product.strip().upper()

    if product not in PRODUCT_VARIABLES:

        raise ValueError(
            f"Unsupported product {product!r}. Choose CLP or ARP."
        )

    return product


def validate_csv(
    csv_file,
    expected_rows,
    expected_columns
):

    df = pd.read_csv(
        csv_file
    )

    if len(df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} rows in {csv_file.name}, "
            f"found {len(df)}."
        )

    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"Missing columns in {csv_file.name}: "
            f"{', '.join(sorted(missing_columns))}."
        )

    if "timestamp" not in df.columns:

        raise ValueError(
            f"Missing timestamp column in {csv_file.name}."
        )

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if timestamps.isna().any():

        raise ValueError(
            f"Invalid or missing timestamps in {csv_file.name}."
        )


# =========================================================
# 2. OUTPUT DIRECTORIES
# =========================================================

# =========================================================
# 3. PROCESS ONE NC FILE
# =========================================================

def read_nc_file(file_path, variables=VARIABLES_CLP):

    parts = file_path.name.split("_")

    try:

        date_str = parts[2]
        time_str = parts[3]

        # -------------------------------------------------
        # Himawari filename timestamp is UTC
        # -------------------------------------------------

        timestamp = datetime.strptime(
            date_str + time_str,
            "%Y%m%d%H%M"
        )

        # -------------------------------------------------
        # Convert UTC to Malaysia Time (UTC+8)
        # -------------------------------------------------

        malaysia_datetime = (
            timestamp + timedelta(hours=8)
        )

    except Exception as e:

        raise ValueError(
            f"Cannot parse timestamp from filename: {e}"
        )


    try:

        ds = xr.open_dataset(
            file_path,
            engine="netcdf4"
        )


        # -------------------------------------------------
        # Find nearest pixel
        # -------------------------------------------------

        point = ds.sel(
            latitude=TARGET_LAT,
            longitude=TARGET_LON,
            method="nearest"
        )


        actual_lat = float(
            point.latitude.values
        )

        actual_lon = float(
            point.longitude.values
        )


        # -------------------------------------------------
        # Create record
        # -------------------------------------------------

        record = {

            # Original Himawari UTC time
            "timestamp": timestamp,

            # Converted Malaysia time
            "malaysia_datetime": malaysia_datetime,

            "latitude": actual_lat,

            "longitude": actual_lon
        }


        # -------------------------------------------------
        # Extract required variables
        # -------------------------------------------------

        for variable in variables:

            if variable not in point:

                record[variable] = np.nan

                continue


            try:

                value = point[variable].values


                if np.size(value) == 1:

                    value = float(value)


                    if not np.isfinite(value):

                        value = np.nan

                else:

                    value = np.nan


                record[variable] = value


            except Exception:

                record[variable] = np.nan


        ds.close()


        return record


    except Exception as e:

        raise RuntimeError(
            f"{type(e).__name__}: {e}"
        )


def make_nan_record(file_path, variables):

    parts = file_path.name.split("_")
    timestamp = datetime.strptime(
        parts[2] + parts[3],
        "%Y%m%d%H%M"
    )

    return {
        "timestamp": timestamp,
        "malaysia_datetime": timestamp + timedelta(hours=8),
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,
        **{variable: np.nan for variable in variables}
    }


# =========================================================
# 4. PROCESS ONE DAY
# =========================================================

def process_day(
    year,
    month,
    day,
    data_dir,
    product="CLP",
    variables=None
):

    product = normalize_product(product)

    if variables is None:

        variables = PRODUCT_VARIABLES[product]

    month_str = f"{month:02d}"

    day_str = f"{day:02d}"


    day_dir = (
        data_dir
        / str(year)
        / month_str
        / day_str
    )


    daily_csv = (
        day_dir
        / f"{product}_{year}{month_str}{day_str}.csv"
    )


    # -----------------------------------------------------
    # Check day directory
    # -----------------------------------------------------

    if not day_dir.exists():

        return {

            "day": day,

            "status": "SKIPPED",

            "records": 0,

            "errors": [],

            "csv": None,

            "reason": "Day directory does not exist"
        }


    # -----------------------------------------------------
    # Find NC files
    # -----------------------------------------------------

    nc_files = sorted(
        day_dir.glob("*/*.nc")
    )


    if not nc_files:

        return {

            "day": day,

            "status": "SKIPPED",

            "records": 0,

            "errors": [],

            "csv": None,

            "reason": "No NC files found"
        }


    records = []

    errors = []
    warnings = []


    # -----------------------------------------------------
    # Read every NC file
    # -----------------------------------------------------

    for file_path in nc_files:

        try:

            record = read_nc_file(
                file_path,
                variables
            )

            records.append(
                record
            )


        except Exception as e:

            error_message = str(e)

            if (
                "KeyError:" in error_message
                and "'latitude' is not a valid dimension or coordinate"
                in error_message
            ):

                try:

                    file_size = file_path.stat().st_size

                except OSError as stat_error:

                    errors.append(
                        f"{file_path.name}: {error_message}; "
                        f"could not check file size: "
                        f"{type(stat_error).__name__}: {stat_error}"
                    )

                    continue


                if file_size <= SMALL_NC_FILE_SIZE_BYTES:

                    records.append(
                        make_nan_record(file_path, variables)
                    )

                    warnings.append(
                        f"{file_path.name}: unavailable NetCDF "
                        f"(latitude coordinate missing, "
                        f"{file_size} bytes <= "
                        f"{SMALL_NC_FILE_SIZE_BYTES} bytes); "
                        f"NaN row written"
                    )

                    continue

                errors.append(
                    f"{file_path.name}: {error_message}; "
                    f"file size {file_size} bytes exceeds "
                    f"{SMALL_NC_FILE_SIZE_BYTES} bytes"
                )

                continue

            errors.append(
                f"{file_path.name}: {e}"
            )


    # -----------------------------------------------------
    # No valid records
    # -----------------------------------------------------

    if not records:

        return {

            "day": day,

            "status": "FAILED",

            "records": 0,

            "errors": errors,

            "csv": None,

            "reason": "No valid NC files could be read"
        }


    # -----------------------------------------------------
    # Create daily DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(
        records
    )


    df = df.sort_values(
        by="timestamp"
    ).reset_index(
        drop=True
    )


    # -----------------------------------------------------
    # Save daily CSV
    # -----------------------------------------------------

    try:

        df.to_csv(
            daily_csv,
            index=False
        )

        validate_csv(
            daily_csv,
            len(df),
            df.columns
        )

    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:

        errors.append(
            f"{daily_csv.name}: CSV write/validation failed: "
            f"{type(e).__name__}: {e}"
        )

        return {

            "day": day,

            "status": "FAILED",

            "records": len(records),

            "errors": errors,

            "csv": None,

            "reason": "Daily CSV validation failed"
        }


    # -----------------------------------------------------
    # Day status
    # -----------------------------------------------------

    if errors:

        status = "FAILED"

    else:

        status = "SUCCESS"


    return {

        "day": day,

        "status": status,

        "records": len(df),

        "errors": errors,

        "warnings": warnings,

        "csv": daily_csv,

        "reason": None
    }


# =========================================================
# 5. COMBINE DAILY CSV FILES INTO MONTHLY CSV
# =========================================================

def combine_month(
    year,
    month,
    results,
    data_dir,
    product="CLP"
):

    product = normalize_product(product)

    month_str = f"{month:02d}"


    # -----------------------------------------------------
    # New monthly output folder
    # -----------------------------------------------------

    monthly_dir = data_dir / "monthly_csv" / str(year)


    monthly_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    monthly_csv = (
        monthly_dir
        / f"{product}_{year}{month_str}.csv"
    )


    daily_csv_files = []


    # -----------------------------------------------------
    # Collect successfully created daily CSV files
    # -----------------------------------------------------

    for result in results:

        csv_file = result.get(
            "csv"
        )


        if (
            csv_file is not None
            and csv_file.exists()
        ):

            daily_csv_files.append(
                csv_file
            )


    # -----------------------------------------------------
    # No daily CSV
    # -----------------------------------------------------

    if not daily_csv_files:

        return {

            "status": "SKIPPED",

            "daily_csv_count": 0,

            "rows": 0,

            "file": None,

            "error": "No daily CSV files available"
        }


    monthly_data = []

    errors = []


    # -----------------------------------------------------
    # Read daily CSV files
    # -----------------------------------------------------

    for csv_file in sorted(
        daily_csv_files
    ):

        try:

            df = pd.read_csv(
                csv_file
            )

            monthly_data.append(
                df
            )


        except Exception as e:

            errors.append(
                f"{csv_file.name}: "
                f"{type(e).__name__}: {e}"
            )


    # -----------------------------------------------------
    # No valid daily CSV
    # -----------------------------------------------------

    if not monthly_data:

        return {

            "status": "FAILED",

            "daily_csv_count": len(
                daily_csv_files
            ),

            "rows": 0,

            "file": None,

            "error": "; ".join(
                errors
            )
        }


    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    monthly_df = pd.concat(
        monthly_data,
        ignore_index=True
    )


    # -----------------------------------------------------
    # Sort by UTC timestamp
    # -----------------------------------------------------

    if "timestamp" in monthly_df.columns:

        monthly_df["timestamp"] = pd.to_datetime(
            monthly_df["timestamp"],
            errors="coerce"
        )


        monthly_df = monthly_df.sort_values(
            by="timestamp"
        ).reset_index(
            drop=True
        )


    # -----------------------------------------------------
    # Make sure Malaysia datetime is also datetime
    # -----------------------------------------------------

    if "malaysia_datetime" in monthly_df.columns:

        monthly_df["malaysia_datetime"] = pd.to_datetime(
            monthly_df["malaysia_datetime"],
            errors="coerce"
        )


    # -----------------------------------------------------
    # Save monthly CSV
    # -----------------------------------------------------

    try:

        monthly_df.to_csv(
            monthly_csv,
            index=False
        )

        validate_csv(
            monthly_csv,
            len(monthly_df),
            monthly_df.columns
        )

    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:

        return {

            "status": "FAILED",

            "daily_csv_count": len(
                daily_csv_files
            ),

            "rows": len(
                monthly_df
            ),

            "file": monthly_csv,

            "error": (
                f"{type(e).__name__}: {e}"
            )
        }


    if errors:

        status = "PARTIAL"

    else:

        status = "SUCCESS"


    return {

        "status": status,

        "daily_csv_count": len(
            daily_csv_files
        ),

        "rows": len(
            monthly_df
        ),

        "file": monthly_csv,

        "error": (
            "; ".join(errors)
            if errors
            else None
        )
    }


# =========================================================
# 6. WRITE MONTHLY LOG
# =========================================================

def write_log(
    year,
    month,
    results,
    monthly_result,
    data_dir
):

    month_str = f"{month:02d}"


    log_file = (
        data_dir
        / "conversion_logs"
        / str(year)
        / f"{month_str}.txt"
    )

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    total_days = len(
        results
    )


    success_count = sum(
        1
        for r in results
        if r["status"] == "SUCCESS"
    )


    failed_count = sum(
        1
        for r in results
        if r["status"] == "FAILED"
    )


    skipped_count = sum(
        1
        for r in results
        if r["status"] == "SKIPPED"
    )


    with open(
        log_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "HIMAWARI NC TO CSV CONVERSION LOG\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )


        f.write(
            f"Year  : {year}\n"
        )

        f.write(
            f"Month : {month_str}\n"
        )

        f.write(
            f"Target Latitude  : {TARGET_LAT}\n"
        )

        f.write(
            f"Target Longitude : {TARGET_LON}\n"
        )

        f.write(
            "Original Time Zone : UTC\n"
        )

        f.write(
            "Malaysia Time      : UTC+8\n"
        )

        f.write("\n")


        # -------------------------------------------------
        # Daily conversion summary
        # -------------------------------------------------

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "DAILY CONVERSION SUMMARY\n"
        )

        f.write(
            "-" * 70 + "\n"
        )


        f.write(
            f"Total days : {total_days}\n"
        )

        f.write(
            f"Success    : {success_count}\n"
        )

        f.write(
            f"Failed     : {failed_count}\n"
        )

        f.write(
            f"Skipped    : {skipped_count}\n"
        )

        f.write("\n")


        # -------------------------------------------------
        # Daily status
        # -------------------------------------------------

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "DAILY STATUS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )


        for result in sorted(
            results,
            key=lambda x: x["day"]
        ):

            day_str = (
                f"{result['day']:02d}"
            )


            f.write(
                f"{year}-{month_str}-{day_str} : "
                f"{result['status']}"
            )


            if result["records"]:

                f.write(
                    f" | Records: "
                    f"{result['records']}"
                )


            if result.get("reason"):

                f.write(
                    f" | Reason: "
                    f"{result['reason']}"
                )


            f.write("\n")


        # -------------------------------------------------
        # Failed NC files
        # -------------------------------------------------

        failed_results = [

            r
            for r in results
            if r["errors"]
        ]


        if failed_results:

            f.write("\n")

            f.write(
                "-" * 70 + "\n"
            )

            f.write(
                "FAILED NC FILES\n"
            )

            f.write(
                "-" * 70 + "\n"
            )


            for result in sorted(
                failed_results,
                key=lambda x: x["day"]
            ):

                day_str = (
                    f"{result['day']:02d}"
                )


                f.write(
                    f"\n"
                    f"{year}-{month_str}-{day_str}\n"
                )


                for error in result["errors"]:

                    f.write(
                        f"    {error}\n"
                    )

        warning_results = [
            result
            for result in results
            if result.get("warnings")
        ]

        if warning_results:

            f.write("\n")

            f.write(
                "-" * 70 + "\n"
            )

            f.write(
                "EXPECTED MISSING NC DATA (NaN ROWS WRITTEN)\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            for result in sorted(
                warning_results,
                key=lambda x: x["day"]
            ):

                day_str = f"{result['day']:02d}"

                f.write(
                    f"\n{year}-{month_str}-{day_str}\n"
                )

                for warning in result["warnings"]:

                    f.write(
                        f"    {warning}\n"
                    )


        # -------------------------------------------------
        # Monthly combination
        # -------------------------------------------------

        f.write("\n")

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "MONTHLY CSV COMBINATION\n"
        )

        f.write(
            "-" * 70 + "\n"
        )


        f.write(
            f"Daily CSV files combined : "
            f"{monthly_result['daily_csv_count']}\n"
        )


        f.write(
            f"Total rows combined      : "
            f"{monthly_result['rows']}\n"
        )


        f.write(
            f"Monthly CSV status        : "
            f"{monthly_result['status']}\n"
        )


        if monthly_result["file"]:

            f.write(
                f"Monthly CSV file         : "
                f"{monthly_result['file']}\n"
            )


        if monthly_result["error"]:

            f.write(
                f"Combination error        : "
                f"{monthly_result['error']}\n"
            )

        f.write("\n")

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "MONTH VALIDATION AND NC CLEANUP\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Month validation status   : "
            f"{monthly_result.get('month_validation', 'NOT RUN')}\n"
        )

        f.write(
            f"Expected month rows       : "
            f"{monthly_result.get('expected_month_rows', 0)}\n"
        )

        f.write(
            f"Source NC files found     : "
            f"{monthly_result.get('source_nc_files_found', 0)}\n"
        )

        f.write(
            f"Source NC files deleted   : "
            f"{monthly_result.get('source_nc_files_deleted', 0)}\n"
        )

        f.write(
            f"Source month folder removed: "
            f"{monthly_result.get('source_month_directory_removed', False)}\n"
        )

        f.write(
            f"NC cleanup status         : "
            f"{monthly_result.get('cleanup_status', 'NOT RUN')}\n"
        )

        for cleanup_error in monthly_result.get("cleanup_errors", []):

            f.write(
                f"NC cleanup error          : "
                f"{cleanup_error}\n"
            )


        f.write("\n")

        f.write(
            "=" * 70 + "\n"
        )


# =========================================================
# 7. PROCESS ONE MONTH
# =========================================================

def process_month(
    year,
    month,
    data_dir=None,
    product="CLP"
):

    product = normalize_product(product)
    variables = PRODUCT_VARIABLES[product]

    if data_dir is None:

        data_dir = DATA_ROOT / product

    month_str = f"{month:02d}"


    days_in_month = calendar.monthrange(
        year,
        month
    )[1]


    print()

    print(
        f"========== {year}-{month_str} =========="
    )


    results = []


    # -----------------------------------------------------
    # Process every day
    # -----------------------------------------------------

    for day in range(
        1,
        days_in_month + 1
    ):

        result = process_day(
            year,
            month,
            day,
            data_dir,
            product,
            variables
        )


        results.append(
            result
        )


        day_str = f"{day:02d}"


        print(
            f"{year}-{month_str}-{day_str} : "
            f"{result['status']}"
        )


    # -----------------------------------------------------
    # Combine daily CSVs
    # -----------------------------------------------------

    print(
        f"\nCombining daily CSVs for "
        f"{year}-{month_str}..."
    )


    monthly_result = combine_month(
        year,
        month,
        results,
        data_dir,
        product
    )


    if monthly_result["status"] == "SUCCESS":

        print(
            f"Monthly CSV : "
            f"{monthly_result['file']}"
        )

        print(
            f"Daily CSVs  : "
            f"{monthly_result['daily_csv_count']}"
        )

        print(
            f"Rows        : "
            f"{monthly_result['rows']}"
        )


    elif monthly_result["status"] == "PARTIAL":

        print(
            "Monthly CSV created with errors"
        )

        print(
            f"Monthly CSV : "
            f"{monthly_result['file']}"
        )

    elif monthly_result["status"] == "FAILED":

        print(
            "Monthly CSV validation/combination failed"
        )

        print(
            f"Reason      : "
            f"{monthly_result['error']}"
        )


    else:

        print(
            "Monthly CSV : SKIPPED"
        )

        print(
            f"Reason      : "
            f"{monthly_result['error']}"
        )


    # -----------------------------------------------------
    # Write log
    # -----------------------------------------------------

    successful_days = [
        result
        for result in results
        if result["status"] == "SUCCESS"
    ]

    expected_month_rows = sum(
        result["records"]
        for result in successful_days
    )

    conversion_validated = (
        monthly_result["status"] == "SUCCESS"
        and bool(successful_days)
        and all(
            result["status"] in ("SUCCESS", "SKIPPED")
            for result in results
        )
        and monthly_result["daily_csv_count"] == len(successful_days)
        and monthly_result["rows"] == expected_month_rows
    )

    monthly_result["month_validation"] = (
        "SUCCESS"
        if conversion_validated
        else "FAILED"
    )

    monthly_result["expected_month_rows"] = expected_month_rows
    monthly_result["source_nc_files_found"] = 0
    monthly_result["source_nc_files_deleted"] = 0
    monthly_result["source_month_directory_removed"] = False
    monthly_result["cleanup_errors"] = []

    if conversion_validated:

        month_dir = (
            data_dir
            / str(year)
            / month_str
        )

        nc_files = list(
            month_dir.glob("*/*/*.nc")
        )
        monthly_result["source_nc_files_found"] = len(nc_files)

        for nc_file in nc_files:

            try:

                nc_file.unlink()
                monthly_result["source_nc_files_deleted"] += 1

            except OSError as e:

                monthly_result["cleanup_errors"].append(
                    f"{nc_file}: {type(e).__name__}: {e}"
                )

        if nc_files and not monthly_result["cleanup_errors"]:

            try:

                shutil.rmtree(month_dir)
                monthly_result["source_month_directory_removed"] = True

            except OSError as e:

                monthly_result["cleanup_errors"].append(
                    f"{month_dir}: {type(e).__name__}: {e}"
                )

        if monthly_result["cleanup_errors"]:

            monthly_result["cleanup_status"] = "PARTIAL"

        elif nc_files:

            monthly_result["cleanup_status"] = "DELETED"

        else:

            monthly_result["cleanup_status"] = "NO SOURCE FILES"

        print(
            f"Entire month validated ({expected_month_rows} rows); "
            f"deleted {monthly_result['source_nc_files_deleted']} "
            f"of {monthly_result['source_nc_files_found']} NC files; "
            f"removed month folder: "
            f"{monthly_result['source_month_directory_removed']}."
        )

    else:

        monthly_result["cleanup_status"] = "RETAINED"

        print(
            "CSV validation/conversion was incomplete; "
            "NC files were retained."
        )

    write_log(
        year,
        month,
        results,
        monthly_result,
        data_dir
    )


    # -----------------------------------------------------
    # Monthly summary
    # -----------------------------------------------------

    success_count = sum(
        1
        for r in results
        if r["status"] == "SUCCESS"
    )


    failed_count = sum(
        1
        for r in results
        if r["status"] == "FAILED"
    )


    skipped_count = sum(
        1
        for r in results
        if r["status"] == "SKIPPED"
    )


    print()

    print(
        f"Summary {year}-{month_str}: "
        f"SUCCESS={success_count}, "
        f"FAILED={failed_count}, "
        f"SKIPPED={skipped_count}"
    )


    print(
        f"Log: "
        f"{data_dir / 'conversion_logs' / str(year) / f'{month_str}.txt'}"
    )


# =========================================================
# 8. MAIN
# =========================================================

def main(product="CLP"):

    product = normalize_product(product)
    data_dir = DATA_ROOT / product

    print("=" * 70)

    print(
        "HIMAWARI NC TO DAILY CSV TO MONTHLY CSV"
    )

    print("=" * 70)


    print(
        f"Year   : {YEAR}"
    )


    print(
        f"Months : {MONTHS}"
    )


    print(
        f"Target : "
        f"{TARGET_LAT}, {TARGET_LON}"
    )


    print(
        "Original timestamp : UTC"
    )


    print(
        "Malaysia datetime  : UTC+8"
    )


    print(
        f"Monthly output : "
        f"{data_dir / 'monthly_csv' / str(YEAR)}"
    )


    print("=" * 70)


    # -----------------------------------------------------
    # Process selected months
    # -----------------------------------------------------

    for month in MONTHS:

        process_month(
            YEAR,
            month,
            data_dir,
            product
        )


    print()

    print("=" * 70)

    print(
        "ALL MONTHS COMPLETED"
    )

    print("=" * 70)


# =========================================================
# 9. RUN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert Himawari NC data to daily and monthly CSV files."
    )
    parser.add_argument(
        "--product",
        choices=tuple(PRODUCT_VARIABLES),
        default="CLP",
        help="Product to convert: CLP or ARP (default: CLP)."
    )

    main(parser.parse_args().product)
