from ftplib import FTP, error_perm
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================================================
# 1. CONFIGURATION
# =========================================================

HOST = "ftp.ptree.jaxa.jp"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

CONFIG_FILE = SCRIPT_DIR / "config.txt"

config = {}

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split(
            "=",
            1
        )

        config[key.strip()] = value.strip()


USERNAME = config["JAXA_FTP_USERNAME"]
PASSWORD = config["JAXA_FTP_PASSWORD"]


# =========================================================
# 2. DOWNLOAD SETTINGS
# =========================================================

YEARS = [
    2026
]

MONTHS = [
    6,
    7
]


PRODUCTS = {
    "CLP": "010",
    # "ARP": "031"
}


# ---------------------------------------------------------
# IMPORTANT:
# These hours are MALAYSIA TIME (MYT / UTC+8)
# --------------------------------s-------------------------

START_HOUR = 9
END_HOUR = 18


# Number of day-level workers
MAX_WORKERS = 8

# Number of retries for an individual file
MAX_RETRIES = 3


# =========================================================
# 3. OUTPUT DIRECTORIES
# =========================================================

OUTPUT_DIR = Path(
    os.environ.get(
        "HIMAWARI_DATA_ROOT",
        str(PROJECT_ROOT / "dataset" / "row" / "himawari8")
    )
).expanduser()

if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR

OUTPUT_DIR = OUTPUT_DIR.resolve()

LOG_DIR = OUTPUT_DIR / "logs"


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 4. VALIDATE TIME CONFIGURATION
# =========================================================

if not 0 <= START_HOUR <= 23:
    raise ValueError(
        "START_HOUR must be between 0 and 23."
    )

if not 0 <= END_HOUR <= 23:
    raise ValueError(
        "END_HOUR must be between 0 and 23."
    )

if START_HOUR > END_HOUR:
    raise ValueError(
        "START_HOUR must be less than or equal to END_HOUR."
    )


# =========================================================
# 5. CONNECT TO FTP
# =========================================================

def connect_ftp():

    ftp = FTP(
        HOST,
        timeout=60
    )

    ftp.login(
        USERNAME,
        PASSWORD
    )

    ftp.set_pasv(True)

    return ftp


# =========================================================
# 6. DOWNLOAD ONE FILE
# =========================================================

def download_file(
    ftp,
    remote_file,
    local_file
):

    # -----------------------------------------------------
    # Skip if file already exists
    # -----------------------------------------------------

    if local_file.exists():

        return "skipped", None


    local_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    last_error = None


    # -----------------------------------------------------
    # Retry individual file
    # -----------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            with open(
                local_file,
                "wb"
            ) as f:

                ftp.retrbinary(
                    f"RETR {remote_file}",
                    f.write
                )


            return "downloaded", None


        except Exception as e:

            last_error = (
                f"{type(e).__name__}: {e}"
            )


            # Remove incomplete file

            if local_file.exists():

                try:
                    local_file.unlink()

                except Exception:
                    pass


            if attempt < MAX_RETRIES:

                time.sleep(2)


    return "failed", last_error


# =========================================================
# 7. DOWNLOAD ONE DAY
# =========================================================

def download_day(
    year,
    month,
    day,
    product_key
):

    ftp = None

    errors = []
    warnings = []

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0


    # -----------------------------------------------------
    # The requested day/time is Malaysia Time.
    #
    # Convert the MYT start and end datetime to UTC.
    # -----------------------------------------------------

    myt_start = datetime(
        year,
        month,
        day,
        START_HOUR,
        0
    )

    myt_end = datetime(
        year,
        month,
        day,
        END_HOUR,
        0
    )


    # Malaysia = UTC + 8
    # Therefore UTC = MYT - 8 hours

    utc_start = (
        myt_start - timedelta(hours=8)
    )

    utc_end = (
        myt_end - timedelta(hours=8)
    )


    try:

        # -------------------------------------------------
        # One FTP connection per day
        # -------------------------------------------------

        ftp = connect_ftp()


        # -------------------------------------------------
        # Loop through every UTC hour required by the
        # Malaysia-time range.
        #
        # Example:
        #
        # 06:00 MYT → 22:00 UTC previous day
        # 18:00 MYT → 10:00 UTC same day
        # -------------------------------------------------

        current_utc = utc_start


        while current_utc <= utc_end:

            utc_year = current_utc.year
            utc_month = current_utc.month
            utc_day = current_utc.day
            utc_hour = current_utc.hour


            # -------------------------------------------------
            # JAXA P-Tree uses UTC date/hour directories
            # -------------------------------------------------

            yyyymm = (
                f"{utc_year:04d}"
                f"{utc_month:02d}"
            )

            dd = (
                f"{utc_day:02d}"
            )

            hh = (
                f"{utc_hour:02d}"
            )


            # -------------------------------------------------
            # Product directories
            # -------------------------------------------------

            for product, version in (
                (product_key, PRODUCTS[product_key]),
            ):

                day_remote = (
                    f"/pub/himawari/L2/"
                    f"{product}/"
                    f"{version}/"
                    f"{yyyymm}/"
                    f"{dd}"
                )


                remote_dir = (
                    f"{day_remote}/"
                    f"{hh}"
                )


                try:

                    ftp.cwd(
                        remote_dir
                    )

                    files = ftp.nlst()


                except Exception as e:

                    error_message = (
                        f"{type(e).__name__}: {e}"
                    )

                    if (
                        isinstance(e, error_perm)
                        and str(e).startswith("550 ")
                        and "no such file or directory"
                        in str(e).lower()
                    ):

                        warnings.append(
                            f"Directory {remote_dir}: "
                            f"{error_message}"
                        )

                        continue

                    errors.append(
                        f"Directory {remote_dir}: "
                        f"{error_message}"
                    )

                    failed_count += 1

                    continue


                # -------------------------------------------------
                # Only download NetCDF files
                # -------------------------------------------------

                nc_files = [
                    filename
                    for filename in files
                    if filename.lower().endswith(".nc")
                ]


                for filename in nc_files:

                    remote_file = (
                        f"{remote_dir}/"
                        f"{filename}"
                    )


                    # -------------------------------------------------
                    # Keep local directory structure based on UTC.
                    #
                    # This matches the JAXA FTP structure:
                    #
                    # product/year/month/day/hour
                    # -------------------------------------------------

                    local_file = (
                        OUTPUT_DIR
                        / product
                        / f"{utc_year:04d}"
                        / f"{utc_month:02d}"
                        / f"{utc_day:02d}"
                        / hh
                        / filename
                    )


                    status, error = download_file(
                        ftp,
                        remote_file,
                        local_file
                    )


                    if status == "downloaded":

                        downloaded_count += 1


                    elif status == "skipped":

                        skipped_count += 1


                    elif status == "failed":

                        failed_count += 1

                        errors.append(
                            f"Download {remote_file}: "
                            f"{error}"
                        )


            # -------------------------------------------------
            # Move to next UTC hour
            # -------------------------------------------------

            current_utc += timedelta(
                hours=1
            )


        # -----------------------------------------------------
        # Determine day status
        # -----------------------------------------------------

        if failed_count > 0:

            status = "FAILED"

        elif warnings:

            status = "WARNING"

        elif downloaded_count > 0:

            status = "DOWNLOADED"


        else:

            status = "SKIPPED"


        return {
            "year": year,
            "month": month,
            "day": day,
            "product": product_key,
            "status": status,
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "errors": errors,
            "warnings": warnings,
        }


    except Exception as e:

        error_message = (
            f"{type(e).__name__}: {e}"
        )


        return {
            "year": year,
            "month": month,
            "day": day,
            "product": product_key,
            "status": "FAILED",
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "failed": failed_count + 1,
            "errors": [
                f"Day connection/process error: "
                f"{error_message}"
            ],
            "warnings": warnings,
        }


    finally:

        # -----------------------------------------------------
        # Close FTP connection
        # -----------------------------------------------------

        if ftp is not None:

            try:
                ftp.quit()

            except Exception:

                try:
                    ftp.close()

                except Exception:
                    pass


# =========================================================
# 8. WRITE MONTHLY LOG
# =========================================================

def write_month_log(
    year,
    month,
    product,
    results
):

    year_log_dir = (
        LOG_DIR
        / f"{year:04d}_{product}"
    )

    year_log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = (
        year_log_dir
        / f"{month:02d}.txt"
    )


    total_days = len(results)

    downloaded_days = sum(
        1
        for result in results
        if result["downloaded"] > 0
    )

    failed_days = sum(
        1
        for result in results
        if result["status"] == "FAILED"
    )

    skipped_days = sum(
        1
        for result in results
        if result["status"] == "SKIPPED"
    )

    warning_days = sum(
        1
        for result in results
        if result.get("warnings")
    )


    total_files_downloaded = sum(
        result["downloaded"]
        for result in results
    )

    total_files_skipped = sum(
        result["skipped"]
        for result in results
    )

    total_files_failed = sum(
        result["failed"]
        for result in results
    )


    # -----------------------------------------------------
    # Write log
    # -----------------------------------------------------

    with open(
        log_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Himawari FTP Download Log\n"
        )

        f.write(
            f"Year: {year}\n"
        )

        f.write(
            f"Product: {product}\n"
        )

        f.write(
            f"Month: {month:02d}\n"
        )

        f.write(
            f"Time Zone: Malaysia Time (MYT, UTC+8)\n"
        )

        f.write(
            f"Requested Time: "
            f"{START_HOUR:02d}:00 - "
            f"{END_HOUR:02d}:00 MYT\n"
        )

        f.write(
            f"UTC Time Range: "
            f"{START_HOUR:02d}:00 - "
            f"{END_HOUR:02d}:00 MYT converted automatically\n"
        )

        f.write(
            f"\n"
        )


        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        f.write(
            "SUMMARY\n"
        )

        f.write(
            "-------\n"
        )

        f.write(
            f"Total days: {total_days}\n"
        )

        f.write(
            f"Downloaded days: {downloaded_days}\n"
        )

        f.write(
            f"Failed days: {failed_days}\n"
        )

        f.write(
            f"Skipped days: {skipped_days}\n"
        )

        f.write(
            f"Warning days: {warning_days}\n"
        )

        f.write(
            f"Files downloaded: "
            f"{total_files_downloaded}\n"
        )

        f.write(
            f"Files skipped: "
            f"{total_files_skipped}\n"
        )

        f.write(
            f"Files failed: "
            f"{total_files_failed}\n"
        )

        f.write(
            "\n"
        )


        # -------------------------------------------------
        # Daily status
        # -------------------------------------------------

        f.write(
            "DAILY STATUS\n"
        )

        f.write(
            "------------\n"
        )


        for result in sorted(
            results,
            key=lambda x: x["day"]
        ):

            date_str = (
                f"{year:04d}-"
                f"{month:02d}-"
                f"{result['day']:02d}"
            )


            f.write(
                f"{date_str} : "
                f"{result['status']}\n"
            )


            # -------------------------------------------------
            # Errors
            # -------------------------------------------------

            for error in result["errors"]:

                f.write(
                    f"    {error}\n"
                )

            for warning in result.get("warnings", []):

                f.write(
                    f"    WARNING: {warning}\n"
                )


# =========================================================
# 9. PROCESS ONE MONTH
# =========================================================

def process_month(
    year,
    month
):

    days_in_month = calendar.monthrange(
        year,
        month
    )[1]


    tasks = [
        (
            year,
            month,
            day,
            product
        )
        for product in PRODUCTS
        for day in range(1, days_in_month + 1)
    ]


    results = []


    # -----------------------------------------------------
    # 8 workers.
    #
    # Each task = one complete day.
    # -----------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_task = {
            executor.submit(
                download_day,
                year,
                month,
                day,
                product
            ): (
                year,
                month,
                day,
                product
            )

            for year, month, day, product in tasks
        }


        for future in as_completed(
            future_to_task
        ):

            result = future.result()

            results.append(
                result
            )


            # -------------------------------------------------
            # Minimal console output
            # -------------------------------------------------

            date_str = (
                f"{result['year']:04d}-"
                f"{result['month']:02d}-"
                f"{result['day']:02d}"
            )


            print(
                f"{date_str} : "
                f"{result['product']} : "
                f"{result['status']}"
            )


    # -----------------------------------------------------
    # Write monthly log
    # -----------------------------------------------------

    for product in PRODUCTS:
        product_results = [
            result
            for result in results
            if result["product"] == product
        ]
        write_month_log(
            year,
            month,
            product,
            product_results
        )

    return all(
        result["failed"] == 0
        for result in results
    )


# =========================================================
# 10. MAIN
# =========================================================

def main():

    print(
        "Himawari FTP Downloader"
    )

    print(
        f"Time range: "
        f"{START_HOUR:02d}:00 - "
        f"{END_HOUR:02d}:00 MYT"
    )

    print(
        "UTC conversion: MYT - 8 hours"
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    print()


    for year in YEARS:

        for month in MONTHS:

            print(
                f"Processing "
                f"{year}-{month:02d}"
            )


            month_downloaded_without_failures = process_month(
                year,
                month
            )

            if month_downloaded_without_failures:

                print(
                    f"Converting downloaded data for "
                    f"{year}-{month:02d}..."
                )

                from nc_to_csv import process_month as convert_month

                for product in PRODUCTS:

                    convert_month(
                        year,
                        month,
                        OUTPUT_DIR / product,
                        product
                    )

            else:

                print(
                    f"Skipping conversion for "
                    f"{year}-{month:02d}: download failures occurred."
                )


            print()


    print(
        "Download completed."
    )


# =========================================================
# 11. RUN
# =========================================================

if __name__ == "__main__":

    main()