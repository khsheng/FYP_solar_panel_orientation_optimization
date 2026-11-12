"""Add PVlib solar-position and clear-sky irradiance features to joined data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "combined"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "pvlib"
TIMEZONE = "Asia/Kuala_Lumpur"
SITE_LATITUDE = 2.450000762939453
SITE_LONGITUDE = 102.69999694824219
SITE_ALTITUDE_M = 0.0
DATASETS = (
    "experiment_CLP_2026_combined.csv",
)


def prepare_dataset(filename: str) -> Path:
    """Compute PVlib solar and Ineichen clear-sky data for one joined CSV."""
    input_path = INPUT_DIR / filename
    if not input_path.is_file():
        raise FileNotFoundError(f"Joined dataset not found: {input_path}")

    data = pd.read_csv(input_path)
    if "Datetime" not in data.columns:
        raise ValueError(
            f"Expected 'Datetime' column in {input_path.name}; "
            f"found {data.columns.tolist()}"
        )

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(data["Datetime"], errors="raise", utc=True)
    )
    if timestamps.hasnans:
        raise ValueError(f"{input_path.name} contains missing Datetime values.")
    if timestamps.has_duplicates:
        raise ValueError(f"{input_path.name} contains duplicate Datetime values.")

    location = pvlib.location.Location(
        latitude=SITE_LATITUDE,
        longitude=SITE_LONGITUDE,
        tz=TIMEZONE,
        altitude=SITE_ALTITUDE_M,
    )
    solar_position = location.get_solarposition(timestamps)
    clear_sky = location.get_clearsky(timestamps, model="ineichen")

    data["Az sun"] = solar_position["azimuth"].to_numpy()
    data["Alt sun"] = solar_position["elevation"].to_numpy()
    data["DNI clear"] = clear_sky["dni"].to_numpy()
    data["DHI clear"] = clear_sky["dhi"].to_numpy()

    computed = data[["Az sun", "Alt sun", "DNI clear", "DHI clear"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(computed).all():
        raise ValueError(
            f"PVlib produced missing or infinite values for {input_path.name}."
        )
    if (data[["DNI clear", "DHI clear"]] < 0).any().any():
        raise ValueError(
            f"PVlib produced negative clear-sky irradiance for {input_path.name}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_path.stem}_pvlib.csv"
    data.to_csv(output_path, index=False)
    print(
        f"{input_path.name}: {len(data):,} rows; "
        f"saved PVlib features to {output_path}"
    )
    return output_path


def main() -> None:
    for filename in DATASETS:
        prepare_dataset(filename)


if __name__ == "__main__":
    main()