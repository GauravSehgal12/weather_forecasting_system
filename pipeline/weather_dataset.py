



import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr



# Years included in the dataset
YEARS = ["2021", "2022", "2023", "2024", "2025"]

# Monthly folders
MONTHS = [
    "JAN", "FEB", "MAR", "APR",
    "MAY", "JUN", "JUL", "AUG",
    "SEP", "OCT", "NOV", "DEC"
]


TARGET_LATITUDE = 26.00
TARGET_LONGITUDE = 83.00


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data"


for year in YEARS:
    for month in MONTHS:

        zip_path = (
            DATASET_DIR
            / "archives"
            / year
            / f"azamgarh_weather_{month}_{year}.zip"
        )

        extract_path = (
            DATASET_DIR
            / "raw"
            / year
            / month
        )

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"{month}_{year} extracted successfully.")

# =============================================================================
# Load Instantaneous Weather Variables
# =============================================================================
# This dataset contains variables such as:
#
# • Temperature
# • Dew Point Temperature
# • Surface Pressure
# • Wind Components
# • Cloud Cover
#
# Monthly datasets are combined into a single dataframe.

instant_dfs = []

for year in YEARS:
    for month in MONTHS:

        file_path = (
            DATASET_DIR
            / "raw"
            / year
            / month
            / "data_stream-oper_stepType-instant.nc"
        )

        with xr.open_dataset(
            file_path,
            engine="netcdf4"
        ) as ds:

            df = ds.to_dataframe().reset_index()

        instant_dfs.append(df)

# Combine all instantaneous datasets
instant_df = pd.concat(
    instant_dfs,
    ignore_index=True
)



precipitation_dfs = []

for year in YEARS:
    for month in MONTHS:

        file_path = (
            DATASET_DIR
            / "raw"
            / year
            / month
            / "data_stream-oper_stepType-accum.nc"
        )

        with xr.open_dataset(
            file_path,
            engine="netcdf4"
        ) as ds:

            df = ds.to_dataframe().reset_index()

        precipitation_dfs.append(df)

# Combine all precipitation datasets
precipitation_df = pd.concat(
    precipitation_dfs,
    ignore_index=True
)



single_temp_df = instant_df[
    (instant_df["latitude"] == TARGET_LATITUDE) &
    (instant_df["longitude"] == TARGET_LONGITUDE)
].reset_index(drop=True)

single_precipitation_df = precipitation_df[
    (precipitation_df["latitude"] == TARGET_LATITUDE) &
    (precipitation_df["longitude"] == TARGET_LONGITUDE)
].reset_index(drop=True)


single_temp_df["tp"] = single_precipitation_df["tp"].values

# Create a consolidated raw weather dataframe
weather_df = single_temp_df.copy()



OUTPUT_DIR = DATASET_DIR / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "azamgarh_weather_raw.csv"
)



weather_df.to_csv(
    OUTPUT_FILE,
    index=False
)



print("\nDataset saved successfully.")

print(f"\nLocation      : {OUTPUT_FILE}")
print(f"Total Rows    : {weather_df.shape[0]}")
print(f"Total Columns : {weather_df.shape[1]}")