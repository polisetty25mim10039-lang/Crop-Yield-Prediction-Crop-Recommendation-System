"""
data_loader.py
---------------
Module 1a — Data Ingestion.
Loads the two independent source datasets this project uses:

  1. Crop Recommendation dataset  -> N, P, K, temperature, humidity, ph,
     rainfall, label  (drives the `recommend-crop` task)
  2. Crop Production dataset      -> State_Name, Season, Crop, Crop_Year,
     Area, Production  (drives the `predict-yield` task)

Each loader does structural validation only (required columns present,
non-empty); cleaning/encoding logic lives in preprocess.py.
"""

import os
import pandas as pd

RECOMMEND_REQUIRED_COLUMNS = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label",
]

PRODUCTION_REQUIRED_COLUMNS = [
    "State_Name", "Season", "Crop", "Crop_Year", "Area", "Production",
]


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded or fails structural checks."""


def _load_csv(path: str, required_columns: list) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataLoadError(
            f"Dataset not found at '{path}'. See data/DATASET_SETUP.md for "
            f"where to download the real dataset, or run the matching "
            f"data/generate_*.py script for a schema-compatible fallback."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise DataLoadError(f"Dataset at '{path}' is empty.")

    # Real Kaggle exports sometimes carry a leading/trailing space in the
    # Season column (e.g. " Kharif     ") - normalize whitespace here so
    # downstream category handling isn't surprised by it.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise DataLoadError(
            f"Dataset at '{path}' is missing required columns: {sorted(missing_cols)}"
        )

    return df


def load_recommendation_data(path: str) -> pd.DataFrame:
    """Load the Crop Recommendation dataset (N-P-K + weather -> label)."""
    return _load_csv(path, RECOMMEND_REQUIRED_COLUMNS)


def load_production_data(path: str) -> pd.DataFrame:
    """Load the Crop Production dataset (state/season/crop/area -> production)."""
    return _load_csv(path, PRODUCTION_REQUIRED_COLUMNS)


if __name__ == "__main__":
    rec = load_recommendation_data("data/raw/crop_recommendation.csv")
    print(f"Recommendation dataset: {len(rec)} rows, {len(rec.columns)} columns.")

    prod = load_production_data("data/raw/crop_production.csv")
    print(f"Production dataset: {len(prod)} rows, {len(prod.columns)} columns.")
