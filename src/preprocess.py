"""
preprocess.py
-------------
Module 1b — Data Preprocessing & Feature Engineering.
Two independent pipelines, one per dataset/task:

  preprocess_recommendation() - cleans the N-P-K/weather dataset and adds
    an original 'npk_balance_score' feature.
  preprocess_production()     - cleans the state/season/area/production
    dataset, derives the yield target, and adds an original
    'area_percentile' feature.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Recommendation pipeline
# ---------------------------------------------------------------------------

RECOMMEND_NUMERIC_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def npk_balance_score(df: pd.DataFrame) -> pd.Series:
    """
    Original engineered feature: how *balanced* a sample's N, P, K levels
    are relative to each other, on a 0-1 scale (1 = perfectly balanced,
    i.e. N == P == K). Some crops (e.g. grapes/apple) want a very
    imbalanced, K-heavy profile, while others want near-even N-P-K, so this
    gives the model a compact summary of nutrient "shape" independent of
    absolute nutrient level.
    """
    npk = df[["N", "P", "K"]].astype(float)
    row_mean = npk.mean(axis=1)
    row_std = npk.std(axis=1)
    # Coefficient of variation, inverted and squashed into [0, 1].
    cv = row_std / row_mean.replace(0, np.nan)
    cv = cv.fillna(0)
    return (1 / (1 + cv)).round(4)


def preprocess_recommendation(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """Clean, engineer features for, and encode the recommendation dataset."""
    df = df.copy()

    for col in RECOMMEND_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        mean, std = df[col].mean(), df[col].std()
        if std and not np.isnan(std):
            df[col] = df[col].clip(mean - 4 * std, mean + 4 * std)

    df["npk_balance_score"] = npk_balance_score(df)

    encoders = encoders or {}
    if fit:
        le = LabelEncoder()
        df["label_enc"] = le.fit_transform(df["label"].astype(str))
        encoders["label"] = le
    else:
        le = encoders["label"]
        df["label_enc"] = df["label"].astype(str).map(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )

    return df, encoders


def get_recommend_features():
    return RECOMMEND_NUMERIC_COLUMNS + ["npk_balance_score"]


# ---------------------------------------------------------------------------
# Production / yield pipeline
# ---------------------------------------------------------------------------

def area_percentile(df: pd.DataFrame) -> pd.Series:
    """
    Original engineered feature: each record's Area expressed as a
    percentile (0-1) within its own crop's area distribution. This lets
    the model distinguish "a large plot for this particular crop" from
    "a large plot in absolute terms", since crops differ hugely in typical
    plot size (sugarcane vs. pulses, for example).
    """
    return df.groupby("Crop")["Area"].rank(pct=True).round(4)


def preprocess_production(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """Clean, derive targets for, engineer features for, and encode the production dataset."""
    df = df.copy()

    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

    # Drop rows where Area is missing/zero or Production is missing - these
    # can't yield a meaningful target and are a small fraction of real
    # agri datasets.
    df = df[(df["Area"] > 0) & (df["Production"].notna())].copy()

    # Derive the actual modeling target: yield per hectare.
    df["yield_tons_per_ha"] = (df["Production"] / df["Area"]).round(4)

    # Clip extreme outliers (data entry errors are common in real agri data).
    mean, std = df["yield_tons_per_ha"].mean(), df["yield_tons_per_ha"].std()
    df["yield_tons_per_ha"] = df["yield_tons_per_ha"].clip(0, mean + 6 * std)

    df["area_percentile"] = area_percentile(df)

    encoders = encoders or {}
    cat_cols = ["State_Name", "Season", "Crop"]
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col + "_enc"] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )

    return df, encoders


def get_production_features():
    return ["Area", "Crop_Year", "area_percentile", "State_Name_enc", "Season_enc", "Crop_enc"]
