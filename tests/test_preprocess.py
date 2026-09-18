import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    preprocess_recommendation, get_recommend_features, npk_balance_score,
    preprocess_production, get_production_features, area_percentile,
)


@pytest.fixture
def recommendation_df():
    return pd.DataFrame({
        "N": [90, 20, 60, np.nan],
        "P": [42, 67, 55, 50],
        "K": [43, 20, 44, 45],
        "temperature": [20.9, 19.0, 23.0, 22.0],
        "humidity": [82.0, 16.0, 82.3, 60.0],
        "ph": [6.5, 7.3, 7.8, 6.5],
        "rainfall": [202.9, 80.0, 264.0, 150.0],
        "label": ["rice", "chickpea", "rice", "maize"],
    })


@pytest.fixture
def production_df():
    return pd.DataFrame({
        "State_Name": ["Punjab", "Haryana", "Punjab", "Karnataka"],
        "Season": ["Kharif", "Rabi", "Kharif", "Whole Year"],
        "Crop": ["Rice", "Wheat", "Rice", "Maize"],
        "Crop_Year": [2010, 2011, 2012, 2013],
        "Area": [100.0, 50.0, 0.0, np.nan],  # includes a zero and a NaN
        "Production": [270.0, 155.0, 90.0, 130.0],
    })


def test_npk_balance_score_is_between_0_and_1(recommendation_df):
    scores = npk_balance_score(recommendation_df.fillna(0))
    assert (scores >= 0).all() and (scores <= 1).all()


def test_preprocess_recommendation_adds_features_and_encodes(recommendation_df):
    processed, encoders = preprocess_recommendation(recommendation_df, fit=True)
    assert "npk_balance_score" in processed.columns
    assert "label_enc" in processed.columns
    assert processed["N"].isna().sum() == 0  # median-filled
    assert "label" in encoders


def test_preprocess_recommendation_transform_handles_unseen_label(recommendation_df):
    _, encoders = preprocess_recommendation(recommendation_df, fit=True)
    new_row = recommendation_df.iloc[[0]].copy()
    new_row["label"] = "totally_unseen_crop"
    transformed, _ = preprocess_recommendation(new_row, encoders=encoders, fit=False)
    assert transformed["label_enc"].iloc[0] == -1


def test_get_recommend_features_matches_columns(recommendation_df):
    processed, _ = preprocess_recommendation(recommendation_df, fit=True)
    features = get_recommend_features()
    assert set(features).issubset(set(processed.columns))


def test_preprocess_production_drops_invalid_area_rows(production_df):
    processed, encoders = preprocess_production(production_df, fit=True)
    # The zero-area and NaN-area rows should be dropped.
    assert len(processed) == 2
    assert (processed["Area"] > 0).all()


def test_preprocess_production_derives_yield_correctly(production_df):
    processed, encoders = preprocess_production(production_df, fit=True)
    row = processed[processed["State_Name"] == "Punjab"].iloc[0]
    assert row["yield_tons_per_ha"] == pytest.approx(270.0 / 100.0, rel=1e-3)


def test_area_percentile_within_bounds(production_df):
    valid = production_df[(production_df["Area"] > 0)]
    pct = area_percentile(valid)
    assert (pct > 0).all() and (pct <= 1).all()


def test_get_production_features_matches_columns(production_df):
    processed, _ = preprocess_production(production_df, fit=True)
    features = get_production_features()
    assert set(features).issubset(set(processed.columns))
