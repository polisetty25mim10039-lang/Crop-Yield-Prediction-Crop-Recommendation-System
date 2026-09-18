"""
predict.py
----------
Module 3a — Yield Prediction Inference.
Loads the saved yield model + production encoders and produces a yield/
production estimate for a single state/season/crop/area combination.
"""

import pandas as pd

from src.model import load_model
from src.preprocess import get_production_features
from src.utils import validate_range, validate_choice


def predict_yield(inputs: dict, config: dict, encoders: dict,
                   valid_states, valid_seasons, valid_crops) -> dict:
    """
    Validate inputs, run the production-pipeline feature engineering, and
    return predicted yield (tons/ha) and estimated production (tons).
    """
    state = validate_choice("state", inputs["state"], valid_states)
    season = validate_choice("season", inputs["season"], valid_seasons)
    crop = validate_choice("crop", inputs["crop"], valid_crops)
    area_ha = validate_range("area_ha", inputs["area_ha"], 0.01, 1_000_000)
    crop_year = validate_range("crop_year", inputs.get("crop_year", 2020), 1990, 2035)

    row = pd.DataFrame([{
        "State_Name": state, "Season": season, "Crop": crop,
        "Area": area_ha, "Crop_Year": crop_year,
    }])

    # area_percentile can't be computed relative to a historical distribution
    # for a single ad-hoc query, so we use the neutral midpoint (0.5).
    row["area_percentile"] = 0.5

    row["State_Name_enc"] = encoders["State_Name"].transform([state])[0] \
        if state in encoders["State_Name"].classes_ else -1
    row["Season_enc"] = encoders["Season"].transform([season])[0] \
        if season in encoders["Season"].classes_ else -1
    row["Crop_enc"] = encoders["Crop"].transform([crop])[0] \
        if crop in encoders["Crop"].classes_ else -1

    model = load_model(config["paths"]["yield_model"])
    features = get_production_features()
    predicted_yield = float(model.predict(row[features])[0])
    predicted_production = round(predicted_yield * area_ha, 3)

    return {
        "crop": crop,
        "state": state,
        "season": season,
        "predicted_yield_tons_per_ha": round(predicted_yield, 3),
        "predicted_production_tons": predicted_production,
        "area_ha": area_ha,
    }
