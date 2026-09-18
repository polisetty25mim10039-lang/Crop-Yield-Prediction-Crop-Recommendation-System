"""
recommend.py
------------
Module 3b — Crop Recommendation Inference.
Loads the saved classifier and recommends the top-N most suitable crops
given N-P-K and weather conditions (the recommendation dataset's schema).
"""

import numpy as np
import pandas as pd

from src.model import load_model
from src.preprocess import get_recommend_features, npk_balance_score
from src.utils import validate_range


def recommend_crops(inputs: dict, config: dict, encoders: dict, top_n: int = 3) -> dict:
    """
    Validate inputs, run inference, and return the top-N recommended crops
    with their predicted suitability probabilities.
    """
    n = validate_range("N", inputs["N"], 0, 500)
    p = validate_range("P", inputs["P"], 0, 500)
    k = validate_range("K", inputs["K"], 0, 500)
    temperature = validate_range("temperature", inputs["temperature"], -10, 55)
    humidity = validate_range("humidity", inputs["humidity"], 0, 100)
    ph = validate_range("ph", inputs["ph"], 0, 14)
    rainfall = validate_range("rainfall", inputs["rainfall"], 0, 5000)

    row = pd.DataFrame([{
        "N": n, "P": p, "K": k, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall,
    }])
    row["npk_balance_score"] = npk_balance_score(row)

    model = load_model(config["paths"]["recommend_model"])
    features = get_recommend_features()
    proba = model.predict_proba(row[features])[0]

    label_encoder = encoders["label"]
    top_idx = np.argsort(proba)[::-1][:top_n]
    top_crops = [
        {"crop": label_encoder.inverse_transform([model.classes_[i]])[0],
         "suitability": round(float(proba[i]), 4)}
        for i in top_idx
    ]

    return {"recommendations": top_crops}
