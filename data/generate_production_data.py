"""
generate_production_data.py
-----------------------------
Synthetic fallback generator matching the column schema of the real Kaggle
"Crop Production in India" dataset (State_Name, Season, Crop, Crop_Year,
Area, Production). See data/DATASET_SETUP.md for how to obtain and swap in
the real file.

Run:
    python data/generate_production_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 8000

STATES = [
    "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh",
    "Maharashtra", "Karnataka", "Tamil Nadu", "West Bengal",
]
SEASONS = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"]
CROPS = [
    "Rice", "Wheat", "Maize", "Sugarcane", "Cotton(lint)",
    "Groundnut", "Soyabean", "Bajra", "Barley", "Arhar/Tur",
]

# Approximate yield-per-hectare (tons/ha) by crop, used only to synthesize
# a plausible Production figure from a sampled Area.
CROP_YIELD_PER_HA = {
    "Rice": 2.7, "Wheat": 3.1, "Maize": 2.6, "Sugarcane": 68.0,
    "Cotton(lint)": 0.45, "Groundnut": 1.4, "Soyabean": 1.1,
    "Bajra": 1.2, "Barley": 2.4, "Arhar/Tur": 0.9,
}


def generate(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        crop = rng.choice(CROPS)
        state = rng.choice(STATES)
        season = rng.choice(SEASONS)
        year = int(rng.integers(2005, 2016))
        area = round(max(0.1, rng.gamma(shape=2.0, scale=800)), 1)  # hectares

        base_yield = CROP_YIELD_PER_HA[crop]
        noise = rng.normal(1.0, 0.15)
        production = round(max(0.1, area * base_yield * noise), 1)

        rows.append(dict(
            State_Name=state,
            Season=season,
            Crop=crop,
            Crop_Year=year,
            Area=area,
            Production=production,
        ))

    df = pd.DataFrame(rows)

    # Real agricultural datasets commonly have some missing Production values.
    mask = rng.random(len(df)) < 0.015
    df.loc[mask, "Production"] = np.nan

    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "raw", "crop_production.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"[fallback/demo data] Generated {len(df)} rows -> {out_path}")
    print("Replace with the real Kaggle dataset before final submission - see data/DATASET_SETUP.md")
