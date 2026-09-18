"""
generate_recommendation_data.py
--------------------------------
Synthetic fallback generator matching the exact column schema of the real
Kaggle "Crop Recommendation Dataset" (N, P, K, temperature, humidity, ph,
rainfall, label). See data/DATASET_SETUP.md for how to obtain and swap in
the real file - this script exists only so the pipeline is runnable before
that swap happens.

Run:
    python data/generate_recommendation_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_PER_CROP = 100

# Approximate per-crop profiles (mean, std) loosely reflecting the public
# summary statistics of the real dataset, for demo purposes only.
CROP_PROFILE = {
    "rice":        dict(N=(80, 15), P=(48, 8),  K=(40, 8),  temp=(23.5, 3), hum=(82, 3),  ph=(6.4, 0.5), rain=(240, 25)),
    "maize":       dict(N=(78, 20), P=(48, 10), K=(20, 5),  temp=(22.5, 4), hum=(65, 8),  ph=(6.2, 0.4), rain=(85, 15)),
    "chickpea":    dict(N=(40, 8),  P=(67, 10), K=(80, 8),  temp=(19, 3),   hum=(16, 4),  ph=(7.3, 0.4), rain=(80, 20)),
    "kidneybeans": dict(N=(20, 6),  P=(67, 10), K=(20, 5),  temp=(20, 2),   hum=(21, 4),  ph=(5.7, 0.4), rain=(105, 20)),
    "pigeonpeas":  dict(N=(20, 6),  P=(67, 10), K=(20, 5),  temp=(27, 4),   hum=(48, 10), ph=(5.8, 0.5), rain=(150, 30)),
    "mothbeans":   dict(N=(21, 6),  P=(48, 8),  K=(20, 5),  temp=(28, 3),   hum=(53, 8),  ph=(6.8, 0.6), rain=(50, 15)),
    "mungbean":    dict(N=(20, 6),  P=(47, 8),  K=(20, 5),  temp=(28, 3),   hum=(85, 5),  ph=(6.7, 0.4), rain=(50, 12)),
    "blackgram":   dict(N=(40, 8),  P=(67, 8),  K=(19, 5),  temp=(29, 3),   hum=(65, 8),  ph=(7.1, 0.4), rain=(68, 15)),
    "lentil":      dict(N=(19, 6),  P=(68, 8),  K=(19, 5),  temp=(24, 3),   hum=(65, 8),  ph=(6.9, 0.4), rain=(46, 10)),
    "pomegranate": dict(N=(19, 6),  P=(18, 5),  K=(40, 6),  temp=(21, 3),   hum=(90, 3),  ph=(6.4, 0.4), rain=(107, 20)),
    "banana":      dict(N=(100, 15),P=(82, 8),  K=(50, 8),  temp=(27, 2),   hum=(80, 3),  ph=(5.9, 0.4), rain=(105, 20)),
    "mango":       dict(N=(20, 6),  P=(27, 6),  K=(30, 6),  temp=(31, 2),   hum=(50, 5),  ph=(5.7, 0.4), rain=(95, 20)),
    "grapes":      dict(N=(23, 6),  P=(132, 10),K=(200, 10),temp=(23.5, 2), hum=(81, 3),  ph=(6.0, 0.3), rain=(70, 10)),
    "watermelon":  dict(N=(99, 10), P=(17, 5),  K=(50, 6),  temp=(25, 2),   hum=(85, 3),  ph=(6.5, 0.4), rain=(50, 10)),
    "muskmelon":   dict(N=(100, 10),P=(17, 5),  K=(50, 6),  temp=(28, 2),   hum=(92, 2),  ph=(6.4, 0.3), rain=(24, 8)),
    "apple":       dict(N=(20, 6),  P=(134, 10),K=(200, 10),temp=(22.5, 2), hum=(92, 2),  ph=(5.9, 0.3), rain=(113, 15)),
    "orange":      dict(N=(19, 6),  P=(16, 5),  K=(10, 3),  temp=(22, 2),   hum=(92, 2),  ph=(7.0, 0.3), rain=(110, 15)),
    "papaya":      dict(N=(50, 8),  P=(59, 8),  K=(50, 6),  temp=(33, 2),   hum=(92, 2),  ph=(6.7, 0.4), rain=(142, 20)),
    "coconut":     dict(N=(22, 6),  P=(17, 5),  K=(31, 5),  temp=(27, 2),   hum=(95, 2),  ph=(5.9, 0.4), rain=(175, 20)),
    "cotton":      dict(N=(118, 10),P=(46, 8),  K=(20, 5),  temp=(24, 3),   hum=(80, 5),  ph=(6.9, 0.4), rain=(80, 15)),
    "jute":        dict(N=(78, 10), P=(47, 8),  K=(40, 6),  temp=(25, 2),   hum=(80, 3),  ph=(6.7, 0.3), rain=(175, 20)),
    "coffee":      dict(N=(101, 10),P=(28, 6),  K=(30, 6),  temp=(25, 2),   hum=(58, 8),  ph=(6.8, 0.4), rain=(160, 20)),
}


def generate(n_per_crop: int = N_PER_CROP, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for crop, p in CROP_PROFILE.items():
        for _ in range(n_per_crop):
            rows.append(dict(
                N=max(0, round(rng.normal(*p["N"]), 0)),
                P=max(0, round(rng.normal(*p["P"]), 0)),
                K=max(0, round(rng.normal(*p["K"]), 0)),
                temperature=round(rng.normal(*p["temp"]), 6),
                humidity=round(np.clip(rng.normal(*p["hum"]), 10, 100), 6),
                ph=round(np.clip(rng.normal(*p["ph"]), 3.5, 9.5), 6),
                rainfall=round(max(5, rng.normal(*p["rain"])), 6),
                label=crop,
            ))
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "raw", "crop_recommendation.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"[fallback/demo data] Generated {len(df)} rows -> {out_path}")
    print("Replace with the real Kaggle dataset before final submission - see data/DATASET_SETUP.md")
