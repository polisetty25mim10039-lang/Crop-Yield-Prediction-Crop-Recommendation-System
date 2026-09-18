# Crop Yield Prediction & Recommendation System

A command-line AI/ML system that combines two capabilities, each trained
from its own real-world public dataset:

1. **Crop Recommendation** — given soil (N, P, K, pH) and weather
   (temperature, humidity, rainfall) conditions, recommends the top-N most
   suitable crops with suitability scores.
2. **Yield Prediction** — given a state, season, crop, and cultivated area,
   predicts expected yield (tons/hectare) and total production.

Built as a course project for *Fundamentals of AI and ML*, demonstrating a
full ML pipeline: data ingestion, preprocessing/feature engineering, model
training & evaluation, and CLI-based inference.

---

## Features

- Two independently trained ML models, each on its own real dataset:
  - `RandomForestClassifier` (or `GradientBoostingClassifier`) for crop
    recommendation
  - `RandomForestRegressor` (or `GradientBoostingRegressor`) for yield
    prediction — algorithm is switchable via `config.yaml` for easy
    comparison
- Two original engineered features:
  - **`npk_balance_score`** — how balanced a sample's N/P/K levels are
    relative to each other (0-1), independent of absolute nutrient level
  - **`area_percentile`** — a record's cultivated area expressed as a
    percentile within its own crop's area distribution, so the model can
    tell "large for this crop" apart from "large in absolute terms"
- Input validation with clear error messages (range checks + allowed
  categorical values)
- Centralized logging to both console and `logs/app.log`
- Config-driven paths and hyperparameters (`config.yaml`) — no hardcoded
  values in source code
- Fully CLI-executable — no GUI/browser required
- Unit tests covering preprocessing, model training, and input validation

---

## Technologies / Tools Used

- Python 3.9+
- pandas, numpy — data handling
- scikit-learn — model training & evaluation
- joblib — model persistence
- PyYAML — configuration
- pytest — unit testing
- matplotlib — diagram generation (`docs/generate_diagrams.py`)

---

## Project Structure

```
crop-yield-predictor/
├── data/
│   ├── DATASET_SETUP.md              # where to get the real datasets
│   ├── generate_recommendation_data.py  # fallback generator (real schema)
│   ├── generate_production_data.py      # fallback generator (real schema)
│   └── raw/                          # place real/fallback CSVs here
├── src/
│   ├── data_loader.py       # Module 1a - loads both datasets, validates schema
│   ├── preprocess.py        # Module 1b - cleaning, feature engineering, encoding
│   ├── model.py              # Module 2  - model definitions & training (both algos)
│   ├── evaluate.py           # Module 2  - evaluation metrics
│   ├── predict.py             # Module 3a - yield prediction inference
│   ├── recommend.py           # Module 3b - crop recommendation inference
│   ├── cli.py                 # Module 3c - command-line interface
│   ├── logger.py              # centralized logging
│   └── utils.py                # config loading & input validation
├── tests/                    # unit tests
├── docs/                      # architecture / workflow / UML diagrams
├── models/                   # trained model artifacts (generated)
├── config.yaml               # paths & hyperparameters
├── train.py                  # end-to-end training entry point
├── main.py                   # CLI entry point
├── requirements.txt
└── statement.md               # problem statement, scope, target users
```

---

## Setup & Installation

### 1. Clone the repository

```bash
[git clone https://github.com/polisetty25mim10039-lang/Crop-Yield-Prediction-Crop-Recommendation-System)
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the datasets

See **`data/DATASET_SETUP.md`** for exact download links to the two real
Kaggle datasets this project uses. Quick start with fallback data (same
schema, synthetic values) if you just want to try the pipeline first:

```bash
python data/generate_recommendation_data.py   # -> data/raw/crop_recommendation.csv
python data/generate_production_data.py       # -> data/raw/crop_production.csv
```

### 5. Train the models

```bash
python train.py
```

This runs the full pipeline for both tasks: preprocessing → training →
evaluation → saving model artifacts to `models/`. You should see a summary
like:

```
=== Training Summary ===
Recommend model (random_forest_classifier) -> Accuracy: 0.95, F1: 0.95
Yield model (random_forest_regressor)      -> RMSE: 3.1, MAE: 1.8, R2: 0.97
```

---

## Usage

### Recommend the best crops for given soil/weather conditions

```bash
python main.py recommend-crop \
  --n 90 --p 42 --k 43 \
  --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 202.9 \
  --top-n 3
```

Output:
```json
{
  "recommendations": [
    {"crop": "rice", "suitability": 0.62},
    {"crop": "jute", "suitability": 0.31},
    {"crop": "coconut", "suitability": 0.04}
  ]
}
```

### Predict yield/production for a chosen crop

```bash
python main.py predict-yield \
  --state Punjab --season Kharif --crop Rice --area 2.5
```

Output:
```json
{
  "crop": "Rice",
  "state": "Punjab",
  "season": "Kharif",
  "predicted_yield_tons_per_ha": 2.71,
  "predicted_production_tons": 6.78,
  "area_ha": 2.5
}
```

### Argument reference

**`recommend-crop`**

| Flag | Description |
|---|---|
| `--n`, `--p`, `--k` | Nitrogen / Phosphorus / Potassium levels |
| `--temperature` | Temperature in °C |
| `--humidity` | Humidity in % |
| `--ph` | Soil pH |
| `--rainfall` | Rainfall in mm |
| `--top-n` | Number of crops to recommend (default 3) |

**`predict-yield`**

| Flag | Description |
|---|---|
| `--state` | State name (must match a state seen during training) |
| `--season` | Season (e.g. Kharif, Rabi — must match training data) |
| `--crop` | Crop name (must match training data) |
| `--area` | Cultivated area in hectares |
| `--year` | Crop year (optional, default 2020) |

Invalid inputs (out-of-range values, unrecognized state/season/crop) return
a clear error message and a non-zero exit code rather than a silent or
incorrect prediction.

---

## Testing

Run the unit test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Missing value handling, outlier clipping, feature engineering, and
  categorical encoding for both pipelines (`tests/test_preprocess.py`)
- Model training (both RandomForest and GradientBoosting options) and
  save/load round-trips (`tests/test_model.py`)
- Input validation logic used by the CLI (`tests/test_predict.py`)

---

## Dataset Note

This project uses **two independent, real, publicly available datasets**
(see `data/DATASET_SETUP.md` for download links), one per task, because no
single public dataset covers both soil-chemistry-driven crop suitability
and area/production yield records together:

1. **Crop Recommendation Dataset** (Kaggle, atharvaingle) — 2,200 rows of
   N, P, K, temperature, humidity, pH, rainfall, and crop label
2. **Crop Production in India** (Kaggle, abhinand05) — state/season/crop/
   area/production records used to derive yield per hectare

Synthetic fallback generators matching each real schema are included so the
pipeline is runnable/demoable without a network connection — **replace
these with the real files before final submission** (see the note in
`data/DATASET_SETUP.md`).

---

## Non-Functional Requirements Addressed

- **Performance** — inference completes in well under a second once models
  are loaded (RandomForest/GradientBoosting inference has no retraining at
  prediction time).
- **Reliability** — every CLI input is validated (type, range, and allowed
  categories) before it reaches the model.
- **Maintainability** — modular `src/` layout, single `config.yaml` for
  paths/hyperparameters, model algorithm swappable without code changes.
- **Logging/Monitoring** — every prediction request and result is logged
  with a timestamp to `logs/app.log`.
- **Scalability** — preprocessing and training operate on the full
  DataFrame in vectorized pandas/sklearn operations, not per-row Python
  loops, so they scale to much larger datasets.
- **Error Handling** — missing model artifacts, malformed config, and bad
  CLI input all fail with clear, actionable messages instead of stack
  traces.

---

## Future Enhancements

- Add a lightweight REST API wrapper around the same `src/` modules
- Hyperparameter tuning via cross-validation (GridSearchCV)
- Add SHAP-based feature importance explanations to predictions
- Join the two datasets on crop name to let yield prediction also factor
  in soil/weather conditions where available
