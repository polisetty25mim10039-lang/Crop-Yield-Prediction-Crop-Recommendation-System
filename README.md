# Crop Yield Prediction & Recommendation System

Command Line AI/ML system consisting of two independent functionalities, each of them is being trained on its own dataset:
Crop recommendation — recommends the top-N best-fitted crops depending on the soil (Nitrogen, Phosphorus, Potassium, pH) and weather (temperature, humidity, rainfall) characteristics along with the corresponding fitness score;
Yield prediction — predicts the yield amount (tons/hectare) and total amount of product depending on the state, season, crop, and cultivation area.
This system was implemented as a part of the project on Fundamentals of AI and ML course, illustrating the whole process of developing a Machine Learning solution from data ingestion to preprocessing/feature engineering, training and evaluating a model and then getting an output through command line interface.

## Features
Two separate ML models that are independently trained using their own real datasets:
RandomForestClassifier (or GradientBoostingClassifier) for crop recommendation;
RandomForestRegressor (or GradientBoostingRegressor) for yield prediction — can be changed by setting the algorithm key in config.yaml file;
Two engineered features:
npk_balance_score — how well-balanced the ratio between nitrogen-phosphorus-potassium content of a particular example is (value range is [0; 1] and does not depend on the actual numbers);
area_percentile — is a percentile of the cultivation area of the example relative to all the areas of this particular crop;
Parameter validation with informative error messages;
Logging into console and logs/app.log file;
Configuration file-driven file paths and parameters (config.yaml) without any hard-coded values in the code;
CLI-only;
Unit tests for feature engineering and training stages and parameter validation

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
git clone https://github.com/polisetty25mim10039-lang/Crop-Yield-Prediction-Crop-Recommendation-System.git
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
venv\Scripts\Activate.ps1        # powershell: .\venv\Scripts\Activate.ps1
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
python data/generate_recommendation_data.py  
python data/generate_production_data.py       
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
python main.py recommend-crop --n 90 --p 42 --k 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 202.9 --top-n 3
```

Output:
<img width="320" height="293" alt="image" src="https://github.com/user-attachments/assets/c74eaf5b-fbf7-4513-95e8-a55b9d2cdeed" />


### Predict yield/production for a chosen crop

```bash
python main.py predict-yield --state Punjab --season Kharif --crop Rice --area 2.5 
```

Output:

<img width="403" height="154" alt="image" src="https://github.com/user-attachments/assets/95f9e4ed-35cd-4633-91aa-3555fd9a3041" />



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
