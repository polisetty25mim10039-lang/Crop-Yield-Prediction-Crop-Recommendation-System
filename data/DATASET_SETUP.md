# Dataset Setup

This project intentionally uses **two independent, real, publicly available
datasets** — one per task — rather than a single fabricated dataset, because
no single public dataset covers both soil-chemistry-driven crop suitability
*and* area/production yield records together.

## 1. Crop Recommendation Dataset (used by `recommend-crop`)

- **Source**: Kaggle — "Crop Recommendation Dataset" (atharvaingle)
  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Rows**: 2,200
- **Columns**: `N,P,K,temperature,humidity,ph,rainfall,label`
- **Download and place at**: `data/raw/crop_recommendation.csv`

```bash
# Using the Kaggle CLI (requires kaggle.json API credentials):
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/raw --unzip
mv data/raw/Crop_recommendation.csv data/raw/crop_recommendation.csv
```

Or download the CSV manually from the Kaggle page and rename/place it as above.

## 2. Crop Production Dataset (used by `predict-yield`)

- **Source**: Kaggle — "Crop Production in India"
  https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
- **Columns used**: `State_Name,District_Name,Crop_Year,Season,Crop,Area,Production`
  (District_Name is present in the original file but not used here)
- **Download and place at**: `data/raw/crop_production.csv`

```bash
kaggle datasets download -d abhinand05/crop-production-in-india -p data/raw --unzip
mv "data/raw/crop_production.csv" data/raw/crop_production.csv   # already correctly named
```

## No internet access right now / working before you've downloaded real data?

Run the synthetic fallback generators — they produce data in the **exact
same column schema** as the real datasets above, so no code changes are
needed when you later swap in the real files:

```bash
python data/generate_recommendation_data.py   # -> data/raw/crop_recommendation.csv
python data/generate_production_data.py       # -> data/raw/crop_production.csv
```

