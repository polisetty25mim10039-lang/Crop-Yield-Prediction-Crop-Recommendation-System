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
│   ├── DATASET_SETUP.md              
│   ├── generate_recommendation_data.py  
│   ├── generate_production_data.py     
│   └── raw/                          
├── src/
│   ├── data_loader.py       
│   ├── preprocess.py        
│   ├── model.py              
│   ├── evaluate.py         
│   ├── predict.py             
│   ├── recommend.py           
│   ├── cli.py                 
│   ├── logger.py             
│   └── utils.py                
├── tests/                    
├── docs/                      
├── models/                   
├── config.yaml               
├── train.py                  
├── main.py                   
├── requirements.txt
└── statement.md               
some changes may be possible in the structure

```
---

# Setup & Installation

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

use kaggle to get thedata sets or download them from data folder in my repository

```bash
python data/generate_recommendation_data.py  
python data/generate_production_data.py       
```

### 5. Train the models

```bash
python train.py
```

output looks like :

<img width="830" height="66" alt="image" src="https://github.com/user-attachments/assets/df0b6b3f-006b-4060-a90f-c88d93c5be71" />

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
THE Result for the test is :

<img width="1151" height="535" alt="image" src="https://github.com/user-attachments/assets/17cf2fe7-b553-4037-a65a-dd2d8991fce9" />

---

## Dataset Note

The project requires two independent, real, publicly available datasets
(check `data/DATASET_SETUP.md` for dataset download instructions), one for
each task, since there are no publicly available datasets that cover both
soil-chemistry based crop suitability and area/production yields at the same
time:

1. **Crop Recommendation Dataset** (Kaggle, atharvaingle) – 2,200 rows of
   N, P, K, temperature, humidity, pH, rainfall, and crop label
2. **Crop Production in India** (Kaggle, abhinand05) – state/season/crop/
   area/production records for calculating the yield per hectare

Two synthetic data generator scripts mimicking the schemas of the two
datasets are provided so that the pipeline can be run/demonstrated without
network access – **they should be replaced with the real datasets before
submission** (check the comment in `data/DATASET_SETUP.md`).

---

## Non-Functional Requirements Addressed

- Performance — inference takes less than a second once the models are
  loaded (RandomForest/GradientBoosting inference does not involve any
  training).
- Reliability — all CLI arguments are validated (type, range, and valid
  categories) before they are passed to the model.
- Maintainability — modular `src/` structure, only one
  `config.yaml` file containing paths and hyperparameters, easy swapping of
  the model algorithm without changing the code.
- Logging/Monitoring — every prediction request and its result are
  logged with a timestamp into `logs/app.log`.
- Scalability — both preprocessing and training use vectorized pandas
  and sklearn operations on the entire DataFrame, rather than iterating
  through it row by row in Python, thus allowing scaling to larger
  datasets.
- Error Handling — missing model artifacts, misconfigured config and
  invalid CLI arguments all fail with descriptive error messages, not
  tracebacks..

   ---
