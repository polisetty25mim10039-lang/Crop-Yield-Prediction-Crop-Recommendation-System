import numpy as np
import pandas as pd
import pytest

from src.preprocess import preprocess_recommendation, preprocess_production
from src.model import (
    train_yield_model, train_recommend_model,
    save_model, load_model,
)


@pytest.fixture
def config():
    return {
        "data": {"test_size": 0.25, "random_state": 42},
        "yield_model": {"type": "random_forest_regressor", "n_estimators": 20, "max_depth": 5},
        "recommend_model": {"type": "random_forest_classifier", "n_estimators": 20, "max_depth": 5},
    }


@pytest.fixture
def recommend_processed():
    rng = np.random.default_rng(0)
    n = 150
    labels = rng.choice(["rice", "wheat", "maize"], size=n)
    df = pd.DataFrame({
        "N": rng.uniform(10, 120, n), "P": rng.uniform(10, 120, n), "K": rng.uniform(10, 120, n),
        "temperature": rng.normal(24, 4, n), "humidity": rng.uniform(30, 95, n),
        "ph": rng.uniform(5, 8, n), "rainfall": rng.uniform(40, 300, n),
        "label": labels,
    })
    processed, _ = preprocess_recommendation(df, fit=True)
    return processed


@pytest.fixture
def production_processed():
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "State_Name": rng.choice(["Punjab", "Haryana"], size=n),
        "Season": rng.choice(["Kharif", "Rabi"], size=n),
        "Crop": rng.choice(["Rice", "Wheat", "Maize"], size=n),
        "Crop_Year": rng.integers(2005, 2016, n),
        "Area": rng.uniform(10, 500, n),
        "Production": rng.uniform(20, 2000, n),
    })
    processed, _ = preprocess_production(df, fit=True)
    return processed


def test_train_recommend_model_returns_fitted_model(recommend_processed, config):
    model, (X_test, y_test), features = train_recommend_model(recommend_processed, config)
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_yield_model_returns_fitted_model(production_processed, config):
    model, (X_test, y_test), features = train_yield_model(production_processed, config)
    assert hasattr(model, "predict")
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_gradient_boosting_option_also_trains(recommend_processed, config):
    config["recommend_model"] = {"type": "gradient_boosting_classifier", "n_estimators": 10, "max_depth": 3}
    model, (X_test, y_test), _ = train_recommend_model(recommend_processed, config)
    assert hasattr(model, "predict")


def test_save_and_load_model_roundtrip(production_processed, config, tmp_path):
    model, _, _ = train_yield_model(production_processed, config)
    path = tmp_path / "model.pkl"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert hasattr(loaded, "predict")
