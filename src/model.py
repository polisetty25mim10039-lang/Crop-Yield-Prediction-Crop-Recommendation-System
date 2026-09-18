"""
model.py
--------
Module 2 — Model Training.

Trains two independent models, each from its own dataset:
  1. Crop Recommender  — RandomForestClassifier or GradientBoostingClassifier
     on the recommendation dataset (N, P, K, weather -> crop label)
  2. Yield Regressor   — RandomForestRegressor or GradientBoostingRegressor
     on the production dataset (state/season/crop/area -> yield per ha)

The algorithm for each is selected via config.yaml ("type" field), so the
same training code supports a straightforward model comparison for the
report's "Design Decisions & Rationale" section.
"""

import joblib
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split

from src.preprocess import get_recommend_features, get_production_features

REGRESSORS = {
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
}

CLASSIFIERS = {
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
}


def _build_estimator(registry: dict, params: dict, random_state: int):
    model_type = params["type"]
    if model_type not in registry:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {list(registry)}")
    cls = registry[model_type]

    kwargs = {"random_state": random_state}
    if "n_estimators" in params:
        kwargs["n_estimators"] = params["n_estimators"]
    if "max_depth" in params:
        kwargs["max_depth"] = params["max_depth"]
    if model_type.startswith("gradient_boosting") and "learning_rate" in params:
        kwargs["learning_rate"] = params["learning_rate"]

    return cls(**kwargs)


def train_yield_model(df, config, logger=None):
    """Train a regressor to predict yield_tons_per_ha from the production dataset."""
    features = get_production_features()
    X = df[features]
    y = df["yield_tons_per_ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    model = _build_estimator(REGRESSORS, config["yield_model"], config["data"]["random_state"])
    model.fit(X_train, y_train)

    if logger:
        logger.info(
            f"Trained yield model ({config['yield_model']['type']}) "
            f"on {len(X_train)} samples, {len(features)} features."
        )

    return model, (X_test, y_test), features


def train_recommend_model(df, config, logger=None):
    """Train a classifier to predict the crop label from the recommendation dataset."""
    features = get_recommend_features()
    X = df[features]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    model = _build_estimator(CLASSIFIERS, config["recommend_model"], config["data"]["random_state"])
    model.fit(X_train, y_train)

    if logger:
        logger.info(
            f"Trained recommendation model ({config['recommend_model']['type']}) "
            f"on {len(X_train)} samples, {len(features)} features."
        )

    return model, (X_test, y_test), features


def save_model(model, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
