"""
evaluate.py
-----------
Module 2b — Model Evaluation.
Computes and reports metrics for both the yield regressor (RMSE, MAE, R^2)
and the crop recommender (accuracy, precision/recall/F1).
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
)


def evaluate_yield_model(model, X_test, y_test, logger=None) -> dict:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    if logger:
        logger.info(f"Yield model evaluation: {metrics}")
    return metrics


def evaluate_recommend_model(model, X_test, y_test, logger=None) -> dict:
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }
    if logger:
        logger.info(f"Recommendation model evaluation: {metrics}")
    return metrics
