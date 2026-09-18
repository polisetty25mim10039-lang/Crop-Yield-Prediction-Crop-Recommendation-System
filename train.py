"""
train.py
--------
End-to-end training pipeline entry point. Trains both models from their
respective source datasets:

  1. Recommendation dataset -> crop recommender (classifier)
  2. Production dataset     -> yield regressor

Run:
    python train.py
"""

import sys

from src.utils import load_config, ensure_parent_dir
from src.logger import get_logger
from src.data_loader import load_recommendation_data, load_production_data, DataLoadError
from src.preprocess import preprocess_recommendation, preprocess_production
from src.model import train_yield_model, train_recommend_model, save_model
from src.evaluate import evaluate_yield_model, evaluate_recommend_model


def main():
    config = load_config("config.yaml")
    logger = get_logger("train", config["paths"]["log_file"], config["logging"]["level"])
    logger.info("Starting training pipeline.")

    try:
        rec_raw = load_recommendation_data(config["paths"]["recommendation_data"])
        prod_raw = load_production_data(config["paths"]["production_data"])
    except DataLoadError as e:
        logger.error(str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1

    logger.info(f"Loaded recommendation dataset: {len(rec_raw)} rows.")
    logger.info(f"Loaded production dataset: {len(prod_raw)} rows.")

    # --- Recommendation pipeline ---
    rec_processed, rec_encoders = preprocess_recommendation(rec_raw, fit=True)
    ensure_parent_dir(config["paths"]["recommend_processed"])
    rec_processed.to_csv(config["paths"]["recommend_processed"], index=False)

    recommend_model, (X_test_r, y_test_r), _ = train_recommend_model(rec_processed, config, logger)
    recommend_metrics = evaluate_recommend_model(recommend_model, X_test_r, y_test_r, logger)
    save_model(recommend_model, config["paths"]["recommend_model"])
    save_model(rec_encoders, config["paths"]["recommend_encoders"])

    # --- Production / yield pipeline ---
    prod_processed, prod_encoders = preprocess_production(prod_raw, fit=True)
    ensure_parent_dir(config["paths"]["production_processed"])
    prod_processed.to_csv(config["paths"]["production_processed"], index=False)

    yield_model, (X_test_y, y_test_y), _ = train_yield_model(prod_processed, config, logger)
    yield_metrics = evaluate_yield_model(yield_model, X_test_y, y_test_y, logger)
    save_model(yield_model, config["paths"]["yield_model"])
    save_model(prod_encoders, config["paths"]["yield_encoders"])

    logger.info("Training pipeline complete.")

    print("\n=== Training Summary ===")
    print(f"Recommend model ({config['recommend_model']['type']}) -> "
          f"Accuracy: {recommend_metrics['accuracy']}, F1: {recommend_metrics['f1_score']}")
    print(f"Yield model ({config['yield_model']['type']})      -> "
          f"RMSE: {yield_metrics['rmse']}, MAE: {yield_metrics['mae']}, R2: {yield_metrics['r2']}")
    print(f"\nModels saved to: {config['paths']['recommend_model']}, {config['paths']['yield_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
