"""
cli.py
------
Module 3c — Command Line Interface.
Parses arguments and dispatches to either the yield-prediction pipeline
(production dataset) or the crop-recommendation pipeline (recommendation
dataset). Single user-facing entry point (invoked via main.py).
"""

import argparse
import json
import sys

from src.utils import load_config
from src.logger import get_logger
from src.model import load_model
from src.predict import predict_yield
from src.recommend import recommend_crops
from src.utils import ValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crop-cli",
        description="Crop Yield Prediction & Crop Recommendation CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- predict-yield: uses the Production dataset's schema ---
    p_yield = sub.add_parser(
        "predict-yield",
        help="Predict yield/production for a crop in a given state, season and area",
    )
    p_yield.add_argument("--state", required=True, help="State name, e.g. Punjab")
    p_yield.add_argument("--season", required=True, help="Season, e.g. Kharif")
    p_yield.add_argument("--crop", required=True, help="Crop name, e.g. Rice")
    p_yield.add_argument("--area", required=True, type=float, help="Area under cultivation (hectares)")
    p_yield.add_argument("--year", type=int, default=2020, help="Crop year (optional, default 2020)")

    # --- recommend-crop: uses the Recommendation dataset's schema ---
    p_recommend = sub.add_parser(
        "recommend-crop",
        help="Recommend the best crops for given soil/weather conditions",
    )
    p_recommend.add_argument("--n", required=True, type=float, help="Nitrogen level")
    p_recommend.add_argument("--p", required=True, type=float, help="Phosphorus level")
    p_recommend.add_argument("--k", required=True, type=float, help="Potassium level")
    p_recommend.add_argument("--temperature", required=True, type=float, help="Temperature in Celsius")
    p_recommend.add_argument("--humidity", required=True, type=float, help="Humidity percentage")
    p_recommend.add_argument("--ph", required=True, type=float, help="Soil pH")
    p_recommend.add_argument("--rainfall", required=True, type=float, help="Rainfall in mm")
    p_recommend.add_argument("--top-n", type=int, default=3, help="Number of crops to recommend")

    return parser


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config("config.yaml")
    logger = get_logger("crop_cli", config["paths"]["log_file"], config["logging"]["level"])

    try:
        if args.command == "predict-yield":
            encoders = load_model(config["paths"]["yield_encoders"])
            valid_states = list(encoders["State_Name"].classes_)
            valid_seasons = list(encoders["Season"].classes_)
            valid_crops = list(encoders["Crop"].classes_)

            inputs = {
                "state": args.state, "season": args.season, "crop": args.crop,
                "area_ha": args.area, "crop_year": args.year,
            }
            result = predict_yield(inputs, config, encoders, valid_states, valid_seasons, valid_crops)
            logger.info(f"predict-yield request={inputs} result={result}")

        elif args.command == "recommend-crop":
            encoders = load_model(config["paths"]["recommend_encoders"])
            inputs = {
                "N": args.n, "P": args.p, "K": args.k,
                "temperature": args.temperature, "humidity": args.humidity,
                "ph": args.ph, "rainfall": args.rainfall,
            }
            result = recommend_crops(inputs, config, encoders, top_n=args.top_n)
            logger.info(f"recommend-crop request={inputs} result={result}")

        else:
            parser.print_help()
            return 1

    except FileNotFoundError:
        logger.error("Model artifacts not found. Run `python train.py` first.")
        print("Error: model artifacts not found. Run `python train.py` first.", file=sys.stderr)
        return 1
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        print(f"Input error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run())
