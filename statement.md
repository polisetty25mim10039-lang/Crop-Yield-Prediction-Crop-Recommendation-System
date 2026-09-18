# Problem Statement

## Problem Statement

Farmers and agricultural planners routinely face two related but distinct
decisions: **(1) which crop is best suited to a given plot of land under
current soil and weather conditions**, and **(2) once a crop is chosen, how
much yield can realistically be expected**, which informs input planning
(fertilizer, labor, storage) and financial forecasting. These decisions are
traditionally made from experience and rules of thumb, which do not adapt
well to atypical seasons or unfamiliar plots. This project builds a
data-driven system that addresses both decisions from the same underlying
environmental and soil data.

## Scope of the Project

The system covers:
- Ingesting two independent structured agricultural datasets: a
  soil-chemistry/weather dataset (N, P, K, temperature, humidity, pH,
  rainfall, crop label) and a state/season/area/production dataset
- Cleaning and engineering features from that data (handling missing
  values, outliers, and deriving original indices such as a nutrient
  balance score and an area percentile)
- Training two supervised ML models, each from its matching dataset:
  - A classification model that recommends the most suitable crop(s) from
    soil/weather conditions
  - A regression model that predicts yield (tons/hectare) for a **specified
    crop, state, season, and area**
- Exposing both capabilities through a single command-line interface

Out of scope: real-time weather/satellite data ingestion, mobile/web
front-ends, and multi-year time-series forecasting — these are noted as
future enhancements.

## Target Users

- **Individual farmers / smallholders** deciding what to plant next season
  given their land's soil and expected weather.
- **Agricultural extension officers / advisors** who need a quick,
  explainable second opinion across many farmer queries.
- **Agri-planning analysts** at the state/district level estimating
  aggregate production for policy or supply-chain planning.

## High-Level Features

1. **Crop Recommendation** — given soil and environmental conditions,
   returns the top-N most suitable crops ranked by predicted suitability.
2. **Yield Prediction** — given a chosen crop and the same conditions,
   returns predicted yield per hectare and total expected production for
   the specified area.
3. **Validated CLI Interface** — a single command-line tool with two
   subcommands (`predict-yield`, `recommend-crop`), with strict input
   validation and structured JSON output suitable for scripting or
   integration into other tools.
4. **Reproducible Training Pipeline** — a one-command `train.py` script that
   regenerates both models from raw data, with logged evaluation metrics
   (RMSE/MAE/R² for yield, accuracy/precision/recall/F1 for recommendation).
