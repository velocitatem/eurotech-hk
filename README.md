# UrbanJEPA — Hong Kong City-State Forecasting

UrbanJEPA is a self-supervised latent world model project for forecasting how Hong Kong evolves under different future conditions.

Core idea:

```text
past city state + future conditions → predicted future city state
X_context + C_future → z_future_pred → Ŷ_future
```

This project is for **conditional scenario forecasting**, not causal simulation.

## Why this project exists

The goal is to ingest heterogeneous Hong Kong signals (traffic, weather, parking, energy, macroeconomic conditions, etc.), align them into a shared spatiotemporal tensor, and learn a compact latent representation that can answer:

> “Given conditions like these, how did the city historically tend to respond?”

Planned scenario comparisons use:

```text
Ŷ_scenario - Ŷ_baseline = estimated city reaction
```

## Data scope (Hong Kong-first)

In-scope sources include:
- Traffic flow
- Weather, observed and forecast
- Parking
- Air quality
- Energy demand/production
- Hospital occupancy
- Crime
- Retail activity
- Visitors arrival
- Typhoon/rainstorm signals
- Hang Seng Index
- Oil prices, unemployment, and other macro indicators

Global series with no zone dimension are broadcast to all zones in condition tensors.

## Architecture overview

```text
Source loaders (ml/data/loaders/)
  ↓
SDM validation for pinned schema-backed types
  ↓
Canonical long observation table
  ↓
Regularized hourly zone panel
  ↓
Windowed tensors (X_context, C_future, Y_target, static, masks)
  ↓
UrbanJEPA model (ml/models/arch.py)
  ↓
Decoded forecasts Ŷ[batch, horizon, zone, feature]
  ↓
Scenario delta analysis
```

## Tensor contract

- `X_context`: 7-day lookback of state features plus missing/quality channels
- `C_future`: future condition timeline used for forecasting
- `Y_target`: supervised future target block
- `static`: zone-level invariant context

Feature definitions and index contracts live in `ml/data/feature_spec.py` as the single source of truth.

## Current implementation status

Implemented:
- Core feature registry (`ml/data/feature_spec.py`)
- Pinned Smart Data Models schemas (`ml/data/schemas/`)
- Resumable ETL orchestration (`ml/data/etl.py`, `ml/data/cache.py`)
- Loader abstraction and auto-registry (`ml/data/loaders/base.py`, `registry.py`)
- UrbanJEPA architecture (`ml/models/arch.py`)
- JEPA-style training loop with EMA target encoder (`ml/models/train.py`)

Still in progress:
- Full real-source loaders and the remaining ETL stages (validation, entity flattening, and panel regularization)
- Evaluation and baseline modules
- Inference/scenario engine completion in `ml/inference.py`
- Expanded unit test coverage

## Run the ML pipeline

```bash
# ETL (build dataset windows)
uv run python -m ml.data.etl

# Train UrbanJEPA
uv run python -m ml.models.train

# Serve inference API (after inference.py scenario engine is ready)
# Default training output: ml/models/weights/model.pt
ML_LATEST_WEIGHTS_PATH=ml/models/weights/model.pt uv run uvicorn ml.inference:app --port 8000
```

## Repository structure

```text
ml/
  data/          ETL pipeline, schema validation, loaders, processed artifacts
  models/        UrbanJEPA architecture + training loop
  inference.py   Scenario inference API (stub/in progress)

apps/
  webapp/        Next.js app
  backend/       FastAPI and Flask services
  worker/        Celery worker

dlib/            Shared Python utilities
```

## Design principles

- Real data sources plug in via loader subclasses, without ETL rewrites
- SDM validation is applied where schema-backed contracts exist
- Leakage boundaries (observed vs forecast features) are explicitly encoded
- Latent JEPA objective and decoder reconstruction objective are separated
- Claims remain conditional and data-grounded (no causal overstatement)

For the full technical vision and detailed roadmap, see `IDEA.md`.
