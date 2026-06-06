# UrbanJEPA — City-Scale Latent World Model

## What this is

A self-supervised latent world model for predicting how a city evolves under different conditions.

The system ingests heterogeneous city data — traffic, weather, hospitals, energy, finance, crime, retail — normalises it into a shared representation, and learns a compressed latent model of how the city transitions from one state to the next. Given a recent history of the city and a set of hypothetical future conditions (a typhoon warning, a stadium event, a road closure), the model outputs a predicted future city state and can compare scenarios to isolate the city's reaction to that specific condition.

Core equation:

```
past city state + future conditions → predicted future city state
```

```
X_context + C_future → z_future_pred → Ŷ_future
```

The strongest claim the system can make: **given that conditions like this appeared historically, the city tended to respond like this.** It is conditional scenario forecasting, not causal simulation.

---

## Why Hong Kong

All data sources in scope are Hong Kong-specific:

| Source | Signal type | Owner |
|---|---|---|
| Hospital occupancy | district-level bed load | Armand |
| Oil price (WTI) | global macro condition | Noa |
| Crime rate | zonal, monthly | — |
| Energy production/demand | zonal hourly load | — |
| Air quality (AQHI) | zonal, continuous | — |
| Temperature / weather | observed + forecast (HKO) | — |
| Unemployment rate | global, monthly | — |
| Visitors arriving | HKTB/airport daily count | — |
| Water consumption | zonal hourly | — |
| Typhoon/rainstorm warning | HKO signal level, issued | — |
| Hang Seng Index | daily close, intraday | — |
| Retail sales | zonal index, monthly | — |

**Global series** (HSI, oil, unemployment, typhoon signal, visitors) have no zone breakdown — they are broadcast uniformly to all zones in the condition tensor `C_future`.

---

## System architecture

```
Real city sources (per-source loaders in ml/data/loaders/)
        ↓
Smart Data Models validation (TrafficFlowObserved, WeatherObserved,
WeatherForecast, ParkingSpot validated against pinned schema.json;
all other sources passed through as trusted dicts)
        ↓
Flattened observation table (timestamp × zone × feature)
        ↓
Regularised hourly zone panel  X[t, zone, feature]
(z-score normalised on train split; missing mask + quality channel per feature)
        ↓
Windowed tensors  (X_context, C_future, Y_target, static, masks)
        ↓
UrbanJEPA model
        ↓
Decoded future city observations  Ŷ[batch, 24h, zones, 17 features]
        ↓
Scenario comparison:  Ŷ_scenario − Ŷ_baseline = city reaction
```

---

## Tensor contract

All shapes use: `B` = batch, `L` = context steps, `T` = target steps, `Lc` = condition steps, `Z` = zones.

| Tensor | Shape | Description |
|---|---|---|
| `X_context` | `[B, 168, Z, 17×3]` | 7-day lookback; each feature has (value, is_missing, quality) |
| `C_future` | `[B, 144, Z, 16]` | conditions from now to 5 days + 24h target block |
| `Y_target` | `[B, 24, Z, 17]` | 24-hour city state to predict |
| `static` | `[B, Z, 5]` | time-invariant zone features |

17 state features, 16 condition features, 5 static features — all defined as the single source of truth in `ml/data/feature_spec.py`.

---

## Model: UrbanJEPA

Architecture (`ml/models/arch.py`, 7.6M params, 4.8M trainable):

```
CityEncoder
  FeatureEncoder:   cat(value, mask, quality) → Linear → [B, L, Z, d]
  PatchEncoder:     fold 168h into 6h patches (28 patches) → Transformer → mean pool → [B, Z, d]

ConditionEncoder:   Linear → Transformer → mean pool → [B, Z, d]

LatentPredictor:    MLP( cat(z_ctx, z_cond) ) → z_pred [B, Z, d]

TargetEncoder:      EMA copy of CityEncoder; stop-gradient; not directly trained
                    updated each step: θ_target ← τ·θ_target + (1−τ)·θ_online

DecoderHeads:       domain-specific linear heads
                    traffic (3 features), parking (1), environment (2), energy (1)
                    → Ŷ [B, 24, Z, 17]
```

Training loss:

```
L_total = L_latent(cosine_dist(z_pred, z_target.detach()))
        + 0.5 × L_raw(masked_MAE(Ŷ, Y_target))
        + 1e-5 × L2_reg
```

The JEPA objective forces the predictor to anticipate a plausible future latent city state without decoding raw pixels (or raw city data). The decoder is a separate supervised objective that grounds the latent in interpretable observations.

---

## What is already built

### `ml/data/feature_spec.py`
Single source of truth for all 17 state features, 16 condition features, 5 static features. Defines the leakage rule (WeatherObserved → state; WeatherForecast → condition), the SDM-attribute map, and per-domain decoder output indices.

### `ml/data/schemas/` (pinned)
Official Smart Data Models `schema.json` files vendored at fixed commit SHAs:
- `TrafficFlowObserved` (transportation, commit `1eee77e`)
- `WeatherObserved` (weather, commit `c857c0c`)
- `WeatherForecast` (weather, commit `9e6bc8c`)
- `ParkingSpot` (parking, commit `be8db6f`)

### `ml/data/cache.py`
Hash-keyed stage cache. Any ETL stage wrapped with `load_or_run` is resumable — if it fails mid-run, re-running picks up from the last completed stage.

### `ml/data/etl.py`
Orchestrates the full ETL contract: loaders → canonical entities → panel → windowed tensors. Currently runs end-to-end on a synthetic generator. Real loaders drop in without changes to etl.py.

Produces `ml/data/processed/dataset.pt` with splits:
- train: 1482 windows
- val:   249 windows
- test:  249 windows

### `ml/data/loaders/base.py` + `registry.py`
Loader ABC (`loader_id`, `schema_type`, `fetch(start, end, zones)`). Registry auto-discovers all subclasses in the `loaders/` package — new real loaders require no changes to the orchestrator.

### `ml/models/arch.py`
Full UrbanJEPA model as described above. Verified output shapes:

```
z_pred:  (2, 8, 256)   [B, Z, d_model]
z_tgt:   (2, 8, 256)
y_hat:   (2, 24, 8, 17)  [B, T, Z, N_STATE]
```

### `ml/models/train.py`
JEPA trainer with EMA target update, masked MAE decoder loss, gradient clipping, TensorBoard logging per loss component (latent / raw / total), and the existing nx `train` target contract.

Training verified: loss decreases 0.52 → 0.46 over 20 epochs on CPU in ~1 minute.

---

## What is not yet built

### Data pipeline (real sources)
- `ml/data/loaders/synthetic.py` — partially written, needs finishing
- Real per-source loaders (hospital, HSI, HKO typhoon feed, etc.) — arriving from parallel work; just subclass `Loader`
- `ml/data/validate.py` — SDM jsonschema validation + passthrough for non-SDM sources
- `ml/data/flatten.py` — SDM entity → long observation rows
- `ml/data/panel.py` — long → regularised hourly zone panel with proper missing handling

### Evaluation
- `ml/models/eval.py` — latent cos/MSE + masked MAE/RMSE per domain + calibration
- `ml/models/baselines.py` — seasonal-naive and persistence baselines for comparison

### Inference / scenario engine
- `ml/inference.py` — currently a stub; needs the scenario engine:
  run predictor twice (baseline vs modified conditions) and return the per-zone × timestep × feature delta table

### Tests
- Unit tests for feature_spec index consistency, SDM validation, windowing leakage, and arch forward shape/EMA

---

## Running it today

```bash
# generate synthetic data + windows
uv run python -m ml.data.etl

# train
uv run python -m ml.models.train

# serve (once inference.py is finished)
ML_LATEST_WEIGHTS_PATH=ml/models/weights/model.pt uv run uvicorn ml.inference:app --port 8000
```

---

## Key design decisions recorded

**Real loaders = plug-in, not rewrite.** The ETL contract is stable. Each new data source is a file in `ml/data/loaders/` that subclasses `Loader`. The orchestrator aggregates whatever is registered.

**Global series broadcast at panel time.** HSI, oil price, typhoon warnings, unemployment — these have no zone breakdown and are copied uniformly across all zones in `C_future`. No separate model path needed.

**SDM validation only for SDM types.** Non-SDM sources (hospital, retail, crime, HSI, oil) are trusted dicts. Adds no friction for new sources.

**Feature spec is Python, not YAML.** Keeps index maps typed and checked by mypy. New sources require a one-line addition to `feature_spec.py`.

**Leakage rule is encoded in `feature_spec.py`.** `WeatherObserved` and all historically-observed series go into `STATE`. `WeatherForecast` and all forward-issued signals go into `CONDITION`. The flatten stage enforces this.

**EMA target encoder, stop-gradient.** Follows JEPA/BYOL-style training stability: the target branch is an exponential moving average of the online encoder; backprop does not flow through it.

**Decoder is separate from the JEPA objective.** The latent loss trains the representation; the decoder loss grounds it in observable city terms. The two can be weighted independently (`lambda_raw` in config).

**No graph structure yet.** Spatial relations are captured via learned zone embeddings only. The `static` tensor in the model signature is a hook for a future graph encoder (Phase 7 in the plan).

**No causal claims.** The system does conditional scenario forecasting. Calling it a "causal simulation" or "digital twin that predicts interventions" would overstate what is learned.
