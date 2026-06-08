![](./banner.png)

A self-supervised world model for predicting how Hong Kong evolves under different conditions. The system ingests heterogeneous city data (traffic, weather, hospitals, energy, air quality, finance), encodes it into a shared latent representation, and predicts future city states given a description of what changed (top news).

```
past city state + transition description → predicted future city state
```

The model follows a JEPA architecture: an online encoder and an EMA target encoder learn jointly via cosine similarity in latent space, with a decoder head grounding predictions in observable units. Transition events are fed as pre-computed text embeddings (Qwen3-Embedding-0.6B, 1024-dim) that condition the latent predictor.

**Current finding:** the 1024-dim text embeddings introduce more representational capacity than the training data can constrain. The model fits the text pathway quickly on whatever co-occurrence patterns exist in the training windows and overfits — latent loss drops fast early then plateaus while validation MAE diverges. Without a large corpus of labelled city events, text conditioning adds complexity without adding generalisation. Cross-signal correlations between city domains (weather × finance etc.) are also weak at accessible timescales, meaning the encoder cannot exploit simple linear structure — the city-state transition is genuinely hard to learn from a 2-year panel.

---

## Quick Start

```bash
cp .env.example .env
make init
```

## ML

```bash
bun x nx run ml:etl
bun x nx run ml:train
```

Config lives in `ml/configs/`. Artifacts go to `ml/data/processed/`, `ml/models/weights/`, and `ml/tensorboard/`.

## Directory

```
ml/
  configs/    YAML hyperparameters (data + training)
  models/     arch.py (UrbanJEPA) + train.py
  data/       ETL pipeline + loaders + processed artifacts
  inference.py
apps/
  webapp/     Next.js dashboard
  worker/     Celery background worker
dlib/         Shared utilities (tracing, scraper, agent)
```
