"""ETL orchestrator — produces dataset.pt with chrono-split windowed tensors.

Current mode: synthetic windows generated directly from feature_spec dimensions.
Real loaders (per-source adapters) will slot in here once they arrive;
the output contract (split keys, tensor shapes) stays fixed.

Output dataset.pt schema:
    {
      "train": split_dict,
      "val":   split_dict,
      "test":  split_dict,
      "meta":  {...},
    }
    split_dict keys:
      values_ctx  [N, context_len, n_zones, N_STATE]  float32  (z-score normed)
      masks_ctx   [N, context_len, n_zones, N_STATE]  bool
      quality_ctx [N, context_len, n_zones, N_STATE]  float32
      cond_future [N, cond_len,    n_zones, N_COND]   float32
      static      [N, n_zones,     N_STATIC]           float32
      values_tgt  [N, target_len,  n_zones, N_STATE]  float32
      masks_tgt   [N, target_len,  n_zones, N_STATE]  bool
"""
from __future__ import annotations
import argparse, csv, io, json, math
from pathlib import Path

import numpy as np
import torch
import yaml

from dlib import get_logger
from ml.data.feature_spec import N_STATE, N_COND, N_STATIC
from ml.data.cache import load_or_run

try:
    from ml.data.openmeteo import fetch_historical, _CACHE as _OM_CACHE
    from ml.data.storage import Storage, make_storage
except ImportError:
    from openmeteo import fetch_historical, _CACHE as _OM_CACHE  # type: ignore[no-redef]
    from storage import Storage, make_storage  # type: ignore[no-redef]

logger = get_logger("ml-etl")


def _time_encodings(n_hours: int) -> np.ndarray:
    """Cyclical hour/dow encodings, shape [n_hours, 4]."""
    h  = np.arange(n_hours) % 24
    d  = (np.arange(n_hours) // 24) % 7
    return np.stack([
        np.sin(2 * math.pi * h / 24),
        np.cos(2 * math.pi * h / 24),
        np.sin(2 * math.pi * d / 7),
        np.cos(2 * math.pi * d / 7),
    ], axis=-1).astype(np.float32)


def _synthetic_panel(
    n_hours: int,
    n_zones: int,
    seed: int,
    missing_rate: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (values, masks, quality, conditions), each [n_hours, n_zones, F]."""
    rng = np.random.default_rng(seed)
    h   = np.arange(n_hours)

    traffic_base = (
        0.4
        + 0.35 * np.exp(-((h % 24 - 8.5) ** 2) / 2.5)
        + 0.40 * np.exp(-((h % 24 - 18.0) ** 2) / 3.0)
    ).clip(0, 1)[:, None]

    doy  = (h // 24) % 365
    temp = 23.5 - 8.5 * np.cos(2 * math.pi * (doy - 15) / 365) + \
           3.0 * np.sin(2 * math.pi * (h % 24) / 24)

    z_off = rng.uniform(0, 0.2, size=(1, n_zones))

    values = np.zeros((n_hours, n_zones, N_STATE), dtype=np.float32)
    traffic = (traffic_base + z_off + rng.normal(0, 0.05, (n_hours, n_zones))).clip(0, 1)
    values[:, :, 0] = traffic * 1200
    values[:, :, 1] = np.maximum(5, 80 * (1 - traffic * 0.7) + rng.normal(0, 3, (n_hours, n_zones)))
    values[:, :, 2] = traffic
    values[:, :, 3] = (traffic * 0.85 + rng.normal(0, 0.05, (n_hours, n_zones))).clip(0, 1)
    values[:, :, 4] = np.maximum(0, rng.normal(2, 1, (n_hours, n_zones)))
    values[:, :, 5] = np.maximum(0, traffic * 800 + rng.normal(0, 50, (n_hours, n_zones)))
    values[:, :, 6] = (1 + 7 * traffic + rng.normal(0, 0.5, (n_hours, n_zones))).clip(1, 10)
    values[:, :, 7] = (45 + 20 * traffic + rng.normal(0, 3, (n_hours, n_zones))).clip(30, 90)
    values[:, :, 8] = (temp[:, None] + rng.normal(0, 1.5, (n_hours, n_zones))).astype(np.float32)
    values[:, :, 9] = (75 + 10 * np.cos(2 * math.pi * (doy - 180) / 365)[:, None] +
                       rng.normal(0, 5, (n_hours, n_zones))).clip(30, 100)
    values[:, :, 10] = (0.55 + 0.15 * ((h // 24) % 7 == 0)[:, None] +
                        rng.normal(0, 0.04, (n_hours, n_zones))).clip(0, 1)
    values[:, :, 11] = np.maximum(0, 2.5 + z_off * 4 + rng.normal(0, 0.3, (n_hours, n_zones)))
    values[:, :, 12] = np.maximum(0, 50 + 30 * traffic + (temp[:, None] - 15) * 3 +
                                  rng.normal(0, 5, (n_hours, n_zones)))
    values[:, :, 13] = np.maximum(0, 100 + 40 * ((h // 24) % 7 >= 5)[:, None] +
                                  rng.normal(0, 10, (n_hours, n_zones)))
    values[:, :, 14] = np.maximum(0, 80 + 20 * traffic + rng.normal(0, 8, (n_hours, n_zones)))
    hsi = np.cumprod(1 + rng.normal(0, 0.003, n_hours)) * 20000
    values[:, :, 15] = hsi[:, None]
    visitors = np.maximum(0, 180000 + 60000 * ((h // 24) % 7 >= 5)[:, None] +
                          rng.normal(0, 15000, (n_hours, n_zones)))
    values[:, :, 16] = visitors

    masks   = rng.random((n_hours, n_zones, N_STATE)) > missing_rate
    quality = np.where(masks, 1.0, 0.0).astype(np.float32)

    typhoon = ((doy >= 152) & (doy <= 304) & (rng.random(n_hours) < 0.004)).astype(np.float32)
    rain_p  = (0.25 + 0.25 * np.sin(2 * math.pi * (doy - 90) / 365)).astype(np.float32)
    oil     = (75 + 10 * np.sin(2 * math.pi * doy / 365) + rng.normal(0, 2, n_hours)).astype(np.float32)
    unemp   = np.full(n_hours, 3.2, dtype=np.float32) + rng.normal(0, 0.1, n_hours).astype(np.float32)
    t_enc   = _time_encodings(n_hours)

    cond = np.zeros((n_hours, n_zones, N_COND), dtype=np.float32)
    cond[:, :, 0] = (temp[:, None] + rng.normal(0, 1, (n_hours, n_zones))).astype(np.float32)
    cond[:, :, 1] = (rain_p[:, None] + rng.normal(0, 0.1, (n_hours, n_zones))).clip(0, 1)
    cond[:, :, 2] = np.maximum(0, 15 + rng.normal(0, 8, (n_hours, n_zones)))
    cond[:, :, 3] = (rain_p > 0.5).astype(np.float32)[:, None]
    cond[:, :, 4] = typhoon[:, None]
    cond[:, :, 5] = (rng.choice([0, 1, 2, 3], p=[0.70, 0.18, 0.08, 0.04], size=n_hours)[:, None]).astype(np.float32)
    for enc_col in range(6, 10):
        cond[:, :, enc_col] = t_enc[:, enc_col - 6][:, None]
    cond[:, :, 10] = rng.poisson(0.3, (n_hours, n_zones)).astype(np.float32)
    cond[:, :, 11] = rng.integers(0, 5, (n_hours, n_zones)).astype(np.float32)
    cond[:, :, 12] = ((h // 24) % 7 == 0).astype(np.float32)[:, None]
    cond[:, :, 13] = (((h // 24) % 7 < 5) & ((h // 24 + 30) % 90 > 7)).astype(np.float32)[:, None]
    cond[:, :, 14] = unemp[:, None]
    cond[:, :, 15] = oil[:, None]

    return values, masks.astype(bool), quality, cond


def _normalise(values: np.ndarray, train_end: int) -> tuple[np.ndarray, dict]:
    mu  = values[:train_end].mean(axis=(0, 1), keepdims=True)
    std = values[:train_end].std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    return ((values - mu) / std).astype(np.float32), {"mu": mu.tolist(), "std": std.tolist()}


def _make_windows(
    values:  np.ndarray,
    masks:   np.ndarray,
    quality: np.ndarray,
    cond:    np.ndarray,
    static:  np.ndarray,
    context_len: int,
    horizon: int,
    target_len: int,
    stride: int,
    t_start: int,
    t_end: int,
) -> dict[str, torch.Tensor]:
    cond_len = horizon + target_len
    idxs = range(t_start, t_end - context_len - cond_len + 1, stride)
    if not idxs:
        idxs = range(t_start, max(t_start + 1, t_end - context_len - cond_len + 1), 1)

    vc, mc, qc, cf, vt, mt = [], [], [], [], [], []
    for i in idxs:
        ctx_s, ctx_e = i, i + context_len
        tgt_s = ctx_e + horizon
        tgt_e = tgt_s + target_len
        if tgt_e > len(values):
            break
        vc.append(values[ctx_s:ctx_e])
        mc.append(masks[ctx_s:ctx_e])
        qc.append(quality[ctx_s:ctx_e])
        cf.append(cond[ctx_e:tgt_e])
        vt.append(values[tgt_s:tgt_e])
        mt.append(masks[tgt_s:tgt_e])

    N = len(vc)
    return {
        "values_ctx":  torch.tensor(np.stack(vc),  dtype=torch.float32),
        "masks_ctx":   torch.tensor(np.stack(mc),  dtype=torch.bool),
        "quality_ctx": torch.tensor(np.stack(qc),  dtype=torch.float32),
        "cond_future": torch.tensor(np.stack(cf),  dtype=torch.float32),
        "static":      torch.tensor(np.tile(static, (N, 1, 1)), dtype=torch.float32),
        "values_tgt":  torch.tensor(np.stack(vt),  dtype=torch.float32),
        "masks_tgt":   torch.tensor(np.stack(mt),  dtype=torch.bool),
    }


def build_dataset(cfg: dict, output_dir: Path, cache_dir: Path) -> Path:
    n_zones     = int(cfg["n_zones"])
    context_len = int(cfg["context_len"])
    horizon     = int(cfg["horizon"])
    target_len  = int(cfg["target_len"])
    stride      = int(cfg.get("stride", 6))
    seed        = int(cfg.get("seed", 42))
    n_hours     = int(cfg.get("total_hours", 8760 + 2190 + 2190))
    train_frac  = float(cfg.get("train_frac", 0.70))
    val_frac    = float(cfg.get("val_frac",   0.15))
    gap         = int(cfg.get("split_gap_hours", 168))

    params = {k: cfg.get(k) for k in
              ("n_zones", "context_len", "horizon", "target_len", "stride", "seed",
               "total_hours", "train_frac", "val_frac", "split_gap_hours")}

    def _build() -> dict:
        logger.info("generating synthetic panel")
        values, masks, quality, cond = _synthetic_panel(n_hours, n_zones, seed)

        train_end = int(n_hours * train_frac)
        val_end   = int(n_hours * (train_frac + val_frac))

        values, scaler = _normalise(values, train_end)

        static = np.zeros((n_zones, N_STATIC), dtype=np.float32)
        static[:, 4] = np.arange(n_zones, dtype=np.float32)

        kw = dict(values=values, masks=masks, quality=quality, cond=cond, static=static,
                  context_len=context_len, horizon=horizon, target_len=target_len, stride=stride)

        logger.info("building windows")
        train_split = _make_windows(**kw, t_start=0,              t_end=train_end)
        val_split   = _make_windows(**kw, t_start=train_end+gap,  t_end=val_end)
        test_split  = _make_windows(**kw, t_start=val_end+gap,    t_end=n_hours)

        logger.info(f"train={len(train_split['values_ctx'])} val={len(val_split['values_ctx'])} test={len(test_split['values_ctx'])}")
        return {"train": train_split, "val": val_split, "test": test_split, "scaler": scaler}

    blob = load_or_run(cache_dir, "windows", params, _build)

    dataset_path = output_dir / "dataset.pt"
    torch.save(blob, dataset_path)

    meta = {
        "n_zones":       n_zones,
        "n_state":       N_STATE,
        "n_cond":        N_COND,
        "n_static":      N_STATIC,
        "context_len":   context_len,
        "horizon":       horizon,
        "target_len":    target_len,
        "train_samples": len(blob["train"]["values_ctx"]),
        "val_samples":   len(blob["val"]["values_ctx"]),
        "test_samples":  len(blob["test"]["values_ctx"]),
        "scaler":        blob["scaler"],
        "dataset_path":  str(dataset_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"saved dataset to {dataset_path}")
    return dataset_path


def build_weather_dataset(storage: Storage, cache_dir: Path | None = None) -> dict:
    raw  = fetch_historical(cache_dir=cache_dir or _OM_CACHE)
    cols = list(raw["hourly"].keys())
    times = raw["hourly"]["time"]

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(zip(*[raw["hourly"][c] for c in cols]))

    meta = {
        "source": "open-meteo-archive",
        "latitude": raw["latitude"],
        "longitude": raw["longitude"],
        "timezone": raw["timezone"],
        "units": raw["hourly_units"],
        "variables": cols,
        "start": times[0],
        "end": times[-1],
        "n_records": len(times),
    }
    storage.put("hk_temperature.csv", buf.getvalue())
    storage.put("hk_temperature_metadata.json", json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "weather"], default="synthetic")
    p.add_argument("--storage", choices=["local", "r2", "both"], default="local")
    p.add_argument("--config", default="ml/configs/data/default.yaml")
    p.add_argument("--output", default="ml/data/processed")
    args = p.parse_args()

    output_dir = Path(args.output)

    if args.source == "weather":
        storage = make_storage(args.storage, output_dir)
        meta = build_weather_dataset(storage)
        print(f"Exported {meta['n_records']} records via {args.storage}")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"
    build_dataset(cfg, output_dir, cache_dir)


if __name__ == "__main__":
    main()
