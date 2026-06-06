"""ETL orchestrator — produces dataset.pt with chrono-split windowed tensors.

Builds an hourly panel [T, Z, N_STATE] from registered Loader subclasses.
Daily records (e.g. immigration) are forward-filled across all 24 hours of
their calendar day. Global features (zone_id='global') are broadcast to all
zones. Unobserved positions stay zero with mask=False.

Output dataset.pt schema:
    {
      "train": split_dict,
      "val":   split_dict,
      "test":  split_dict,
      "scaler": {"mu": ..., "std": ...},
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
import pandas as pd
import torch
import yaml

from datetime import datetime, timedelta

from dlib import get_logger
from ml.data.feature_spec import (
    N_STATE, N_COND, N_STATIC,
    STATE_IDX, COND_IDX, NON_SDM_FEATURE_MAP, SDM_ATTR_MAP,
)
from ml.data.cache import load_or_run
from ml.data.loaders.registry import get_all_loaders
from ml.data.loaders.base import Loader

try:
    from ml.data.openmeteo import fetch_historical, _CACHE as _OM_CACHE
    from ml.data.storage import Storage, make_storage
    from ml.data import ha_hospitals, centanet, aed, hkma
except ImportError:
    from openmeteo import fetch_historical, _CACHE as _OM_CACHE  # type: ignore[no-redef]
    from storage import Storage, make_storage  # type: ignore[no-redef]
    import ha_hospitals, centanet, aed, hkma  # type: ignore[no-redef]

logger = get_logger("ml-etl")


def _state_features(loader: Loader) -> list[tuple[str, int]]:
    """(record_key, state_idx) pairs for this loader's state-group features."""
    if loader.schema_type in NON_SDM_FEATURE_MAP:
        ft, grp = NON_SDM_FEATURE_MAP[loader.schema_type]
        return [(ft, STATE_IDX[ft])] if grp == "state" and ft in STATE_IDX else []
    return [
        (attr, STATE_IDX[ft])
        for (stype, attr), (ft, grp) in SDM_ATTR_MAP.items()
        if stype == loader.schema_type and grp == "state" and ft in STATE_IDX
    ]


def _cond_features(loader: Loader) -> list[tuple[str, int]]:
    """(record_key, cond_idx) pairs for this loader's condition-group features."""
    if loader.schema_type in NON_SDM_FEATURE_MAP:
        ft, grp = NON_SDM_FEATURE_MAP[loader.schema_type]
        return [(ft, COND_IDX[ft])] if grp == "condition" and ft in COND_IDX else []
    return [
        (attr, COND_IDX[ft])
        for (stype, attr), (ft, grp) in SDM_ATTR_MAP.items()
        if stype == loader.schema_type and grp == "condition" and ft in COND_IDX
    ]


def _time_encodings(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical hour/DOW encodings anchored to real timestamps, shape [T, 4]."""
    h, d = timestamps.hour.values, timestamps.dayofweek.values
    return np.stack([
        np.sin(2 * math.pi * h / 24), np.cos(2 * math.pi * h / 24),
        np.sin(2 * math.pi * d / 7),  np.cos(2 * math.pi * d / 7),
    ], axis=-1).astype(np.float32)


def _fill_panel(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    feats: list[tuple[str, int]],
    records,
    ts_to_h: dict,
    z_to_i: dict,
    n_zones: int,
    state: bool,
) -> int:
    values, masks, quality = arrays
    n_rec = 0
    for rec in records:
        zone = rec["zone_id"]
        zs   = list(range(n_zones)) if zone == "global" else \
               [z_to_i[zone]] if zone in z_to_i else []
        if not zs:
            continue
        t0 = pd.Timestamp(rec["timestamp"]).replace(tzinfo=None)
        hs = ([ts_to_h[t0 + pd.Timedelta(hours=h)]
               for h in range(24) if (t0 + pd.Timedelta(hours=h)) in ts_to_h]
              if t0.hour == 0 else
              [ts_to_h[t0]] if t0 in ts_to_h else [])
        for h in hs:
            for z in zs:
                for key, fi in feats:
                    if key in rec:
                        values[h, z, fi] = float(rec[key])
                        if state:
                            masks[h, z, fi]   = True
                            quality[h, z, fi] = 1.0
        n_rec += 1
    return n_rec


def _build_panel(
    loaders: list[Loader],
    timestamps: pd.DatetimeIndex,
    n_zones: int,
    zone_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_hours = len(timestamps)
    ts_to_h = {ts: i for i, ts in enumerate(timestamps)}
    z_to_i  = {z: i for i, z in enumerate(zone_ids)}
    start, end = timestamps[0].isoformat(), timestamps[-1].isoformat()

    values  = np.zeros((n_hours, n_zones, N_STATE), np.float32)
    masks   = np.zeros((n_hours, n_zones, N_STATE), bool)
    quality = np.zeros((n_hours, n_zones, N_STATE), np.float32)

    for loader in loaders:
        feats = _state_features(loader)
        if not feats:
            logger.debug(f"skip loader={loader.loader_id} (no state mapping)")
            continue
        n = _fill_panel((values, masks, quality), feats,
                        loader.fetch(start, end, zone_ids),
                        ts_to_h, z_to_i, n_zones, state=True)
        logger.info(f"loader={loader.loader_id} state_records={n}")

    return values, masks, quality


def _build_cond(
    timestamps: pd.DatetimeIndex,
    n_zones: int,
    zone_ids: list[str],
    loaders: list[Loader] | None = None,
) -> np.ndarray:
    """[T, Z, N_COND] panel — time encodings always present; condition loaders fill the rest."""
    cond = np.zeros((len(timestamps), n_zones, N_COND), np.float32)
    enc  = _time_encodings(timestamps)
    for ci, name in enumerate(("hour_sin", "hour_cos", "dow_sin", "dow_cos")):
        cond[:, :, COND_IDX[name]] = enc[:, ci, None]

    if not loaders:
        return cond

    ts_to_h = {ts: i for i, ts in enumerate(timestamps)}
    z_to_i  = {z: i for i, z in enumerate(zone_ids)}
    start, end = timestamps[0].isoformat(), timestamps[-1].isoformat()
    # reuse _fill_panel with the cond array; masks/quality placeholders unused here
    _unused = np.empty(0), np.empty(0)
    for loader in loaders:
        feats = _cond_features(loader)
        if not feats:
            continue
        n = _fill_panel((cond, *_unused), feats,
                        loader.fetch(start, end, zone_ids),
                        ts_to_h, z_to_i, n_zones, state=False)
        logger.info(f"loader={loader.loader_id} cond_records={n}")

    return cond


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
        tgt_s, tgt_e = ctx_e + horizon, ctx_e + horizon + target_len
        if tgt_e > len(values):
            break
        vc.append(values[ctx_s:ctx_e]); mc.append(masks[ctx_s:ctx_e])
        qc.append(quality[ctx_s:ctx_e]); cf.append(cond[ctx_e:tgt_e])
        vt.append(values[tgt_s:tgt_e]); mt.append(masks[tgt_s:tgt_e])

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
    start_date  = cfg["start_date"]
    n_hours     = int(cfg["total_hours"])
    train_frac  = float(cfg.get("train_frac", 0.70))
    val_frac    = float(cfg.get("val_frac",   0.15))
    gap         = int(cfg.get("split_gap_hours", 168))
    zone_ids    = cfg.get("zone_ids", [f"zone_{i}" for i in range(n_zones)])

    params = {k: cfg.get(k) for k in
              ("n_zones", "context_len", "horizon", "target_len", "stride",
               "start_date", "total_hours", "train_frac", "val_frac", "split_gap_hours")}

    loaders    = get_all_loaders()
    timestamps = pd.date_range(start_date, periods=n_hours, freq="h")

    def _build() -> dict:
        logger.info(f"building panel: start={start_date} n_hours={n_hours} loaders={[l.loader_id for l in loaders]}")
        values, masks, quality = _build_panel(loaders, timestamps, n_zones, zone_ids)
        cond = _build_cond(timestamps, n_zones, zone_ids, loaders)

        train_end = int(n_hours * train_frac)
        val_end   = int(n_hours * (train_frac + val_frac))
        values, scaler = _normalise(values, train_end)

        static = np.zeros((n_zones, N_STATIC), np.float32)
        static[:, 4] = np.arange(n_zones, dtype=np.float32)

        kw = dict(values=values, masks=masks, quality=quality, cond=cond, static=static,
                  context_len=context_len, horizon=horizon, target_len=target_len, stride=stride)
        train_split = _make_windows(**kw, t_start=0,             t_end=train_end)
        val_split   = _make_windows(**kw, t_start=train_end+gap, t_end=val_end)
        test_split  = _make_windows(**kw, t_start=val_end+gap,   t_end=n_hours)

        logger.info(f"train={len(train_split['values_ctx'])} val={len(val_split['values_ctx'])} test={len(test_split['values_ctx'])}")
        return {"train": train_split, "val": val_split, "test": test_split, "scaler": scaler}

    blob = load_or_run(cache_dir, "windows", params, _build)
    dataset_path = output_dir / "dataset.pt"
    torch.save(blob, dataset_path)

    meta = {
        "n_zones": n_zones, "n_state": N_STATE, "n_cond": N_COND, "n_static": N_STATIC,
        "context_len": context_len, "horizon": horizon, "target_len": target_len,
        "start_date": start_date, "n_hours": n_hours,
        "train_samples": len(blob["train"]["values_ctx"]),
        "val_samples":   len(blob["val"]["values_ctx"]),
        "test_samples":  len(blob["test"]["values_ctx"]),
        "scaler": blob["scaler"], "dataset_path": str(dataset_path),
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
        "latitude": raw["latitude"], "longitude": raw["longitude"],
        "timezone": raw["timezone"], "units": raw["hourly_units"],
        "variables": cols, "start": times[0], "end": times[-1],
        "n_records": len(times),
    }
    storage.put("hk_temperature.csv", buf.getvalue())
    storage.put("hk_temperature_metadata.json", json.dumps(meta, indent=2))
    return meta


def build_hospital_dataset(storage: Storage, cache_dir: Path | None = None) -> dict[str, int]:
    cd = cache_dir or _OM_CACHE
    datasets: dict[str, "pd.DataFrame"] = {
        "hk_patientday.csv": ha_hospitals.tidy_patientday(cd),
        "hk_patientday_age_gender.csv": ha_hospitals.tidy_patientday_age_gender(cd),
        "hk_ip_genspec.csv": ha_hospitals.tidy_ip_genspec(cd),
        "hk_ahip_attendances.csv": ha_hospitals.tidy_ahip_attnd(cd),
        "hk_disease_group.csv": ha_hospitals.tidy_disease_group(cd),
        "hk_agg_patient_days.csv": ha_hospitals.aggregate_patient_days(cd),
        "hk_agg_ahip.csv": ha_hospitals.aggregate_ahip(cd),
    }
    for key, df in datasets.items():
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        storage.put(key, buf.getvalue())
    return {k: len(v) for k, v in datasets.items()}


def build_aed_dataset(
    storage: Storage,
    start: datetime | None = None,
    end: datetime | None = None,
    cache_dir: Path | None = None,
    workers: int = 8,
    rate_sec: float = 0.1,
) -> int:
    end = end or datetime.now().replace(second=0, microsecond=0, minute=(datetime.now().minute // 15) * 15)
    start = start or end - timedelta(days=730)
    df = aed.fetch_range(start, end, cache_dir or aed._CACHE, workers, rate_sec)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    storage.put("hk_aed_wait_times.csv", buf.getvalue())
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "weather", "hospitals", "cci", "aed", "hkma"], default="synthetic")
    parser.add_argument("--days", type=int, default=730, help="history window for aed (default 730 = 2 years)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent fetch workers for aed")
    parser.add_argument("--rate", type=float, default=0.1, help="seconds between request submissions for aed")
    parser.add_argument("--storage", choices=["local", "r2", "both"], default="local")
    parser.add_argument("--config", default="ml/configs/data/default.yaml")
    parser.add_argument("--output", default="ml/data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.source == "weather":
        storage = make_storage(args.storage, output_dir)
        meta = build_weather_dataset(storage)
        print(f"exported {meta['n_records']} weather records via {args.storage}")
        return

    if args.source == "hkma":
        storage = make_storage(args.storage, output_dir)
        df = hkma.fetch_interbank_liquidity()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        storage.put("hk_hkma_interbank_liquidity.csv", buf.getvalue())
        print(f"Exported {len(df)} rows via {args.storage}")
        return

    if args.source == "aed":
        storage = make_storage(args.storage, output_dir)
        n = build_aed_dataset(storage, workers=args.workers, rate_sec=args.rate)
        print(f"Exported {n:,} rows via {args.storage}")
        return

    if args.source == "cci":
        storage = make_storage(args.storage, output_dir)
        df = centanet.tidy_cci()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        storage.put("hk_cci.csv", buf.getvalue())
        print(f"Exported {len(df)} rows via {args.storage}")
        return

    if args.source == "hospitals":
        storage = make_storage(args.storage, output_dir)
        counts = build_hospital_dataset(storage)
        for key, n in counts.items():
            print(f"  {key}: {n} rows")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(cfg, output_dir, output_dir / ".cache")


if __name__ == "__main__":
    main()
