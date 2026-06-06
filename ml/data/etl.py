import argparse
import csv
import io
import json
from pathlib import Path

import torch
import yaml

from datetime import datetime, timedelta

try:
    from ml.data.openmeteo import fetch_historical, _CACHE as _OM_CACHE
    from ml.data.storage import Storage, make_storage
    from ml.data import ha_hospitals, centanet, aed, hkma
except ImportError:
    from openmeteo import fetch_historical, _CACHE as _OM_CACHE  # type: ignore[no-redef]
    from storage import Storage, make_storage  # type: ignore[no-redef]
    import ha_hospitals, centanet, aed, hkma  # type: ignore[no-redef]


def build_dataset(
    train_samples: int, input_dim: int, num_classes: int, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(train_samples, input_dim, generator=generator)
    labels = torch.randint(0, num_classes, (train_samples,), generator=generator)
    return {"features": features, "labels": labels}


def build_weather_dataset(storage: Storage, cache_dir: Path | None = None) -> dict:
    raw = fetch_historical(cache_dir=cache_dir or _OM_CACHE)
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
        print(f"Exported {meta['n_records']} records via {args.storage}")
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
    dataset = build_dataset(
        train_samples=int(cfg["train_samples"]),
        input_dim=int(cfg["input_dim"]),
        num_classes=int(cfg["num_classes"]),
        seed=int(cfg["seed"]),
    )
    dataset_path = output_dir / "dataset.pt"
    torch.save(dataset, dataset_path)

    metadata = {
        "dataset_name": cfg["dataset_name"],
        "train_samples": int(cfg["train_samples"]),
        "input_dim": int(cfg["input_dim"]),
        "num_classes": int(cfg["num_classes"]),
        "seed": int(cfg["seed"]),
        "dataset_path": str(dataset_path),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
