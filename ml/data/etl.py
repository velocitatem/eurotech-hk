import argparse
import csv
import io
import json
from pathlib import Path

import torch
import yaml

try:
    from ml.data.openmeteo import fetch_historical, _CACHE as _OM_CACHE
    from ml.data.storage import Storage, make_storage
except ImportError:
    from openmeteo import fetch_historical, _CACHE as _OM_CACHE  # type: ignore[no-redef]
    from storage import Storage, make_storage  # type: ignore[no-redef]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "weather"], default="synthetic")
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
