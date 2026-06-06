import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

HK_LAT = 22.302711
HK_LON = 114.177216
_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_CACHE = Path("ml/data/processed/.cache")


def _key(url: str, params: dict) -> str:
    return hashlib.sha256(json.dumps({"url": url, **params}, sort_keys=True).encode()).hexdigest()[:16]


def _get(url: str, params: dict, cache_dir: Path, ttl: int) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{_key(url, params)}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        return json.loads(p.read_text())
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    p.write_text(json.dumps(data))
    return data


_DEFAULT_VARS = ["temperature_2m", "rain", "wind_speed_10m", "uv_index"]


def fetch_historical(
    start: date | None = None,
    end: date | None = None,
    variables: list[str] | None = None,
    cache_dir: Path = _CACHE,
) -> dict[str, Any]:
    end = end or date.today()
    start = start or end - timedelta(days=730)
    params = {
        "latitude": HK_LAT,
        "longitude": HK_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables or _DEFAULT_VARS),
    }
    return _get(_ARCHIVE, params, cache_dir, ttl=2_592_000)  # 30d; historical data is immutable


def fetch_forecast(
    variables: list[str] | None = None,
    cache_dir: Path = _CACHE,
    ttl: int = 3600,
) -> dict[str, Any]:
    params = {
        "latitude": HK_LAT,
        "longitude": HK_LON,
        "hourly": ",".join(variables or _DEFAULT_VARS),
    }
    return _get(_FORECAST, params, cache_dir, ttl=ttl)
