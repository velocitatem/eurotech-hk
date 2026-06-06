import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

HK_LAT, HK_LON = 22.302711, 114.177216
_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_CACHE = Path("ml/data/processed/.cache")

# Variables covering all major natural-disaster precursor signals:
# precipitation intensity, wind (speed/gust/direction), pressure drop (cyclone),
# humidity (landslide saturation), cloud cover, convective energy (CAPE), weather code.
_DISASTER_VARS = [
    "precipitation", "rain", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "surface_pressure", "temperature_2m",
    "relative_humidity_2m", "cloud_cover", "weather_code",
]
_FORECAST_VARS = _DISASTER_VARS + ["precipitation_probability", "cape"]


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


def fetch_historical(
    start: date | None = None,
    end: date | None = None,
    variables: list[str] | None = None,
    lat: float = HK_LAT,
    lon: float = HK_LON,
    cache_dir: Path = _CACHE,
) -> dict[str, Any]:
    end = end or date.today()
    start = start or end - timedelta(days=730)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(variables or _DISASTER_VARS),
    }
    return _get(_ARCHIVE, params, cache_dir, ttl=2_592_000)  # immutable; cache 30d


def fetch_disaster_event(event: dict, pad_days: int = 3, cache_dir: Path = _CACHE) -> dict[str, Any]:
    """Fetch hourly weather for a disaster event with ±pad_days context window."""
    start = date.fromisoformat(event["date_range"]["start"]) - timedelta(days=pad_days)
    end = date.fromisoformat(event["date_range"]["end"]) + timedelta(days=pad_days)
    coord = event["coordinates"]
    return {
        "event_id": event["id"],
        "weather": fetch_historical(start, end, lat=coord["lat"], lon=coord["lon"], cache_dir=cache_dir),
    }


def fetch_all_disasters(
    json_path: Path = Path("ml/data/processed/global_natural_disasters.json"),
    pad_days: int = 3,
    cache_dir: Path = _CACHE,
) -> list[dict[str, Any]]:
    """Fetch weather for every event in the disasters JSON; returns list keyed by event_id."""
    disasters = json.loads(json_path.read_text())["disasters"]
    return [fetch_disaster_event(ev, pad_days, cache_dir) for ev in disasters]


def fetch_hk_live(variables: list[str] | None = None, cache_dir: Path = _CACHE, ttl: int = 3600) -> dict[str, Any]:
    """Current conditions + 7-day hourly forecast for Hong Kong with full disaster-signal variables."""
    vars_ = variables or _FORECAST_VARS
    params = {
        "latitude": HK_LAT, "longitude": HK_LON,
        "current": ",".join(vars_),
        "hourly": ",".join(vars_),
        "forecast_days": 7,
    }
    return _get(_FORECAST, params, cache_dir, ttl=ttl)


# back-compat alias
def fetch_forecast(variables: list[str] | None = None, cache_dir: Path = _CACHE, ttl: int = 3600) -> dict[str, Any]:
    return fetch_hk_live(variables, cache_dir, ttl)
