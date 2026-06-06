"""Open-Meteo weather loaders for observed and forecast condition features.

WeatherObservedLoader  → WeatherObserved (state): temperature_observed, humidity_observed
WeatherForecastLoader  → WeatherForecast (condition): temperature_forecast, wind_forecast,
                         precipitation_forecast, weather_alert_level

Both use the local Open-Meteo cache (openmeteo.py), fetching historical archive as a
training-time proxy. Cache TTL is 30 days (immutable historical data).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from dlib import get_logger
from .base import Loader

try:
    from ml.data.openmeteo import fetch_historical, _CACHE
except ImportError:
    from openmeteo import fetch_historical, _CACHE  # type: ignore[no-redef]

logger = get_logger("loader-weather")

_OBS_VARS  = ["temperature_2m", "relative_humidity_2m"]
_FCST_VARS = ["temperature_2m", "wind_speed_10m", "rain"]

# weatherType string → weather_alert_level (0-3)
_WEATHER_ALERT = {"Clear": 0, "Partly cloudy": 1, "Rainy": 2, "Storm": 3}


def _hourly_df(raw: dict, variables: list[str]) -> pd.DataFrame:
    h   = raw["hourly"]
    df  = pd.DataFrame({k: h[k] for k in ["time"] + variables if k in h})
    df["time"] = pd.to_datetime(df["time"])
    return df


def _window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    return df[(df["time"] >= t0) & (df["time"] <= t1)]


class WeatherObservedLoader(Loader):
    """Open-Meteo historical archive → WeatherObserved state features (global broadcast)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache = cache_dir or _CACHE

    @property
    def loader_id(self) -> str:
        return "openmeteo_observed"

    @property
    def schema_type(self) -> str:
        return "WeatherObserved"

    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        t0, t1 = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        raw = fetch_historical(start=t0, end=t1, variables=_OBS_VARS, cache_dir=self._cache)
        df  = _window(_hourly_df(raw, _OBS_VARS), start, end)
        for row in df.itertuples(index=False):
            yield {
                "schema_type":     self.schema_type,
                "zone_id":         "global",
                "timestamp":       row.time.isoformat(),
                "temperature":     float(row.temperature_2m)      if pd.notna(row.temperature_2m)      else 0.0,
                "relativeHumidity": float(row.relative_humidity_2m) if pd.notna(row.relative_humidity_2m) else 0.0,
            }


class WeatherForecastLoader(Loader):
    """Open-Meteo historical archive as training-time forecast proxy → WeatherForecast conditions.

    precipitation_forecast is derived from hourly rain (mm): min(1, rain/10).
    weather_alert_level: 0=dry (<0.1 mm), 1=light (<2), 2=moderate (<10), 3=heavy (≥10).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache = cache_dir or _CACHE

    @property
    def loader_id(self) -> str:
        return "openmeteo_forecast"

    @property
    def schema_type(self) -> str:
        return "WeatherForecast"

    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        t0, t1 = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        raw = fetch_historical(start=t0, end=t1, variables=_FCST_VARS, cache_dir=self._cache)
        df  = _window(_hourly_df(raw, _FCST_VARS), start, end)
        for row in df.itertuples(index=False):
            rain = float(row.rain) if pd.notna(row.rain) else 0.0
            alert = 0 if rain < 0.1 else 1 if rain < 2 else 2 if rain < 10 else 3
            yield {
                "schema_type":              self.schema_type,
                "zone_id":                  "global",
                "timestamp":                row.time.isoformat(),
                "temperature":              float(row.temperature_2m)  if pd.notna(row.temperature_2m)  else 0.0,
                "windSpeed":                float(row.wind_speed_10m)  if pd.notna(row.wind_speed_10m)  else 0.0,
                "precipitationProbability": min(1.0, rain / 10.0),
                "weatherType":              float(alert),
            }
