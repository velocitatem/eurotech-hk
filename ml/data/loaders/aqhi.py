"""EPD hourly AQHI loader → air_quality_index state feature.

Source: https://www.aqhi.gov.hk/epd/ddata/html/history/{year}/{YYYYMM}_Eng.csv
Coverage: Dec 2013 – present (6-month lag on most recent months).
18 monitoring stations averaged to a single global value per hour.
Monthly CSV files are cached locally for 30 days (historical data is immutable).
"""
from __future__ import annotations
import io, time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from dlib import get_logger
from .base import Loader

logger = get_logger("loader-aqhi")

_BASE = "https://www.aqhi.gov.hk/epd/ddata/html/history/{year}/{ym}_Eng.csv"
_CACHE_DIR = Path("ml/data/processed/.aqhi_cache")
_TTL = 2_592_000  # 30 days — historical files never change
_HEADER_SKIP = 7  # rows before the actual column header


def _fetch_month(year: int, month: int) -> pd.DataFrame | None:
    ym = f"{year}{month:02d}"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / f"{ym}.parquet"

    if cached.exists() and (time.time() - cached.stat().st_mtime) < _TTL:
        return pd.read_parquet(cached)

    url = _BASE.format(year=year, ym=ym)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"aqhi fetch failed {ym}: {e}")
        return None

    raw = resp.text
    lines = raw.split("\n")
    # find header row (starts with "Date")
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Date")), _HEADER_SKIP)
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]

    # forward-fill Date (only first hour of each day has it)
    df["Date"] = df["Date"].ffill()
    df = df.dropna(subset=["Date", "Hour"])
    df = df[pd.to_numeric(df["Hour"], errors="coerce").notna()]  # drop "Daily Max" etc.
    df["Hour"] = df["Hour"].astype(int)
    df["timestamp"] = pd.to_datetime(df["Date"].str.strip()) + pd.to_timedelta(df["Hour"] - 1, unit="h")

    station_cols = [c for c in df.columns if c not in ("Date", "Hour", "timestamp")]
    df[station_cols] = df[station_cols].apply(pd.to_numeric, errors="coerce")
    df["aqhi_mean"] = df[station_cols].mean(axis=1)

    out = df[["timestamp", "aqhi_mean"]].dropna()
    out.to_parquet(cached)
    logger.info(f"aqhi cached {ym} ({len(out)} rows)")
    return out


class AQHILoader(Loader):
    """EPD hourly AQHI (18-station mean) → air_quality_index state feature (global broadcast)."""

    @property
    def loader_id(self) -> str:
        return "epd_aqhi"

    @property
    def schema_type(self) -> str:
        return "air_quality_index"

    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        months = pd.period_range(t0.to_period("M"), t1.to_period("M"), freq="M")

        for period in months:
            df = _fetch_month(period.year, period.month)
            if df is None:
                continue
            window = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
            for row in window.itertuples(index=False):
                yield {
                    "schema_type":     self.schema_type,
                    "zone_id":         "global",
                    "timestamp":       row.timestamp.isoformat(),
                    "air_quality_index": float(row.aqhi_mean),
                }
