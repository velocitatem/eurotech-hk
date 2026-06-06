"""IMMD daily passenger traffic → visitors_count (global state feature).

Source: https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic
Format: CSV with columns [Date, Control Point, Arrival / Departure,
        Hong Kong Residents, Mainland Visitors, Other Visitors, Total]

Aggregation strategy: sum arrivals across all control points per calendar day,
yielding one global record per day (visitors_count is broadcast in feature_spec).
Raw CSV is cached locally for _CACHE_TTL seconds to avoid hammering the server.
"""
from __future__ import annotations
import io, time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from dlib import get_logger
from .base import Loader

logger = get_logger("loader-immigration")

_CSV_URL = (
    "https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance"
    "/statistics_on_daily_passenger_traffic.csv"
)
_CACHE_TTL = 86_400  # seconds; refresh once per day

# IMMD control point → nearest HK 18-district zone_id
# Used as metadata only; the feature itself is broadcast globally.
CONTROL_POINT_ZONE: dict[str, str] = {
    "Airport":                        "lantau",
    "Express Rail Link West Kowloon": "yau_tsim_mong",
    "Hong Kong-Zhuhai-Macao Bridge":  "tuen_mun",
    "Lo Wu":                          "north",
    "Lok Ma Chau Loop":               "yuen_long",
    "Lok Ma Chau Spur Line":          "yuen_long",
    "Man Kam To":                     "north",
    "Sha Tau Kok":                    "north",
    "Shenzhen Bay":                   "yuen_long",
    "Kai Tak Cruise Terminal":        "kowloon_city",
    "Macau Ferry Terminal":           "yau_tsim_mong",
    "China Ferry Terminal":           "yau_tsim_mong",
    "Tuen Mun Ferry Terminal":        "tuen_mun",
    "Heliport":                       "wan_chai",
}


def _parse_int(s: Any) -> int:
    """Strip commas from IMMD numeric strings; return 0 on failure."""
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


class ImmigrationLoader(Loader):
    """Fetches and caches IMMD daily passenger traffic; emits visitors_count records."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit_s: float = 1.0,
    ) -> None:
        self._cache = (cache_dir or Path("ml/data/processed/.http_cache")) / "immd_daily.parquet"
        self._rate_s = rate_limit_s

    @property
    def loader_id(self) -> str:
        return "immd_daily_passenger"

    @property
    def schema_type(self) -> str:
        # matches NON_SDM_FEATURE_MAP["visitors_count"] in feature_spec.py
        return "visitors_count"

    # ------------------------------------------------------------------
    def _load_csv(self) -> pd.DataFrame:
        self._cache.parent.mkdir(parents=True, exist_ok=True)
        if self._cache.exists() and (time.time() - self._cache.stat().st_mtime) < _CACHE_TTL:
            logger.debug("immd cache hit")
            return pd.read_parquet(self._cache)

        logger.info(f"downloading IMMD passenger CSV from {_CSV_URL}")
        time.sleep(self._rate_s)
        resp = requests.get(_CSV_URL, timeout=60)
        resp.raise_for_status()

        raw = pd.read_csv(io.BytesIO(resp.content))
        raw.columns = [c.strip() for c in raw.columns]

        # normalise numeric columns that may contain commas
        for col in ("Hong Kong Residents", "Mainland Visitors", "Other Visitors", "Total"):
            if col in raw.columns:
                raw[col] = raw[col].map(_parse_int)

        # parse date — IMMD uses DD-MM-YYYY
        raw["date"] = pd.to_datetime(raw["Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
        raw = raw.dropna(subset=["date"])
        raw.to_parquet(self._cache)
        logger.info(f"cached {len(raw)} IMMD rows to {self._cache}")
        return raw

    # ------------------------------------------------------------------
    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        df = self._load_csv()
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        window = df[(df["date"] >= t0) & (df["date"] <= t1)]

        direction_col = next(
            (c for c in df.columns if "arrival" in c.lower() or "departure" in c.lower()),
            "Arrival / Departure",
        )
        arrivals = window[window[direction_col].str.strip() == "Arrival"]

        daily = arrivals.groupby("date", as_index=False).agg(
            visitors_count=("Total",               "sum"),
            hk_residents  =("Hong Kong Residents", "sum"),
            mainland       =("Mainland Visitors",  "sum"),
            other          =("Other Visitors",     "sum"),
        )

        for row in daily.itertuples(index=False):
            yield {
                "schema_type":      self.schema_type,
                "zone_id":          "global",
                "timestamp":        row.date.strftime("%Y-%m-%dT00:00:00Z"),
                "visitors_count":   float(row.visitors_count),
                "hk_residents":     float(row.hk_residents),
                "mainland_visitors": float(row.mainland),
                "other_visitors":   float(row.other),
            }
