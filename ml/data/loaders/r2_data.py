"""Loaders for datasets stored in Cloudflare R2.

Reads are cached locally as parquet (TTL 24 h) to avoid repeated downloads.
Requires env vars: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME.
"""
from __future__ import annotations
import io, os, time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from dlib import get_logger
from .base import Loader

logger = get_logger("loader-r2")

_CACHE_DIR = Path("ml/data/processed/.r2_cache")
_TTL = 86_400  # 1 day


def _r2_csv(key: str, sep: str = ",", cache_dir: Path = _CACHE_DIR) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key.replace('/', '_')}.parquet"
    if cached.exists() and (time.time() - cached.stat().st_mtime) < _TTL:
        return pd.read_parquet(cached)

    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    raw = s3.get_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=key)["Body"].read().decode()
    df = pd.read_csv(io.StringIO(raw), sep=sep)
    df.to_parquet(cached)
    logger.info(f"r2 cached {key} → {cached} ({len(df)} rows)")
    return df


def _fy_to_dates(fy: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """'2021-22' → (2021-04-01, 2022-03-31) — HK financial year starts April."""
    y = int(fy.split("-")[0])
    return pd.Timestamp(y, 4, 1), pd.Timestamp(y + 1, 4, 1) - pd.Timedelta(days=1)


class HospitalOccupancyLoader(Loader):
    """HA inpatient bed occupancy by cluster (annual FY) → hospital_occupancy state feature.

    System-wide average across all clusters is broadcast globally; each day of the
    financial year receives the annual value (forward-filled by _build_panel).
    """

    @property
    def loader_id(self) -> str:
        return "ha_hospital_occupancy"

    @property
    def schema_type(self) -> str:
        return "hospital_occupancy"

    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        df = _r2_csv("hk_ip_genspec.csv")
        df = df[df["financial_year"].str.match(r"\d{4}-\d{2}$", na=False)].copy()

        yearly = (
            df.groupby("financial_year")["inpatient_bed_occupancy_rate_pct"]
            .mean()
            .div(100.0)
        )
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)

        for fy, occ in yearly.items():
            fy_start, fy_end = _fy_to_dates(fy)
            day = max(fy_start, t0.normalize())
            last = min(fy_end, t1.normalize())
            while day <= last:
                yield {
                    "schema_type":        self.schema_type,
                    "zone_id":            "global",
                    "timestamp":          day.strftime("%Y-%m-%dT00:00:00Z"),
                    "hospital_occupancy": float(occ),
                }
                day += pd.Timedelta(days=1)


class HSILoader(Loader):
    """Hang Seng Index daily close → hsi_close state feature (global broadcast)."""

    @property
    def loader_id(self) -> str:
        return "hsi_close"

    @property
    def schema_type(self) -> str:
        return "hsi_close"

    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        df = _r2_csv("stock_index.csv", sep=";")
        df["date"] = pd.to_datetime(df["date"])
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        for row in df[(df["date"] >= t0) & (df["date"] <= t1)].itertuples(index=False):
            yield {
                "schema_type": self.schema_type,
                "zone_id":     "global",
                "timestamp":   row.date.strftime("%Y-%m-%dT00:00:00Z"),
                "hsi_close":   float(row.close),
            }


class PetrolPriceLoader(Loader):
    """HK retail petrol price (HKD/litre) as oil_price condition proxy (global broadcast).

    Not WTI USD/barrel, but highly correlated with global energy prices; used as-is
    since no HKD→USD/bbl conversion factor is available without an FX loader.
    """

    @property
    def loader_id(self) -> str:
        return "hk_petrol_price"

    @property
    def schema_type(self) -> str:
        return "oil_price"

    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]:
        df = _r2_csv("hk_petrol_daily_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        for row in df[(df["date"] >= t0) & (df["date"] <= t1)].itertuples(index=False):
            yield {
                "schema_type":   self.schema_type,
                "zone_id":       "global",
                "timestamp":     row.date.strftime("%Y-%m-%dT00:00:00Z"),
                "oil_price_usd": float(row.petrol_hkd_per_litre),
            }
