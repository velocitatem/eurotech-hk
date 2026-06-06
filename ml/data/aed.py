import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_FILE = "https://www.ha.org.hk/opendata/aed/aedwtdata2-en.json"
_ARCHIVE = "https://app.data.gov.hk/v1/historical-archive/get-file"
_CACHE = Path("ml/data/processed/.cache/aed")
_BATCH = 50
_WORKERS = 8
_RATE = 0.1  # seconds between request submissions (~10 req/s)


def _parse_min(s: str) -> float:
    s = s.strip().lower()
    if re.match(r"0\s*(minute|hour)", s):
        return 0.0
    m = re.search(r"([\d.]+)\s*(minute|hour)", s)
    return (float(m.group(1)) * 60 if "hour" in m.group(2) else float(m.group(1))) if m else float("nan")


def _to_rows(ts: datetime, records: list) -> list[dict]:
    return [{
        "snapshot_time": ts.isoformat(),
        "hosp_name": r.get("hospName", ""),
        "t1_wait_min": _parse_min(r.get("t1wt", "0 minute")),
        "manage_t1": r.get("manageT1case", "N") == "Y",
        "t2_wait_min": _parse_min(r.get("t2wt", "0 minute")),
        "manage_t2": r.get("manageT2case", "N") == "Y",
        "t3_p50_min": _parse_min(r.get("t3p50", "0 minute")),
        "t3_p95_min": _parse_min(r.get("t3p95", "0 minute")),
        "t45_p50_min": _parse_min(r.get("t45p50", "0 hour")),
        "t45_p95_min": _parse_min(r.get("t45p95", "0 hour")),
    } for r in records]


def _cache_path(ts: datetime, cache_dir: Path) -> Path:
    return cache_dir / f"{ts.strftime('%Y%m%d-%H%M')}.json"


def _fetch(ts: datetime, cache_dir: Path) -> list | None:
    p = _cache_path(ts, cache_dir)
    if p.exists():
        return json.loads(p.read_text()).get("waitTime")
    cache_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(_ARCHIVE, params={"url": _FILE, "time": ts.strftime("%Y%m%d-%H%M")}, timeout=15)
    if r.status_code == 404:
        p.write_text("{}")
        return None
    r.raise_for_status()
    data = r.json()
    p.write_text(json.dumps(data))
    return data.get("waitTime")


def fetch_range(
    start: datetime,
    end: datetime,
    cache_dir: Path = _CACHE,
    workers: int = _WORKERS,
    rate_sec: float = _RATE,
) -> pd.DataFrame:
    ts_list = [start + timedelta(minutes=15 * i)
               for i in range(int((end - start).total_seconds() // 900) + 1)]
    cached = [ts for ts in ts_list if _cache_path(ts, cache_dir).exists()]
    uncached = [ts for ts in ts_list if not _cache_path(ts, cache_dir).exists()]

    eta_min = int(len(uncached) * rate_sec / 60)
    print(f"  {len(cached)} cached, {len(uncached)} to fetch (~{eta_min} min at {1/rate_sec:.0f} req/s)")

    rows: list[dict] = [
        row
        for ts in cached
        for records in [json.loads(_cache_path(ts, cache_dir).read_text()).get("waitTime") or []]
        for row in _to_rows(ts, records)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(uncached), _BATCH):
            batch = uncached[i:i + _BATCH]
            futs: dict = {}
            for ts in batch:
                futs[pool.submit(_fetch, ts, cache_dir)] = ts
                time.sleep(rate_sec)
            for fut in as_completed(futs):
                try:
                    records = fut.result()
                    if records:
                        rows.extend(_to_rows(futs[fut], records))
                except Exception as e:
                    print(f"\n  {futs[fut]}: {e}")
            print(f"  {min(i + _BATCH, len(uncached))}/{len(uncached)} fetched", end="\r")

    print()
    return pd.DataFrame(rows).sort_values(["snapshot_time", "hosp_name"]).reset_index(drop=True)
