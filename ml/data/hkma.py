import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

_BASE = "https://api.hkma.gov.hk/public/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity"
_CACHE = Path("ml/data/processed/.cache")
_PAGE = 100


def fetch_interbank_liquidity(
    start: date | None = None,
    end: date | None = None,
    cache_dir: Path = _CACHE,
) -> pd.DataFrame:
    end = end or date.today()
    start = start or end - timedelta(days=730)

    p = cache_dir / f"hkma_interbank_{start}_{end}.json"
    if p.exists():
        return pd.DataFrame(json.loads(p.read_text())).sort_values("end_of_date").reset_index(drop=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    records, offset = [], 0

    while True:
        r = requests.get(
            _BASE,
            params={"startdate": start.isoformat(), "enddate": end.isoformat(), "limit": _PAGE, "offset": offset},
            timeout=15,
        )
        r.raise_for_status()
        page = r.json()["result"]["records"]
        if not page:
            break

        records.extend(rec for rec in page if date.fromisoformat(rec["end_of_date"]) >= start)

        if date.fromisoformat(page[-1]["end_of_date"]) < start or len(page) < _PAGE:
            break

        offset += _PAGE
        time.sleep(0.3)

    p.write_text(json.dumps(records))
    return pd.DataFrame(records).sort_values("end_of_date").reset_index(drop=True)
