import hashlib
import io
import re
from pathlib import Path

import pandas as pd
import requests

_CACHE = Path("ml/data/processed/.cache")
_URLS = {
    "patientday": "https://www.ha.org.hk/opendata/patientday-en.xlsx",
    "patientday_age_gender": "https://www.ha.org.hk/opendata/patientday-age-gender-en.xlsx",
    "ip_genspec": "https://www.ha.org.hk/opendata/ip-genspec-en.xlsx",
    "ahip_attnd": "https://www.ha.org.hk/opendata/ahip-attnd-en.xlsx",
    "disease_group": "https://www3.ha.org.hk/data/HAStatistics/downloadMajorReport/4?isPreview=False",
}


def _fetch(key: str, cache_dir: Path = _CACHE) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.xlsx"
    if p.exists():
        return p.read_bytes()
    r = requests.get(_URLS[key], timeout=30)
    r.raise_for_status()
    p.write_bytes(r.content)
    return r.content


def _read(raw: bytes, header: int) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw), header=header, sheet_name=0).dropna(how="all").reset_index(drop=True)


def _snake(s) -> str:
    s = str(s).strip().lower().replace("%", "pct")
    s = re.sub(r"[()./&]", "", s)
    return re.sub(r"_+", "_", re.sub(r"[\s\-]+", "_", s)).strip("_")


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_snake(c) for c in df.columns]
    return df


def _ffill(df: pd.DataFrame, *cols) -> None:
    df[list(cols)] = df[list(cols)].ffill()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def tidy_patientday(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = _read(_fetch("patientday", cache_dir), 2)
    fy, cluster, hospital = df.columns[:3]
    _ffill(df, fy, cluster)
    df = df[~df[hospital].astype(str).str.startswith("Overall")]
    df = _rename(df)
    for c in df.columns[3:]:
        df[c] = _to_num(df[c])
    return df.reset_index(drop=True)


def tidy_patientday_age_gender(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = _read(_fetch("patientday_age_gender", cache_dir), 2)
    fy, age, gender = df.columns[:3]
    _ffill(df, fy, age)
    df = df[df[gender] != "Overall"]
    df = _rename(df)
    for c in df.columns[3:]:
        df[c] = _to_num(df[c])
    return df.reset_index(drop=True)


def tidy_ip_genspec(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = _read(_fetch("ip_genspec", cache_dir), 2)
    fy, cluster = df.columns[:2]
    _ffill(df, fy)
    df = df[df[cluster] != "Overall"]
    df = _rename(df)
    for c in df.columns[2:]:
        df[c] = _to_num(df[c])
    return df.reset_index(drop=True)


def tidy_ahip_attnd(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = _read(_fetch("ahip_attnd", cache_dir), 3)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    fy, cluster, hospital = df.columns[:3]
    _ffill(df, fy, cluster)
    df = df[~df[hospital].astype(str).str.startswith("Overall")]
    df = _rename(df)
    id_cols, dept_cols = list(df.columns[:3]), list(df.columns[3:])
    df = df.melt(id_vars=id_cols, value_vars=dept_cols, var_name="department", value_name="attendances")
    df["attendances"] = _to_num(df["attendances"])
    return df.reset_index(drop=True)


def tidy_disease_group(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = _read(_fetch("disease_group", cache_dir), 2)
    id_cols, year_cols = list(df.columns[:2]), list(df.columns[2:])
    df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="discharges_and_deaths")
    df = df.rename(columns={id_cols[0]: "disease_group", id_cols[1]: "icd_code"})
    df["discharges_and_deaths"] = _to_num(df["discharges_and_deaths"])
    return df.sort_values(["year", "disease_group"]).reset_index(drop=True)


def aggregate_patient_days(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = tidy_patientday(cache_dir)
    fy_col = df.columns[0]
    return df.groupby(fy_col)[list(df.columns[3:])].sum().reset_index()


def aggregate_ahip(cache_dir: Path = _CACHE) -> pd.DataFrame:
    df = tidy_ahip_attnd(cache_dir)
    fy_col = df.columns[0]
    return df.groupby([fy_col, "department"])["attendances"].sum().reset_index()
