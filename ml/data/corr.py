"""Compute Xi (Chatterjee 2021) and Spearman correlations between weather and health datasets.

Weather (hourly) is resampled to match each target's granularity:
  - weekly   → CCI
  - annual-FY → hospital aggregates  (April–March fiscal year)
  - annual-cal → disease group totals (calendar year)

Usage:
  python -m ml.data.corr
  python -m ml.data.corr --processed ml/data/processed --output ml/data/processed/corr_results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROCESSED = Path("ml/data/processed")
WEATHER_VARS = ["temperature_2m", "rain", "wind_speed_10m", "uv_index"]
MIN_N = 5


def xi(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee (2021) xi: 0 = independent, 1 = functional dependence."""
    n = len(x)
    if n < MIN_N:
        return np.nan
    r = np.argsort(np.argsort(y[np.argsort(x, kind="stable")], kind="stable"), kind="stable") + 1
    return 1.0 - 3.0 * float(np.abs(np.diff(r)).sum()) / (n**2 - 1)


def _corr_row(granularity: str, wvar: str, tvar: str, x: np.ndarray, y: np.ndarray) -> dict:
    sp, p = spearmanr(x, y)
    return {
        "granularity": granularity, "weather_var": wvar, "target_var": tvar,
        "n": len(x), "xi": round(xi(x, y), 4),
        "spearman": round(float(sp), 4), "spearman_p": round(float(p), 4),
    }


def _pairs(merged: pd.DataFrame, w_cols: list[str], t_cols: list[str], tag: str) -> list[dict]:
    return [
        _corr_row(tag, wc, tc, *map(lambda s: s.values, [sub[wc], sub[tc]]))
        for wc in w_cols for tc in t_cols
        if len(sub := merged[[wc, tc]].dropna()) >= MIN_N
    ]


def _load_weather(processed: Path) -> pd.DataFrame:
    df = pd.read_csv(processed / "hk_temperature.csv", parse_dates=["time"])
    return df.set_index("time").sort_index()


def _resample(w: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {c: ("sum" if c == "rain" else "mean") for c in WEATHER_VARS if c in w.columns}
    return w.resample(freq).agg(agg).dropna(how="all")


def _valid_fy(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["financial_year"].str.match(r"^\d{4}-\d{2}$", na=False)].copy()


# ── weekly: weather × CCI ────────────────────────────────────────────────────

def _corr_cci(processed: Path, w: pd.DataFrame) -> list[dict]:
    cci = pd.read_csv(processed / "hk_cci.csv", parse_dates=["week_start"])
    ww = _resample(w, "W-SUN")  # Mon–Sun periods, labeled with Sunday
    ww.index = (ww.index - pd.Timedelta(6, "D")).normalize()  # shift label → Monday
    merged = ww.join(cci.set_index("week_start")[["cci"]], how="inner")
    w_cols = [c for c in WEATHER_VARS if c in merged.columns]
    return _pairs(merged, w_cols, ["cci"], "weekly")


# ── annual fiscal year: weather × hospital aggregates ────────────────────────

def _corr_annual_fy(processed: Path, w: pd.DataFrame) -> list[dict]:
    wfy = _resample(w, "YE-MAR")  # periods ending March 31 (HK FY boundary)
    wfy.index = wfy.index.year  # FY end year: "2024-25" → 2025

    datasets = {
        "patient_days": (
            pd.read_csv(processed / "hk_agg_patient_days.csv"),
            ["inpatient_patient_days", "day_inpatient_discharges_and_deaths", "patient_days"],
        ),
        "ahip": (
            pd.read_csv(processed / "hk_agg_ahip.csv"),
            ["attendances"],
        ),
        "ip_genspec": (
            pd.read_csv(processed / "hk_ip_genspec.csv"),
            ["inpatient_discharges_and_deaths", "inpatient_patient_days",
             "inpatient_bed_occupancy_rate_pct", "inpatient_average_length_of_stay_days"],
        ),
    }
    rows = []
    for name, (df, targets) in datasets.items():
        df = _valid_fy(df)
        df["_yr"] = df["financial_year"].map(lambda fy: int(fy.split("-")[0]) + 1)
        agg = df.groupby("_yr")[[c for c in targets if c in df.columns]].sum()
        merged = wfy.join(agg, how="inner").dropna()
        w_cols = [c for c in WEATHER_VARS if c in merged.columns]
        t_cols = [c for c in targets if c in merged.columns]
        rows += _pairs(merged, w_cols, t_cols, f"annual_fy:{name}")
    return rows


# ── annual calendar: weather × disease groups ─────────────────────────────────

def _corr_disease(processed: Path, w: pd.DataFrame) -> list[dict]:
    wyr = _resample(w, "YE")
    wyr.index = wyr.index.year

    df = pd.read_csv(processed / "hk_disease_group.csv")
    df = df[df["year"].astype(str).str.match(r"^\d{4}$")].copy()
    df["year"] = df["year"].astype(int)

    by_year = df.groupby("year")["discharges_and_deaths"].sum().rename("total_discharges")
    merged = wyr.join(by_year, how="inner").dropna()
    w_cols = [c for c in WEATHER_VARS if c in merged.columns]
    return _pairs(merged, w_cols, ["total_discharges"], "annual_cal:disease")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", default=PROCESSED, type=Path)
    parser.add_argument("--output", default=None, type=Path,
                        help="Output CSV path (default: <processed>/corr_results.csv)")
    args = parser.parse_args()
    out_path = args.output or args.processed / "corr_results.csv"

    w = _load_weather(args.processed)
    avail = [c for c in WEATHER_VARS if c in w.columns]
    print(f"Weather: {w.index[0].date()} → {w.index[-1].date()}  vars={avail}")

    analyses = [
        ("weekly×cci",        _corr_cci),
        ("annual_fy×hospital", _corr_annual_fy),
        ("annual_cal×disease", _corr_disease),
    ]
    all_rows: list[dict] = []
    for label, fn in analyses:
        try:
            rows = fn(args.processed, w)
            all_rows += rows
            if rows:
                ns = [r["n"] for r in rows]
                print(f"  {label}: {len(rows)} pairs, n={min(ns)}–{max(ns)}")
            else:
                print(f"  {label}: no pairs with n≥{MIN_N} (insufficient temporal overlap)")
        except Exception as exc:
            print(f"  {label}: skipped — {exc}", file=sys.stderr)

    if not all_rows:
        print("No correlations computed.")
        return

    out = (
        pd.DataFrame(all_rows)
        .sort_values(["granularity", "xi"], ascending=[True, False])
        .reset_index(drop=True)
    )
    out.to_csv(out_path, index=False)
    print(f"\nResults ({len(out)} rows) → {out_path}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
