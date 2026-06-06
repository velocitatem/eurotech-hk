"""Score current and forecast HK weather conditions against the disaster risk model.

Usage:
  python -m ml.disaster_risk              # use cached live data
  python -m ml.disaster_risk --live       # re-fetch from Open-Meteo
"""
import argparse
import json
import numpy as np
from pathlib import Path

from ml.data.disaster_analysis import extract_features, score_features, combined_risk
from ml.data.openmeteo import fetch_hk_live

MODEL_PATH   = Path("ml/data/processed/disasters/disaster_model.json")
HK_LIVE_PATH = Path("ml/data/processed/disasters/hk_live.json")

_LEVELS = [(0.72, "CRITICAL"), (0.58, "HIGH"), (0.44, "MODERATE"), (0.30, "ELEVATED"), (0.0, "LOW")]


def risk_level(score: float) -> str:
    return next(label for thresh, label in _LEVELS if score >= thresh)


def _window_features(hourly: dict, start: int, n: int) -> dict:
    return {k: v[start:start + n] for k, v in hourly.items() if k != "time"}


def _current_features(live: dict) -> dict:
    """Current snapshot merged with next 24h for rolling stats."""
    cur = live["current"]
    h = live["hourly"]
    fields = ["precipitation", "wind_gusts_10m", "wind_speed_10m", "surface_pressure", "relative_humidity_2m"]
    window = {f: [cur.get(f) or 0] + list(h[f][:23]) for f in fields}
    return extract_features(window)


def forecast_profile(live: dict, model: dict, window_h: int = 24) -> list[dict]:
    """Sliding-window risk scores across the 7-day hourly forecast."""
    h = live["hourly"]
    n = len(h["time"])
    step = window_h // 2
    results = []
    for i in range(0, n - window_h + 1, step):
        w = _window_features(h, i, window_h)
        s = score_features(extract_features(w), model)
        r = combined_risk(s)
        results.append({"window_start": h["time"][i], "risk_score": round(r, 3), "risk_level": risk_level(r)})
    return results


def assess(live: dict | None = None, refetch: bool = False) -> dict:
    model = json.loads(MODEL_PATH.read_text())

    if live is None:
        if refetch or not HK_LIVE_PATH.exists():
            live = fetch_hk_live()
            HK_LIVE_PATH.write_text(json.dumps(live, indent=2))
        else:
            live = json.loads(HK_LIVE_PATH.read_text())

    cur = live["current"]
    cur_feat = _current_features(live)
    scores = score_features(cur_feat, model)
    risk = combined_risk(scores)
    level = risk_level(risk)
    profile = forecast_profile(live, model)

    peak = max(profile, key=lambda x: x["risk_score"])

    result = {
        "as_of": cur["time"],
        "risk_score": round(risk, 3),
        "risk_level": level,
        "feature_scores": {k: round(v, 3) for k, v in scores.items()},
        "current_features": {k: round(v, 2) for k, v in cur_feat.items()},
        "forecast_peak": peak,
        "forecast_profile": profile,
    }

    _print_report(cur, cur_feat, scores, model, risk, level, profile, peak)
    return result


def _print_report(cur, cur_feat, scores, model, risk, level, profile, peak):
    sep = "=" * 48
    print(f"\n{sep}")
    print(f"  Hong Kong Natural Disaster Risk Assessment")
    print(f"  As of: {cur['time']}")
    print(sep)

    print("\nCurrent conditions:")
    print(f"  Precipitation       {cur.get('precipitation', 0):.1f} mm/h")
    print(f"  Wind gust           {cur.get('wind_gusts_10m', 0):.1f} km/h")
    print(f"  Surface pressure    {cur.get('surface_pressure', 0):.1f} hPa")
    print(f"  Relative humidity   {cur.get('relative_humidity_2m', 0):.0f}%")
    print(f"  CAPE                {cur.get('cape', 'N/A')} J/kg")
    print(f"  Precip probability  {cur.get('precipitation_probability', 'N/A')}%")

    print("\nFeature scores vs disaster distribution (1.0 = exceeds disaster mean):")
    for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
        p = model["features"][feat]
        dir_sym = "↑" if p["direction"] == 1 else "↓"
        bar = "█" * int(score * 15)
        print(f"  {feat:22s} {dir_sym} {score:.3f}  {bar:15s}  "
              f"(val={cur_feat.get(feat, 0):.1f}, disaster_μ={p['mu']:.1f})")

    print(f"\n{sep}")
    print(f"  OVERALL RISK SCORE:  {risk:.3f}  →  {level}")
    print(f"{sep}\n")

    print("7-day forecast risk profile (24h windows):")
    for p in profile:
        bar = "█" * int(p["risk_score"] * 20)
        marker = " ◄ PEAK" if p["window_start"] == peak["window_start"] else ""
        print(f"  {p['window_start']}  {p['risk_score']:.3f}  {p['risk_level']:8s}  {bar}{marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Re-fetch live data from Open-Meteo")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    args = parser.parse_args()

    result = assess(refetch=args.live)
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
