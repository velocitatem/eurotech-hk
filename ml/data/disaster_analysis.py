"""Fit per-feature normal distributions over historical disaster events and save a scoring model.

Usage:
  python -m ml.data.disaster_analysis
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp

DISASTERS_JSON = Path("ml/data/processed/global_natural_disasters.json")
WEATHER_JSON   = Path("ml/data/processed/disasters/historical_weather.json")
MODEL_OUT      = Path("ml/data/processed/disasters/disaster_model.json")

# direction: +1 = higher value is more dangerous, -1 = lower value is more dangerous
FEATURE_DIR = {
    "max_precip_1h":      +1,
    "max_precip_24h":     +1,
    "max_wind_gust":      +1,
    "max_wind_speed":     +1,
    "min_pressure":       -1,
    "pressure_drop_24h":  +1,
    "max_humidity":       +1,
    "hrs_humidity_90":    +1,
}


def _arr(vals: list) -> np.ndarray:
    return np.array([x if isinstance(x, (int, float)) else np.nan for x in vals], dtype=float)


def extract_features(hourly: dict) -> dict:
    """Aggregate peak/extreme statistics from an hourly data dict."""
    p  = _arr(hourly.get("precipitation", []))
    g  = _arr(hourly.get("wind_gusts_10m", []))
    w  = _arr(hourly.get("wind_speed_10m", []))
    pr = _arr(hourly.get("surface_pressure", []))
    h  = _arr(hourly.get("relative_humidity_2m", []))

    p_clean = np.nan_to_num(p)
    p24 = float(np.max(np.convolve(p_clean, np.ones(24), "valid"))) if len(p_clean) >= 24 else float(p_clean.sum())

    pr_v = pr[~np.isnan(pr)]
    pr_drop = max((float(pr_v[i] - pr_v[i + 24]) for i in range(len(pr_v) - 24)), default=0.0)

    return {
        "max_precip_1h":     float(np.nanmax(p)) if len(p) else 0.0,
        "max_precip_24h":    p24,
        "max_wind_gust":     float(np.nanmax(g)) if len(g) else 0.0,
        "max_wind_speed":    float(np.nanmax(w)) if len(w) else 0.0,
        "min_pressure":      float(np.nanmin(pr)) if len(pr) else 1013.0,
        "pressure_drop_24h": max(pr_drop, 0.0),
        "max_humidity":      float(np.nanmax(h)) if len(h) else 0.0,
        "hrs_humidity_90":   float(np.sum(h > 90)),
    }


def score_features(features: dict, model: dict) -> dict[str, float]:
    """Map each feature value to [0,1]: 1 = matches or exceeds disaster levels."""
    return {
        feat: float(sp.norm.cdf(z) if p["direction"] == 1 else sp.norm.sf(z))
        for feat, p in model["features"].items()
        if (z := (features.get(feat, p["mu"]) - p["mu"]) / max(p["sigma"], 1e-9)) is not None
    }


def combined_risk(scores: dict[str, float]) -> float:
    return float(np.mean(list(scores.values()))) if scores else 0.0


def _fit_model(event_feats: list[dict]) -> dict:
    df = pd.DataFrame(event_feats)
    features = {}
    for feat, direction in FEATURE_DIR.items():
        vals = df[feat].dropna().values
        mu, sigma = sp.norm.fit(vals)
        features[feat] = {
            "direction": direction, "mu": round(mu, 4), "sigma": round(sigma, 4),
            "min": round(vals.min(), 4), "max": round(vals.max(), 4),
            "p25": round(float(np.percentile(vals, 25)), 4),
            "p75": round(float(np.percentile(vals, 75)), 4),
        }
    corr = df[list(FEATURE_DIR)].corr(method="spearman").round(3).to_dict()
    return {"n_events": len(event_feats), "features": features, "spearman_corr": corr}


def main() -> dict:
    disasters = {d["id"]: d for d in json.loads(DISASTERS_JSON.read_text())["disasters"]}
    weather   = json.loads(WEATHER_JSON.read_text())

    event_feats = []
    for ev in weather:
        f = extract_features(ev["weather"]["hourly"])
        meta = disasters.get(ev["event_id"], {})
        event_feats.append({**f, "event_id": ev["event_id"], "type": meta.get("type", "unknown")})

    feat_only = [{k: v for k, v in f.items() if k in FEATURE_DIR} for f in event_feats]
    model = _fit_model(feat_only)
    MODEL_OUT.write_text(json.dumps(model, indent=2))

    df = pd.DataFrame(event_feats).set_index("event_id")
    print(f"\nDisaster event feature summary (N={model['n_events']})\n")
    print(df[list(FEATURE_DIR)].describe().round(2).to_string())

    print("\nFitted distributions (μ ± σ):")
    for feat, p in model["features"].items():
        dir_sym = "↑" if p["direction"] == 1 else "↓"
        print(f"  {feat:22s} {dir_sym}  μ={p['mu']:8.2f}  σ={p['sigma']:7.2f}  "
              f"[{p['min']:.1f} – {p['max']:.1f}]")

    print("\nTop Spearman correlations:")
    corr_df = pd.DataFrame(model["spearman_corr"])
    cols = list(FEATURE_DIR)
    pairs = [(corr_df.loc[a, b], a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    for r, a, b in sorted(pairs, key=lambda x: abs(x[0]), reverse=True)[:8]:
        print(f"  {a:22s} × {b:22s}  ρ={r:+.3f}")

    print(f"\nModel saved → {MODEL_OUT}")
    return model


if __name__ == "__main__":
    main()
