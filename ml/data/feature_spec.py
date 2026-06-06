"""Single source of truth for all feature groups and SDM-attribute mappings.

JEPA formulation:  S_future = f(S_past, A_transition)

STATE = observed / measurable world facts. A_transition is now free-form text
(news, event description, policy announcement) encoded offline and passed to
the model as a dense vector — no structured CONDITION_FEATURES array required.
"""
from __future__ import annotations

# ---------- state features: observed context + prediction target ----------
STATE_FEATURES: list[str] = [
    "traffic_intensity",        # vehicles/hour per road zone
    "traffic_speed",            # avg km/h
    "traffic_occupancy",        # fraction 0-1
    "parking_occupancy",        # fraction 0-1
    "transit_delay",            # minutes vs schedule
    "crowd_flow",               # pedestrians/hour
    "air_quality_index",        # AQHI 1-10+
    "noise_level",              # dB(A)
    "temperature_observed",     # °C
    "humidity_observed",        # % RH
    "hospital_occupancy",       # fraction 0-1 (by district)
    "crime_rate",               # incidents per 100k population (zonal, monthly-updated)
    "energy_demand",            # MWh (zonal)
    "retail_sales",             # index (zonal)
    "water_consumption",        # m³/hr (zonal)
    "hsi_close",                # (global) Hang Seng Index close, broadcast to all zones
    "visitors_count",           # (global) daily arrivals, broadcast
]

# ---------- static features: per-zone, time-invariant ----------
STATIC_FEATURES: list[str] = [
    "road_density",             # km/km²
    "poi_density",              # POIs/km²
    "population_density",       # persons/km²
    "transit_connectivity",     # normalized 0-1
    "district_id",              # integer zone identifier
]

# ---------- SDM-attribute → (feature_name, group) mapping ----------
# group 'state' | 'condition' enforced at flatten time to prevent leakage.
# Non-SDM loaders return feature_name directly; they are not in this map.
SDM_ATTR_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("TrafficFlowObserved", "intensity"):                ("traffic_intensity",    "state"),
    ("TrafficFlowObserved", "averageVehicleSpeed"):      ("traffic_speed",        "state"),
    ("TrafficFlowObserved", "occupancy"):                ("traffic_occupancy",    "state"),
    ("WeatherObserved",     "temperature"):              ("temperature_observed", "state"),
    ("WeatherObserved",     "relativeHumidity"):         ("humidity_observed",    "state"),
    ("WeatherForecast",     "temperature"):              ("temperature_forecast", "condition"),
    ("WeatherForecast",     "precipitationProbability"): ("precipitation_forecast", "condition"),
    ("WeatherForecast",     "windSpeed"):                ("wind_forecast",        "condition"),
    ("WeatherForecast",     "weatherType"):              ("weather_alert_level",  "condition"),
    ("ParkingSpot",         "occupancyRate"):            ("parking_occupancy",    "state"),
    ("CrowdFlowObserved",   "passengerCount"):           ("visitors_count",        "state"),
    ("CrowdFlowObserved",   "hkResidents"):              ("visitors_count",        "state"),  # sub-field; primary key is passengerCount
}

# Non-SDM schema_type → (feature_name, group) — loader passes feature dicts directly
NON_SDM_FEATURE_MAP: dict[str, tuple[str, str]] = {
    "hospital_occupancy":   ("hospital_occupancy",   "state"),
    "crime_rate":           ("crime_rate",            "state"),
    "energy_demand":        ("energy_demand",         "state"),
    "retail_sales":         ("retail_sales",          "state"),
    "water_consumption":    ("water_consumption",     "state"),
    "hsi_close":            ("hsi_close",             "state"),
    "visitors_count":       ("visitors_count",        "state"),
    "air_quality_index":    ("air_quality_index",     "state"),
    "typhoon_warning":      ("typhoon_warning_level", "condition"),
    "rainstorm_warning":    ("rainstorm_warning_level", "condition"),
    "unemployment_rate":    ("unemployment_rate",     "condition"),
    "oil_price":            ("oil_price_usd",         "condition"),
}

STATE_IDX:  dict[str, int] = {f: i for i, f in enumerate(STATE_FEATURES)}
STATIC_IDX: dict[str, int] = {f: i for i, f in enumerate(STATIC_FEATURES)}

N_STATE  = len(STATE_FEATURES)
N_STATIC = len(STATIC_FEATURES)

# Kept for etl.py backward compat — will be removed when dataset is rebuilt.
CONDITION_FEATURES: list[str] = []
COND_IDX: dict[str, int] = {}
N_COND = 0

# Decoder domain → indices into STATE_FEATURES
DECODER_DOMAINS: dict[str, list[int]] = {
    "traffic": [STATE_IDX[f] for f in ("traffic_intensity", "traffic_speed", "traffic_occupancy")],
    "parking": [STATE_IDX["parking_occupancy"]],
    "environment": [STATE_IDX[f] for f in ("air_quality_index", "noise_level")],
    "energy": [STATE_IDX["energy_demand"]],
}
