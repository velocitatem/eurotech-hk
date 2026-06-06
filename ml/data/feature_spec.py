"""Single source of truth for all feature groups, scopes, and SDM-attribute mappings.

Leakage rule (spec §11):
  WeatherObserved.* and all historically-observed series → STATE (appears in X_context AND Y_target)
  WeatherForecast.* and all forward-issued signals → CONDITION (C_future only)
  Global conditions (typhoon level, HSI, oil price, unemployment) are broadcast uniformly
  across all zones at panel-build time; no special model architecture needed.
"""
from __future__ import annotations

# ---------- state features: observed context + prediction target ----------
# Each is a zonal time series unless marked (global) — globals are broadcast.
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

# ---------- condition features: known/forecasted inputs for C_future ----------
CONDITION_FEATURES: list[str] = [
    "temperature_forecast",     # °C
    "precipitation_forecast",   # probability 0-1
    "wind_forecast",            # km/h
    "weather_alert_level",      # 0-3 mapped from weatherType string
    "typhoon_warning_level",    # 0,1,3,8,10 (HKO T-signal)
    "rainstorm_warning_level",  # 0=none, 1=amber, 2=red, 3=black
    "hour_sin",                 # sin(2π*h/24)
    "hour_cos",                 # cos(2π*h/24)
    "dow_sin",                  # sin(2π*d/7)
    "dow_cos",                  # cos(2π*d/7)
    "event_count",              # planned public events (zonal)
    "event_type",               # 0=none,1=sports,2=concert,3=protest,4=holiday_mkt
    "holiday",                  # 0/1 HK public holiday
    "school_day",               # 0/1
    "unemployment_rate",        # % (global, monthly-updated, broadcast)
    "oil_price_usd",            # WTI USD/barrel (global, broadcast)
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
COND_IDX:   dict[str, int] = {f: i for i, f in enumerate(CONDITION_FEATURES)}
STATIC_IDX: dict[str, int] = {f: i for i, f in enumerate(STATIC_FEATURES)}

N_STATE  = len(STATE_FEATURES)
N_COND   = len(CONDITION_FEATURES)
N_STATIC = len(STATIC_FEATURES)

# Decoder domain → indices into STATE_FEATURES
DECODER_DOMAINS: dict[str, list[int]] = {
    "traffic": [STATE_IDX[f] for f in ("traffic_intensity", "traffic_speed", "traffic_occupancy")],
    "parking": [STATE_IDX["parking_occupancy"]],
    "environment": [STATE_IDX[f] for f in ("air_quality_index", "noise_level")],
    "energy": [STATE_IDX["energy_demand"]],
}
