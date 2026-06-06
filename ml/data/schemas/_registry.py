from pathlib import Path

_HERE = Path(__file__).parent

# Only SDM types get jsonschema validation; all other loaders pass through.
SCHEMA_REGISTRY: dict[str, dict] = {
    "TrafficFlowObserved": {
        "path": _HERE / "TrafficFlowObserved" / "schema.json",
        "source_repo": "smart-data-models/dataModel.Transportation",
        "commit": "1eee77e333c9",
    },
    "WeatherObserved": {
        "path": _HERE / "WeatherObserved" / "schema.json",
        "source_repo": "smart-data-models/dataModel.Weather",
        "commit": "c857c0c440c7",
    },
    "WeatherForecast": {
        "path": _HERE / "WeatherForecast" / "schema.json",
        "source_repo": "smart-data-models/dataModel.Weather",
        "commit": "9e6bc8cd2de0",
    },
    "ParkingSpot": {
        "path": _HERE / "ParkingSpot" / "schema.json",
        "source_repo": "smart-data-models/dataModel.Parking",
        "commit": "be8db6f7ae5a",
    },
    "CrowdFlowObserved": {
        "path": _HERE / "CrowdFlowObserved" / "schema.json",
        "source_repo": "eurotech-hk/dataModel.Immigration",
        "commit": "local",
    },
}
