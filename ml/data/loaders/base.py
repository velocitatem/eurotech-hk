from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterator


class Loader(ABC):
    """Base class for all city data source adapters.

    Subclass this and register in loaders/registry.py. Real loaders
    from parallel work need only implement these three attributes.

    Returned dicts must include at minimum:
      schema_type: str
      zone_id: str
      timestamp: str  (ISO-8601)
      Any number of feature key-value pairs

    For SDM sources (TrafficFlowObserved, WeatherObserved, WeatherForecast,
    ParkingSpot) the full SDM entity structure is expected so jsonschema
    validation runs. All others are passed through as-is.
    """

    @property
    @abstractmethod
    def loader_id(self) -> str: ...

    @property
    @abstractmethod
    def schema_type(self) -> str: ...

    @abstractmethod
    def fetch(self, start: str, end: str, zones: list[str]) -> Iterator[dict[str, Any]]: ...
