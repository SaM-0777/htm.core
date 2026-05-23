from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedMetar:

    #
    # Raw
    #

    raw_metar: str

    #
    # Temporal
    #

    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None

    #
    # Station
    #

    station: str | None = None

    #
    # Wind
    #

    wind: dict[str, Any] = field(default_factory=dict)

    #
    # Visibility
    #

    visibility: dict[str, Any] = field(default_factory=dict)

    #
    # Clouds
    #

    clouds: list[dict[str, Any]] = field(default_factory=list)

    #
    # Weather phenomena
    #

    weather: list[dict[str, Any]] = field(default_factory=list)

    #
    # Temperature
    #

    temperature_c: int | None = None

    dew_point_c: int | None = None

    relative_humidity: float | None = None

    #
    # Pressure
    #

    pressure_hpa: int | None = None

    pressure_inhg: float | None = None

    #
    # Unknown tokens
    #

    unknown_tokens: list[str] = field(default_factory=list)

    #
    # Remarks
    #

    remarks: list[str] = field(default_factory=list)
