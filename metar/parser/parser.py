import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Union

from metar import Metar  # type: ignore


def parse_recorded_time(recorded_time_str: str) -> Dict[str, int]:
    """Convert '2026-04-27T09:00' → year/month/day/hour/minute"""
    if recorded_time_str.endswith("Z"):
        recorded_time_str = recorded_time_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(recorded_time_str)
    return {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
    }


def safe_value(obj: Any) -> Any:
    """Safely get .value() from python-metar objects"""
    if obj is None:
        return None
    try:
        if hasattr(obj, "value") and callable(obj.value):
            return obj.value()
        return obj
    except:
        return str(obj) if obj else None


def parse_metar_to_dict(raw_metar: str, station: str | None = None) -> Dict[str, Any]:
    """Parse METAR using python-metar"""
    try:
        obs = Metar.Metar(raw_metar)
        
        print("\n" , obs)

        # Wind
        wind: Dict[str, Any] = {}
        if (
            getattr(obs, "wind_dir", None) is not None
            or getattr(obs, "wind_speed", None) is not None
        ):
            wind = {
                "direction_deg": safe_value(getattr(obs, "wind_dir", None)),
                "speed_kt": safe_value(getattr(obs, "wind_speed", None)),
                "gust_kt": safe_value(getattr(obs, "wind_gust", None)),
                "variable_range": None,
            }
            if (
                getattr(obs, "wind_dir_from", None) is not None
                and getattr(obs, "wind_dir_to", None) is not None
            ):
                wind["variable_range"] = {
                    "from": safe_value(getattr(obs, "wind_dir_from", None)),
                    "to": safe_value(getattr(obs, "wind_dir_to", None)),
                }

        # Clouds
        clouds: List[Dict[str, Any]] = []
        for cloud in getattr(obs, "sky", []):
            coverage = cloud[0] if len(cloud) > 0 else None
            height = cloud[1] if len(cloud) > 1 else None
            ctype = cloud[2] if len(cloud) > 2 else None
            clouds.append(
                {
                    "coverage": coverage,
                    "height_ft": safe_value(height),
                    "type": ctype,
                }
            )

        # Weather
        weather: List[Dict[str, Any]] = []
        for w in getattr(obs, "weather", []):
            weather.append(
                {
                    "intensity": w[0] if len(w) > 0 else None,
                    "description": w[1] if len(w) > 1 else None,
                    "precipitation": w[2] if len(w) > 2 else None,
                    "obscuration": w[3] if len(w) > 3 else None,
                    "other": w[4] if len(w) > 4 else None,
                }
            )

        parsed = {
            "raw_metar": raw_metar,
            "station": station or getattr(obs, "station_id", None),
            "wind": wind,
            "visibility": (
                {
                    "distance": safe_value(getattr(obs, "vis", None)),
                    "units": "m",
                }
                if getattr(obs, "vis", None) is not None
                else {}
            ),
            "clouds": clouds,
            "weather": weather,
            "temperature_c": safe_value(getattr(obs, "temp", None)),
            "dew_point_c": safe_value(getattr(obs, "dewpt", None)),
            "pressure_hpa": safe_value(getattr(obs, "press", None)),
            "pressure_inhg": None,
            "relative_humidity": None,
            "remarks": getattr(obs, "_remarks", []),
            "unknown_tokens": getattr(obs, "_unparsed_groups", []),
            "auto": "AUTO" in raw_metar.upper(),
            "error": None,
        }

        return parsed

    except Exception as e:
        return {
            "raw_metar": raw_metar,
            "station": station,
            "error": str(e),
            "wind": {},
            "visibility": {},
            "clouds": [],
            "weather": [],
            "temperature_c": None,
            "dew_point_c": None,
            "pressure_hpa": None,
            "remarks": [],
            "unknown_tokens": [],
            "auto": False,
        }


def main(
    input_file: str, output_file: str, filter_stations: List[str] | None = None
) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        data: Union[Dict[str, Any], List[Dict[str, Any]]] = json.load(f)

    records: List[Dict[str, Any]] = [data] if isinstance(data, dict) else data
    output_records: List[Dict[str, Any]] = []

    filter_set = {s.upper() for s in filter_stations} if filter_stations else None

    for record in records:
        raw_metar: str | None = record.get("raw_metar_code") or record.get("raw_metar")
        station: str | None = record.get("icao_id") or record.get("station")

        if not raw_metar:
            continue

        # Filter by station if provided
        if filter_set and station:
            if station.upper() not in filter_set:
                continue

        parsed = parse_metar_to_dict(raw_metar, station)

        if "recorded_time" in record and record["recorded_time"]:
            time_fields = parse_recorded_time(record["recorded_time"])
            parsed.update(time_fields)

        enriched: Dict[str, Any] = {
            "id": record.get("id"),
            "data_provider": record.get("data_provider"),
            "city_name": record.get("city_name"),
            "name": record.get("name"),
            "icao_id": record.get("icao_id"),
            **parsed,
            "original_temperature": record.get("temperature"),
            "original_dew_point": record.get("dew_point"),
        }
        output_records.append(enriched)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            output_records[0] if len(output_records) == 1 else output_records,
            f,
            indent=2,
            ensure_ascii=False,
        )

    filter_info = f" (filtered to {len(filter_set)} stations)" if filter_set else ""
    print(
        f"✅ Successfully parsed {len(output_records)} METAR record(s){filter_info} → {output_file}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python parser.py <input.json> <output.json> [STATION1] [STATION2] ...")
        print("Examples:")
        print("  python parser.py ./data/metar.json ./data/parsed.json")
        print("  python parser.py ./data/metar.json ./data/parsed.json EGLC")
        print("  python parser.py ./data/metar.json ./data/parsed.json EGLC LFPB RFKI")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    stations = sys.argv[3:] if len(sys.argv) > 3 else None

    main(input_file, output_file, stations)
