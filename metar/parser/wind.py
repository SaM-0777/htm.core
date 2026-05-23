import re

WIND_RE = re.compile(r"^(VRB|\d{3}|///)" r"(\d{2,3})" r"(G(\d{2,3}))?" r"KT$")

VARIABLE_WIND_RE = re.compile(r"^(\d{3})V(\d{3})$")


def parse_wind(
    token: str,
) -> dict:

    result = {
        "raw": token,
        "direction": None,
        "speed_kt": None,
        "gust_kt": None,
        "variable": False,
        "variable_range": None,
    }

    match = WIND_RE.match(token)

    if not match:
        return result

    direction, speed, _, gust = match.groups()

    if direction == "VRB":
        result["variable"] = True

    elif direction != "///":
        result["direction"] = int(direction)

    result["speed_kt"] = int(speed)

    if gust:
        result["gust_kt"] = int(gust)

    return result


def parse_variable_wind(
    token: str,
) -> dict | None:

    match = VARIABLE_WIND_RE.match(token)

    if not match:
        return None

    low, high = match.groups()

    return {
        "raw": token,
        "from": int(low),
        "to": int(high),
    }
