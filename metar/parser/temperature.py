import math
import re

TEMP_RE = re.compile(r"^(M?\d{2})/(M?\d{2}|//)$")


def parse_temp(
    token: str,
) -> dict | None:

    match = TEMP_RE.match(token)

    if not match:
        return None

    temp_raw, dew_raw = match.groups()

    def parse(v: str):

        if v == "//":
            return None

        if v.startswith("M"):
            return -int(v[1:])

        return int(v)

    temp = parse(temp_raw)
    dew = parse(dew_raw)

    humidity = None

    #if temp is not None and dew is not None:

    #    #
    #    # Magnus approximation
    #    #

    #    a = 17.625
    #    b = 243.04

    #    rh = (math.exp((a * dew) / (b + dew)) / math.exp((a * temp) / (b + temp))) * 100

    #    humidity = round(rh, 2)

    return {
        "raw": token,
        "temperature_c": temp,
        "dew_point_c": dew,
        "relative_humidity": humidity,
    }
