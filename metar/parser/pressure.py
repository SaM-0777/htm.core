import re

PRESSURE_RE = re.compile(r"^(Q\d{4}|A\d{4})$")


def parse_pressure(
    token: str,
) -> dict | None:

    match = PRESSURE_RE.match(token)

    if not match:
        return None

    result = {
        "raw": token,
        "hpa": None,
        "inhg": None,
    }

    if token.startswith("Q"):
        result["hpa"] = int(token[1:])

    elif token.startswith("A"):

        value = int(token[1:])

        result["inhg"] = round(
            value / 100,
            2,
        )

    return result
