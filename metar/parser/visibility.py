def parse_visibility(
    token: str,
) -> dict:

    result = {
        "raw": token,
        "meters": None,
        "cavok": False,
    }

    if token == "CAVOK":
        result["cavok"] = True
        result["meters"] = 10000
        return result

    if token.isdigit():
        result["meters"] = int(token)

    return result
