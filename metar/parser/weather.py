INTENSITIES = {
    "+": "heavy",
    "-": "light",
}

DESCRIPTORS = {
    "TS": "thunderstorm",
    "SH": "showers",
    "FZ": "freezing",
}

PHENOMENA = {
    "RA": "rain",
    "SN": "snow",
    "FG": "fog",
    "BR": "mist",
    "DZ": "drizzle",
    "GR": "hail",
    "HZ": "haze",
}


def parse_weather(
    token: str,
) -> dict:

    result = {
        "raw": token,
        "intensity": None,
        "descriptors": [],
        "phenomena": [],
    }

    #
    # Intensity
    #

    if token[:1] in INTENSITIES:

        result["intensity"] = INTENSITIES[token[:1]]

        token = token[1:]

    #
    # Descriptors
    #

    for code, meaning in DESCRIPTORS.items():

        if code in token:
            result["descriptors"].append(meaning)

    #
    # Phenomena
    #

    for code, meaning in PHENOMENA.items():

        if code in token:
            result["phenomena"].append(meaning)

    return result
