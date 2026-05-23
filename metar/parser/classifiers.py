import re
from dataclasses import dataclass
from typing import Final

#
# =========================
# METAR TOKEN TAXONOMY
# =========================
#

CLOUD_PREFIXES: Final = {
    "FEW",
    "SCT",
    "BKN",
    "OVC",
    "VV",
    "NSC",
    "NCD",
    "SKC",
    "CLR",
}

CLOUD_TYPES: Final = {
    "CB",
    "TCU",
    "ACC",
    "ACSL",
    "CI",
    "CC",
    "CS",
}

WEATHER_CODES: Final = {
    "RA",
    "SN",
    "FG",
    "TS",
    "DZ",
    "BR",
    "HZ",
    "GR",
    "GS",
    "PL",
    "SG",
    "IC",
    "UP",
    "SQ",
    "FC",
    "SS",
    "DS",
}

WEATHER_DESCRIPTORS: Final = {
    "SH",
    "TS",
    "FZ",
    "BL",
    "DR",
    "MI",
    "BC",
    "PR",
}

INTENSITY_CODES: Final = {
    "+",
    "-",
}

#
# =========================
# REGEXES
# =========================
#

WIND_RE = re.compile(r"^(VRB|\d{3}|///)" r"(\d{2,3})" r"(G\d{2,3})?" r"KT$")

VARIABLE_WIND_RE = re.compile(r"^\d{3}V\d{3}$")

VISIBILITY_RE = re.compile(r"^\d{4}$|^CAVOK$")

TEMP_RE = re.compile(r"^(M?\d{2})/(M?\d{2}|//)$")

PRESSURE_RE = re.compile(r"^(Q\d{4}|A\d{4})$")

TIME_RE = re.compile(r"^\d{6}Z$")

RUNWAY_RE = re.compile(r"^R\d{2}[LCR]?/")

#
# =========================
# CLASSIFICATION RESULT
# =========================
#


@dataclass(slots=True)
class ClassificationResult:

    token: str

    #
    # Multi-label support
    #

    labels: list[str]

    #
    # Confidence score
    #

    confidence: float


#
# =========================
# CLASSIFIER
# =========================
#


def classify_token(
    token: str,
) -> ClassificationResult:

    labels: list[str] = []

    confidence = 0.0

    #
    # Wind
    #

    if WIND_RE.match(token):
        labels.append("wind")
        confidence += 1.0

    #
    # Variable wind
    #

    if VARIABLE_WIND_RE.match(token):
        labels.append("variable_wind")
        confidence += 1.0

    #
    # Visibility
    #

    if VISIBILITY_RE.match(token):
        labels.append("visibility")
        confidence += 1.0

    #
    # Temperature
    #

    if TEMP_RE.match(token):
        labels.append("temperature")
        confidence += 1.0

    #
    # Pressure
    #

    if PRESSURE_RE.match(token):
        labels.append("pressure")
        confidence += 1.0

    #
    # Time
    #

    if TIME_RE.match(token):
        labels.append("time")
        confidence += 1.0

    #
    # Runway
    #

    if RUNWAY_RE.match(token):
        labels.append("runway")
        confidence += 0.9

    #
    # Clouds
    #

    if token.startswith(tuple(CLOUD_PREFIXES)):
        labels.append("cloud")
        confidence += 1.0

    #
    # Standalone cloud types
    #
    # Example:
    # //////CB
    #

    elif any(cloud_type in token for cloud_type in CLOUD_TYPES):
        labels.append("cloud")
        confidence += 0.9

    #
    # Weather descriptors
    #

    weather_hits = 0

    for code in WEATHER_CODES:

        if code in token:
            weather_hits += 1

    for descriptor in WEATHER_DESCRIPTORS:

        if descriptor in token:
            weather_hits += 1

    if weather_hits > 0:
        labels.append("weather")
        confidence += min(
            1.0,
            weather_hits * 0.25,
        )

    #
    # Intensity
    #

    if token[:1] in INTENSITY_CODES:
        labels.append("intensity")

    #
    # Unknown fallback
    #

    if not labels:
        labels.append("unknown")

    #
    # Normalize confidence
    #

    confidence = min(
        confidence,
        1.0,
    )

    return ClassificationResult(
        token=token,
        labels=labels,
        confidence=confidence,
    )
