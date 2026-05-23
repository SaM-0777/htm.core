import re

CLOUD_COVERAGES = {
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

CLOUD_TYPES = {
    "CB",
    "TCU",
    "ACC",
    "ACSL",
    "CI",
    "CC",
    "CS",
}


def parse_cloud(
    token: str,
) -> dict:

    result = {
        "raw": token,
        "coverage": None,
        "altitude_ft": None,
        "cloud_type": None,
    }

    #
    # Coverage
    #

    for coverage in CLOUD_COVERAGES:

        if token.startswith(coverage):
            result["coverage"] = coverage
            break

    #
    # Altitude
    #

    altitude = re.search(
        r"(\d{3})",
        token,
    )

    if altitude:
        result["altitude_ft"] = int(altitude.group(1)) * 100

    #
    # Cloud type
    #

    for cloud_type in CLOUD_TYPES:

        if cloud_type in token:
            result["cloud_type"] = cloud_type
            break

    return result
