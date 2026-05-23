def parse_station(
    tokens: list[str],
) -> str | None:

    if len(tokens) >= 2:
        return tokens[1]

    return None
