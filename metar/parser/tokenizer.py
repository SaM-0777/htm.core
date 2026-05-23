from dataclasses import dataclass


@dataclass(slots=True)
class Token:

    raw: str
    type: str = "unknown"
