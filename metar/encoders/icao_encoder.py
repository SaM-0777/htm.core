from htm.bindings.encoders import SDRCategoryEncoder
from htm.bindings.sdr import SDR


class ICAOEncoder:
    """
    Encodes ICAO airport identifiers.

    Example:
    EGLC -> sparse SDR
    """

    def __init__(
        self,
        categories: list[str],
        size: int = 128,
        active_bits: int = 8,
    ) -> None:
        if not categories:
            raise ValueError("categories cannot be empty")

        self.encoder = SDRCategoryEncoder(
            n=size,
            w=active_bits,
            categoryList=categories,
        )

        self.output_width = size

    def encode(self, icao_id: str) -> SDR:
        if not isinstance(icao_id, str):
            raise TypeError("icao_id must be str")

        return self.encoder.encode(icao_id)
