from dataclasses import dataclass
from htm.bindings.sdr import SDR

# Major cloud types reported in METAR
CLOUD_TYPE_MAP = {
    "NONE": 0,
    "CLR": 0,
    "SKC": 0,  # Clear sky
    "CI": 1,
    "CS": 1,
    "CC": 1,  # High clouds (Cirrus group)
    "AC": 2,  # Altocumulus
    "AS": 3,  # Altostratus
    "SC": 4,  # Stratocumulus
    "ST": 5,  # Stratus
    "NS": 6,  # Nimbostratus
    "TCU": 7,  # Towering Cumulus
    "CB": 8,  # Cumulonimbus (most severe)
}


@dataclass(frozen=True)
class CloudTypeEncoderConfig:
    sdr_size: int = 128
    active_bits: int = 16


class CloudTypeEncoder:
    """
    Cloud Type Encoder with meteorologically meaningful semantic overlap.

    Rule: Similar atmospheric impact = Higher SDR overlap
    """

    def __init__(self, config: CloudTypeEncoderConfig | None = None):
        self.config = config or CloudTypeEncoderConfig()

        self.category_sdrs = {}

        # Designed based on real atmospheric similarity:
        patterns = [
            list(range(0, 16)),  # 0: CLR / SKC          - Fair weather
            list(range(8, 24)),  # 1: CI/CS/CC           - High thin clouds
            list(range(14, 30)),  # 2: AC                 - Mid-level
            list(range(20, 36)),  # 3: AS                 - Mid-level thickening
            list(range(26, 42)),  # 4: SC                 - Low stable
            list(range(32, 48)),  # 5: ST                 - Low stratus
            list(range(38, 54)),  # 6: NS                 - Rain-bearing
            list(range(44, 60)),  # 7: TCU                - Building convection
            list(
                range(50, 66)
            ),  # 8: CB                 - Thunderstorm (most different)
        ]

        for i, indices in enumerate(patterns):
            sdr = SDR(self.config.sdr_size)
            sdr.sparse = indices
            self.category_sdrs[i] = sdr

        print(
            f"CloudTypeEncoder ready - size={self.config.sdr_size}, active_bits={self.config.active_bits}"
        )

    @property
    def output_size(self) -> int:
        return self.config.sdr_size

    def encode(self, cloud_type: str | None) -> SDR:
        """
        cloud_type: METAR cloud type like 'CB', 'NS', 'CI', 'CLR', 'TCU', etc.
        """
        key = (cloud_type or "CLR").strip().upper()
        level = CLOUD_TYPE_MAP.get(key, 0)  # Default to Clear
        return self.category_sdrs[level]

    def overlap(self, a: str, b: str) -> float:
        """Calculate semantic overlap between two cloud types"""
        sdr_a = self.encode(a)
        sdr_b = self.encode(b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.config.active_bits, 4)
