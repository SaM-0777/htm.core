from dataclasses import dataclass
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class CloudHeightEncoderConfig:
    sdr_size: int = 128
    active_bits: int = 16


class CloudHeightEncoder:
    """
    Cloud Height Encoder - Takes height in feet.
    Designed with proper meteorological semantic overlap.
    Lower height = higher atmospheric impact.
    """

    def __init__(self, config: CloudHeightEncoderConfig | None = None):
        self.config = config or CloudHeightEncoderConfig()

        self.category_sdrs = {}

        # Semantic level patterns with controlled overlap
        # Closer heights + similar impact = higher overlap
        self.level_patterns = [
            list(
                range(0, 16)
            ),  # Level 0: Extremely Low (<800 ft)     - Very High Impact
            list(range(6, 22)),  # Level 1: Very Low (800 - 2000 ft)    - High Impact
            list(
                range(13, 29)
            ),  # Level 2: Low (2000 - 4000 ft)        - Medium-High Impact
            list(range(21, 37)),  # Level 3: Medium (4000 - 7000 ft)     - Medium Impact
            list(range(30, 46)),  # Level 4: High (7000 - 14000 ft)      - Low Impact
            list(
                range(42, 58)
            ),  # Level 5: Very High (>14000 ft or CLR) - Very Low Impact
        ]

        for i, indices in enumerate(self.level_patterns):
            sdr = SDR(self.config.sdr_size)
            sdr.sparse = indices
            self.category_sdrs[i] = sdr

        print(
            f"CloudHeightEncoder initialized → size={self.config.sdr_size}, "
            f"active_bits={self.config.active_bits}"
        )

    @property
    def output_size(self) -> int:
        return self.config.sdr_size

    def _get_semantic_level(self, height_feet: float | int | None) -> int:
        """Convert height in feet to semantic impact level (0-5)"""
        if height_feet is None or height_feet > 25000:
            return 5

        height = float(height_feet)

        if height < 800:
            return 0
        elif height < 2000:
            return 1
        elif height < 4000:
            return 2
        elif height < 7000:
            return 3
        elif height < 14000:
            return 4
        else:
            return 5

    def encode(self, height_feet: float | int | None) -> SDR:
        """Encode cloud height in feet"""
        level = self._get_semantic_level(height_feet)
        return self.category_sdrs[level]

    def overlap(
        self, height_a: float | int | None, height_b: float | int | None
    ) -> float:
        """Calculate overlap between two heights"""
        sdr_a = self.encode(height_a)
        sdr_b = self.encode(height_b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.config.active_bits, 4)
