from dataclasses import dataclass
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class CloudHeightEncoderConfig:
    sdr_size: int = 256
    active_bits: int = 11


class CloudHeightEncoder:
    def __init__(self, config: CloudHeightEncoderConfig | None = None):
        self.config = config or CloudHeightEncoderConfig()
        self.category_sdrs = {}
        self.level_patterns = [
            list(range(0, 16)),
            list(range(6, 22)),
            list(range(13, 29)),
            list(range(21, 37)),
            list(range(30, 46)),
            list(range(42, 58)),
        ]

        for i, indices in enumerate(self.level_patterns):
            sdr = SDR(self.config.sdr_size)
            sdr.sparse = indices
            self.category_sdrs[i] = sdr

    @property
    def output_size(self) -> int:
        return self.config.sdr_size

    def _get_semantic_level(self, height_feet: float | int | None) -> int:
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
        level = self._get_semantic_level(height_feet)
        return self.category_sdrs[level]

    def overlap(
        self, height_a: float | int | None, height_b: float | int | None
    ) -> float:
        sdr_a = self.encode(height_a)
        sdr_b = self.encode(height_b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.config.active_bits, 4)
