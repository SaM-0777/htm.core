from dataclasses import dataclass
from htm.bindings.sdr import SDR

COVERAGE_MAP = {
    "CLR": 0,
    "SKC": 0,
    "FEW": 1,
    "SCT": 2,
    "BKN": 3,
    "OVC": 4,
    "VV": 5,
}


@dataclass(frozen=True)
class CoverageEncoderConfig:
    sdr_size: int = 128
    active_bits: int = 16


class CoverageEncoder:
    """
    Hardcoded with graded semantic overlap based on oktas distance
    """

    def __init__(self, config: CoverageEncoderConfig | None = None):
        self.config = config or CoverageEncoderConfig()

        self.category_sdrs = {}

        # Base patterns with increasing overlap for closer categories
        base_patterns = [
            list(range(0, 16)),  # 0: CLR / SKC
            list(range(5, 21)),  # 1: FEW     (strong overlap with CLR)
            list(range(11, 27)),  # 2: SCT     (strong overlap with FEW)
            list(range(17, 33)),  # 3: BKN     (strong overlap with SCT)
            list(range(24, 40)),  # 4: OVC     (medium overlap with BKN)
            list(range(29, 45)),  # 5: VV      (very strong overlap with OVC)
        ]

        for i, indices in enumerate(base_patterns):
            sdr = SDR(self.config.sdr_size)
            sdr.sparse = indices
            self.category_sdrs[i] = sdr

        print("CoverageEncoder (Graded Semantic Overlap) ready")

    @property
    def output_size(self) -> int:
        return self.config.sdr_size

    def encode(self, coverage: str | None) -> SDR:
        cat_id = COVERAGE_MAP.get(coverage or "CLR", 0)
        return self.category_sdrs[cat_id]

    def overlap(self, a: str, b: str) -> float:
        sdr_a = self.encode(a)
        sdr_b = self.encode(b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.config.active_bits, 4)
