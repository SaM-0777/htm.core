from dataclasses import dataclass
from htm.bindings.sdr import SDR

from metar.encoders.coverage_encoder import CoverageEncoder
from metar.encoders.cloud_height_encoder import CloudHeightEncoder
from metar.encoders.cloud_type_encoder import CloudTypeEncoder


@dataclass(frozen=True)
class CloudLayerEncoderConfig:
    total_size: int = 512


class CloudLayerEncoder:
    def __init__(self, config: CloudLayerEncoderConfig | None = None):
        self.config = config or CloudLayerEncoderConfig()
        self.coverage_enc = CoverageEncoder()
        self.height_enc = CloudHeightEncoder()
        self.type_enc = CloudTypeEncoder()
        self.output_sdr = SDR(self.config.total_size)

    def encode(
        self,
        coverage: str | None = None,
        height_feet: float | int | None = None,
        cloud_type: str | None = None,
    ) -> SDR:
        cov_sdr = self.coverage_enc.encode(coverage)
        hgt_sdr = self.height_enc.encode(height_feet)
        typ_sdr = self.type_enc.encode(cloud_type)

        self.output_sdr.sparse = []

        for bit in cov_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit]

        for bit in hgt_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + 192]

        for bit in typ_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + 352]

        return self.output_sdr

    def overlap(self, layer1: tuple, layer2: tuple) -> float:
        sdr1 = self.encode(*layer1)
        sdr2 = self.encode(*layer2)
        intersection = len(set(sdr1.sparse) & set(sdr2.sparse))
        return round(intersection / self.config.total_size, 4)
