from dataclasses import dataclass
from htm.bindings.sdr import SDR

from encoders.coverage_encoder import CoverageEncoder
from encoders.cloud_height_encoder import CloudHeightEncoder
from encoders.cloud_type_encoder import CloudTypeEncoder


@dataclass(frozen=True)
class CloudLayerEncoderConfig:
    total_size: int = 512


class CloudLayerEncoder:
    """
    Single Cloud Layer Encoder - 512 bits total
    Clean Regional Concatenation (Correct htm.core API)
    """

    def __init__(self, config: CloudLayerEncoderConfig | None = None):
        self.config = config or CloudLayerEncoderConfig()

        self.coverage_enc = CoverageEncoder()  # 128 bits
        self.height_enc = CloudHeightEncoder()  # 128 bits
        self.type_enc = CloudTypeEncoder()  # 128 bits

        self.output_sdr = SDR(self.config.total_size)

        print(f"CloudLayerEncoder ready → Total size = {self.config.total_size} bits")

    def encode(
        self,
        coverage: str | None = None,
        height_feet: float | int | None = None,
        cloud_type: str | None = None,
    ) -> SDR:

        # Encode each component
        cov_sdr = self.coverage_enc.encode(coverage)
        hgt_sdr = self.height_enc.encode(height_feet)
        typ_sdr = self.type_enc.encode(cloud_type)

        # Clear the output SDR first
        self.output_sdr.sparse = []

        # Copy each sub-SDR into its region using sparse indices
        # Coverage: bits 0 - 127
        for bit in cov_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit]

        # Height: bits 192 - 319
        for bit in hgt_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + 192]

        # Type: bits 352 - 479
        for bit in typ_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + 352]

        return self.output_sdr

    def overlap(self, layer1: tuple, layer2: tuple) -> float:
        """layer1 and layer2 = (coverage, height_feet, cloud_type)"""
        sdr1 = self.encode(*layer1)
        sdr2 = self.encode(*layer2)
        intersection = len(set(sdr1.sparse) & set(sdr2.sparse))
        return round(intersection / self.config.total_size, 4)
