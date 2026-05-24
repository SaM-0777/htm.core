from __future__ import annotations
from dataclasses import dataclass
from htm.bindings.encoders import (
    ScalarEncoder,
    ScalarEncoderParameters,
)
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class WindSpeedEncoderConfig:
    minimum_kt: float = 0.0
    maximum_kt: float = 150.0
    active_bits: int = 21
    sdr_size: int = 512
    resolution_kt: float = 1.0
    clip_input: bool = True


class WindSpeedEncoder:
    def __init__(
        self,
        config: WindSpeedEncoderConfig | None = None,
    ) -> None:
        self.config = config or WindSpeedEncoderConfig()
        params = ScalarEncoderParameters()
        params.minimum = self.config.minimum_kt
        params.maximum = self.config.maximum_kt
        params.activeBits = self.config.active_bits
        params.size = self.config.sdr_size
        params.periodic = False

        self.encoder = ScalarEncoder(params)
        self.output_sdr = SDR(self.encoder.dimensions)

    @property
    def output_dimensions(self) -> tuple[int]:
        return tuple(self.encoder.dimensions)

    @property
    def output_size(self) -> int:
        return self.output_sdr.size

    @property
    def active_bits(self) -> int:
        return self.config.active_bits

    def encode(
        self,
        speed_kt: float,
    ) -> SDR:
        value = float(speed_kt)

        if self.config.clip_input:
            value = max(self.config.minimum_kt, value)
            value = min(self.config.maximum_kt, value)

        self.encoder.encode(value, self.output_sdr)

        return self.output_sdr

    def encode_dense(
        self,
        speed_kt: float,
    ):
        sdr = self.encode(speed_kt)
        return sdr.dense.copy()

    def encode_sparse(
        self,
        speed_kt: float,
    ) -> list[int]:
        sdr = self.encode(speed_kt)
        return list(sdr.sparse)

    def overlap(
        self,
        speed_a: float,
        speed_b: float,
    ) -> float:
        sdr_a = SDR(self.output_dimensions)
        sdr_b = SDR(self.output_dimensions)

        self.encoder.encode(float(speed_a), sdr_a)
        self.encoder.encode(float(speed_b), sdr_b)

        overlap = len(set(sdr_a.sparse) & set(sdr_b.sparse))

        return overlap / self.config.active_bits

    def similarity(
        self,
        speed_a: float,
        speed_b: float,
    ) -> float:
        return self.overlap(speed_a, speed_b)
