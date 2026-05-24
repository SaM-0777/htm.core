from __future__ import annotations
from dataclasses import dataclass
from math import log1p
from htm.bindings.encoders import (
    ScalarEncoder,
    ScalarEncoderParameters,
)
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class VisibilityEncoderConfig:
    sdr_size: int = 256
    active_bits: int = 12
    clip_input: bool = True

    minimum_visibility_m: float = 0.0
    maximum_visibility_m: float = 20000.0


class VisibilityEncoder:
    def __init__(self, config: VisibilityEncoderConfig | None = None):
        self.config = config or VisibilityEncoderConfig()
        self.output_sdr = SDR(self.config.sdr_size)

        params = ScalarEncoderParameters()
        params.minimum = log1p(self.config.minimum_visibility_m)
        params.maximum = log1p(self.config.maximum_visibility_m)
        params.activeBits = self.config.active_bits
        params.resolution = 0.12
        params.periodic = False
        self.encoder = ScalarEncoder(params)
        self.output_sdr = SDR(self.encoder.dimensions)

    @property
    def output_size(self) -> int:
        return self.config.sdr_size

    @property
    def output_dimensions(
        self,
    ) -> tuple[int]:
        return tuple(self.output_sdr.dimensions)

    def encode(self, visibility_meters: float | int) -> SDR:
        value = float(visibility_meters)
        value = min(max(value, 0.0), 20000.0)

        if self.config.clip_input:
            value = max(
                self.config.minimum_visibility_m,
                value,
            )
            value = min(
                self.config.maximum_visibility_m,
                value,
            )

        transformed = log1p(value)
        self.encoder.encode(
            transformed,
            self.output_sdr,
        )

        result = SDR(self.output_sdr.dimensions)
        result.setSDR(self.output_sdr)

        return result

    def encode_dense(
        self,
        visibility_m: float,
    ):
        sdr = self.encode(visibility_m)
        return sdr.dense.copy()

    def overlap(self, vis_a: float | int, vis_b: float | int) -> float:
        sdr_a = self.encode(vis_a)
        sdr_b = self.encode(vis_b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.config.active_bits, 4)
