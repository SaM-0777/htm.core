from __future__ import annotations
from dataclasses import dataclass
from htm.bindings.encoders import (
    ScalarEncoder,
    ScalarEncoderParameters,
)
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class WindGustEncoderConfig:
    minimum_kt: float = 0.0
    maximum_kt: float = 200.0
    sdr_size: int = 256
    active_bits: int = 12
    clip_input: bool = True


class WindGustEncoder:
    def __init__(
        self,
        config: WindGustEncoderConfig | None = None,
    ):

        self.config = config or WindGustEncoderConfig()
        params = ScalarEncoderParameters()
        params.minimum = self.config.minimum_kt
        params.maximum = self.config.maximum_kt
        params.size = self.config.sdr_size
        params.activeBits = self.config.active_bits
        params.periodic = False

        self.encoder = ScalarEncoder(params)
        self.output_sdr = SDR(self.encoder.dimensions)

    @property
    def output_size(self) -> int:
        return self.output_sdr.size

    @property
    def output_dimensions(
        self,
    ) -> tuple[int]:
        return tuple(self.output_sdr.dimensions)

    @property
    def active_bits(self) -> int:
        return self.config.active_bits

    def encode(
        self,
        gust_kt: float | None,
    ) -> SDR:
        if gust_kt is None:
            empty_sdr = SDR(self.output_dimensions)
            empty_sdr.zero()
            return empty_sdr

        value = float(gust_kt)

        if self.config.clip_input:
            value = max(
                self.config.minimum_kt,
                value,
            )
            value = min(
                self.config.maximum_kt,
                value,
            )

        self.encoder.encode(
            value,
            self.output_sdr,
        )

        result = SDR(self.output_sdr.dimensions)
        result.setSDR(self.output_sdr)

        return result

    def encode_dense(
        self,
        gust_kt: float | None,
    ):
        sdr = self.encode(gust_kt)
        return sdr.dense.copy()

    def encode_sparse(
        self,
        gust_kt: float | None,
    ) -> list[int]:
        sdr = self.encode(gust_kt)
        return list(sdr.sparse)

    def overlap(
        self,
        gust_a: float | None,
        gust_b: float | None,
    ) -> float:
        sdr_a = self.encode(gust_a)
        sdr_b = self.encode(gust_b)
        active = max(
            len(sdr_a.sparse),
            1,
        )

        overlap = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return overlap / active
