from dataclasses import dataclass
from htm.bindings.encoders import (
    ScalarEncoder,
    ScalarEncoderParameters,
)
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class DewPointEncoderConfig:
    minimum_celsius: float = -80.0
    maximum_celsius: float = 50.0
    active_bits: int = 8
    sdr_size: int = 192
    resolution_celsius: float = 0.25
    periodic: bool = False
    clip_input: bool = True


class DewPointEncoder:
    def __init__(
        self,
        config: DewPointEncoderConfig | None = None,
    ):
        self.config = config or DewPointEncoderConfig()
        params = ScalarEncoderParameters()
        params.minimum = self.config.minimum_celsius
        params.maximum = self.config.maximum_celsius
        params.activeBits = self.config.active_bits
        params.size = self.config.sdr_size
        params.periodic = self.config.periodic

        self.encoder = ScalarEncoder(params)
        self.output_sdr = SDR(self.encoder.dimensions)

    @property
    def output_dimensions(
        self,
    ) -> tuple[int]:
        return tuple(self.encoder.dimensions)

    @property
    def output_size(
        self,
    ) -> int:
        return self.output_sdr.size

    def encode(
        self,
        dew_point_celsius: float,
    ) -> SDR:
        value = dew_point_celsius

        if self.config.clip_input:
            value = max(
                self.config.minimum_celsius,
                value,
            )
            value = min(
                self.config.maximum_celsius,
                value,
            )

        self.encoder.encode(
            value,
            self.output_sdr,
        )

        return self.output_sdr

    def encode_dense(
        self,
        dew_point_celsius: float,
    ):
        sdr = self.encode(dew_point_celsius)
        return sdr.dense

    def overlap(
        self,
        dew_point_a: float,
        dew_point_b: float,
    ) -> float:
        sdr_a = SDR(self.output_dimensions)
        sdr_b = SDR(self.output_dimensions)

        self.encoder.encode(
            dew_point_a,
            sdr_a,
        )
        self.encoder.encode(
            dew_point_b,
            sdr_b,
        )

        overlap = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return overlap / self.config.active_bits
