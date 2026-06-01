from __future__ import annotations
from dataclasses import dataclass
from htm.bindings.sdr import SDR

from metar.encoders.wind_direction_encoder import (
    WindDirectionEncoder,
)
from metar.encoders.wind_speed_encoder import (
    WindSpeedEncoder,
)
from metar.encoders.wind_gust_encoder import (
    WindGustEncoder,
)


@dataclass(frozen=True)
class WindEncoderConfig:
    padding_bits: int = 16


class WindEncoder:
    def __init__(
        self,
        config: WindEncoderConfig | None = None,
    ):
        self.config = config or WindEncoderConfig()
        self.direction_encoder = WindDirectionEncoder()
        self.speed_encoder = WindSpeedEncoder()
        self.gust_encoder = WindGustEncoder()
        self.direction_size = self.direction_encoder.output_size
        self.speed_size = self.speed_encoder.output_size
        self.gust_size = self.gust_encoder.output_size
        self.padding = self.config.padding_bits

        self._output_size = (
            self.padding
            + self.direction_size
            + self.padding
            + self.speed_size
            + self.padding
            + self.gust_size
            + self.padding
        )
        self._output_dimensions = (self._output_size,)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def output_dimensions(
        self,
    ) -> tuple[int]:
        return self._output_dimensions

    def encode(
        self,
        direction_degrees: float | None,
        speed_kt: float,
        gust_kt: float | None = None,
        variable_direction: bool = False,
    ) -> SDR:
        direction_sdr = self.direction_encoder.encode(
            direction_degrees,
            variable_direction,
        )
        speed_sdr = self.speed_encoder.encode(speed_kt)
        gust_sdr = self.gust_encoder.encode(gust_kt)
        direction_offset = self.padding

        speed_offset = direction_offset + self.direction_size + self.padding
        gust_offset = speed_offset + self.speed_size + self.padding

        final_sparse: list[int] = []

        final_sparse.extend([idx + direction_offset for idx in direction_sdr.sparse])
        final_sparse.extend([idx + speed_offset for idx in speed_sdr.sparse])
        final_sparse.extend([idx + gust_offset for idx in gust_sdr.sparse])

        final_sdr = SDR(self.output_dimensions)
        final_sdr.sparse = sorted(final_sparse)

        return final_sdr

    def encode_dense(
        self,
        direction_degrees: float,
        speed_kt: float,
        gust_kt: float | None = None,
        variable_direction: bool = False,
    ):
        sdr = self.encode(
            direction_degrees,
            speed_kt,
            gust_kt,
            variable_direction,
        )

        return sdr.dense.copy()
