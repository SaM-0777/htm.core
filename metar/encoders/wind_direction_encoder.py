from dataclasses import dataclass
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.sdr import SDR
import numpy as np


@dataclass(frozen=True)
class WindDirectionEncoderConfig:
    sdr_size: int = 512
    active_bits: int = 21
    minimum: float = 0.0
    maximum: float = 360.0
    resolution: float = 3.0
    periodic: bool = True
    clip_input: bool = True
    variable_flag_bits: int = 4


class WindDirectionEncoder:
    def __init__(
        self,
        config: WindDirectionEncoderConfig | None = None,
    ):
        self.config = config or WindDirectionEncoderConfig()
        params = ScalarEncoderParameters()
        params.minimum = self.config.minimum
        params.maximum = self.config.maximum
        params.size = self.config.sdr_size
        params.activeBits = self.config.active_bits
        params.periodic = self.config.periodic

        self.encoder = ScalarEncoder(params)
        self._output_size = self.encoder.size + self.config.variable_flag_bits
        self._output_dimensions = (self._output_size,)
        self.output_sdr = SDR(self._output_dimensions)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def output_dimensions(self) -> tuple[int]:
        return self._output_dimensions

    def encode(self, direction_degrees: float, variable: bool = False) -> SDR:
        value = float(direction_degrees)

        if self.config.clip_input:
            value = value % 360.0

        direction_sdr = SDR((self.encoder.size,))
        self.encoder.encode(
            value,
            direction_sdr,
        )

        sparse_indices = list(direction_sdr.sparse)
        flag_start = self.encoder.size

        if variable:
            sparse_indices.extend(
                range(
                    flag_start,
                    flag_start + self.config.variable_flag_bits,
                )
            )

        final_sdr = SDR(self.output_dimensions)
        final_sdr.sparse = sparse_indices
        return final_sdr

    def encode_dense(
        self,
        direction_degrees: float,
        variable: bool = False,
    ) -> np.ndarray:
        sdr = self.encode(direction_degrees, variable)
        return sdr.dense.astype(np.float32)

    def overlap(
        self,
        direction_a: float,
        direction_b: float,
        variable_a: bool = False,
        variable_b: bool = False,
    ) -> float:
        sdr_a = self.encode(
            direction_a,
            variable_a,
        )
        sdr_b = self.encode(
            direction_b,
            variable_b,
        )
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        active = max(
            len(sdr_a.sparse),
            1,
        )

        return intersection / active
