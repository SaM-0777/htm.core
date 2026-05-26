from datetime import datetime
from dataclasses import dataclass
from htm.bindings.encoders import (
    DateEncoder,
    DateEncoderParameters,
)
from htm.bindings.sdr import SDR
import numpy as np


@dataclass(frozen=True)
class TimeEncoderConfig:
    timeOfDay_width: int = 96
    timeOfDay_radius: int = 6


class TimeEncoder:
    def __init__(self, config: TimeEncoderConfig | None = None) -> None:
        self.config = config or TimeEncoderConfig()
        params = DateEncoderParameters()
        params.timeOfDay_width = self.config.timeOfDay_width
        params.timeOfDay_radius = self.config.timeOfDay_radius

        self.encoder = DateEncoder(params)
        self.output_sdr = SDR(self.encoder.dimensions)

    def encode(self, dt: datetime) -> SDR:
        self.encoder.encode(dt, self.output_sdr)
        return self.output_sdr

    @property
    def output_size(self) -> int:
        return self.encoder.dimensions[0]

    @property
    def output_dimensions(self) -> tuple[int]:
        return tuple(self.encoder.dimensions)

    def encode_dense(self, dt: datetime) -> np.ndarray:
        sdr = self.encode(dt)
        return sdr.dense.astype(np.float32)

    def overlap(self, dt_a: datetime, dt_b: datetime) -> float:
        sdr_a = self.encode(dt_a)
        sdr_b = self.encode(dt_b)

        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        active = max(len(sdr_a.sparse), 1)

        return intersection / active
