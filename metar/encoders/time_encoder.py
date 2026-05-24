from datetime import datetime
from htm.bindings.encoders import (
    DateEncoder,
    DateEncoderParameters,
)
from htm.bindings.sdr import SDR


class TimeEncoder:
    def __init__(self) -> None:
        params = DateEncoderParameters()
        params.timeOfDay_width = 64
        params.timeOfDay_radius = 6

        self.encoder = DateEncoder(params)
        self.output_width = self.encoder.dimensions[0]

    def encode(self, dt: datetime) -> SDR:
        return self.encoder.encode(dt)
