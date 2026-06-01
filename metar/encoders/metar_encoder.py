from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

from htm.bindings.sdr import SDR

from metar.encoders.time_encoder import TimeEncoder
from metar.encoders.pressure_encoder import PressureEncoder
from metar.encoders.dew_point_encoder import DewPointEncoder
from metar.encoders.visibility_encoder import VisibilityEncoder
from metar.encoders.temperature_encoder import TemperatureEncoder
from metar.encoders.wind_encoder import WindEncoder
from metar.encoders.cloud_encoder import CloudEncoder


@dataclass(frozen=True)
class MetarEncoderConfig:
    padding_bits: int = 144


class MetarEncoder:
    def __init__(self, config: MetarEncoderConfig | None = None):
        self.config = config or MetarEncoderConfig()

        self.time_enc = TimeEncoder()
        self.pressure_enc = PressureEncoder()
        self.dew_point_enc = DewPointEncoder()
        self.temperature_enc = TemperatureEncoder()
        self.visibility_enc = VisibilityEncoder()
        self.wind_enc = WindEncoder()
        self.cloud_enc = CloudEncoder()

        self.total_size = (
            self.config.padding_bits
            + self.time_enc.output_size
            + self.config.padding_bits
            + self.pressure_enc.output_size
            + self.config.padding_bits
            + self.dew_point_enc.output_size
            + self.config.padding_bits
            + self.temperature_enc.output_size
            + self.config.padding_bits
            + self.visibility_enc.output_size
            + self.config.padding_bits
            + self.wind_enc.output_size
            + self.config.padding_bits
            + self.cloud_enc.total_size
            + self.config.padding_bits
        )

        self.output_sdr = SDR(self.total_size)

    @property
    def output_size(self) -> int:
        return self.total_size

    def encode(
        self,
        recorded_time: datetime,
        pressure_hpa: float,
        dew_point_c: float,
        temperature_c: float,
        visibility_m: float,
        wind_direction_deg: float | None,
        wind_speed_kt: float,
        is_wind_variable: bool = False,
        wind_gust_kt: float | None = None,
        cloud_layers: List[Dict[str, Any]] | None = None,
    ) -> SDR:
        """
        Encode full METAR record into one unified SDR.
        recorded_time
        pressure_hpa
        dew_point_c
        temperature_c
        wind
        clouds
        """
        if cloud_layers is None:
            cloud_layers = []

        self.output_sdr.sparse = []
        offset = 0

        offset += self.config.padding_bits

        # recorded_time
        time_sdr = self.time_enc.encode(recorded_time)
        for bit in time_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]
        offset += self.time_enc.output_size + self.config.padding_bits

        if pressure_hpa is not None:
            pressure_sdr = self.pressure_enc.encode(pressure_hpa)
            for bit in pressure_sdr.sparse:
                self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]
        offset += self.pressure_enc.output_size + self.config.padding_bits

        if dew_point_c is not None:
            dew_sdr = self.dew_point_enc.encode(dew_point_c)
            for bit in dew_sdr.sparse:
                self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]
        offset += self.dew_point_enc.output_size + self.config.padding_bits

        if temperature_c is not None:
            temp_sdr = self.temperature_enc.encode(temperature_c)
            for bit in temp_sdr.sparse:
                self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]
        offset += self.temperature_enc.output_size + self.config.padding_bits

        if visibility_m is not None:
            visibility_sdr = self.visibility_enc.encode(visibility_m)
            for bit in visibility_sdr.sparse:
                self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]
        offset += self.visibility_enc.output_size + self.config.padding_bits

        wind_sdr = self.wind_enc.encode(
            direction_degrees=wind_direction_deg,
            speed_kt=wind_speed_kt,
            gust_kt=wind_gust_kt,
            variable_direction=is_wind_variable,
        )
        for bit in wind_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]
        offset += self.wind_enc.output_size + self.config.padding_bits

        cloud_sdr = self.cloud_enc.encode(cloud_layers)
        for bit in cloud_sdr.sparse:
            self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]

        return self.output_sdr
