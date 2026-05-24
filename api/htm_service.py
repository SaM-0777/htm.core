import logging
from api.schemas import MetarDataInput

from metar.encoders.time_encoder import TimeEncoder
from metar.encoders.pressure_encoder import PressureEncoder
from metar.encoders.temperature_encoder import TemperatureEncoder
from metar.encoders.dew_point_encoder import DewPointEncoder
from metar.encoders.visibility_encoder import VisibilityEncoder
from metar.encoders.wind_encoder import WindEncoder
from metar.encoders.cloud_encoder import CloudEncoder

logger = logging.getLogger(__name__)


class HTMOrchestrator:
    def __init__(self):
        self.time_encoder = TimeEncoder()
        self.pressure_encoder = PressureEncoder()
        self.dew_point_encoder = DewPointEncoder()
        self.temperature_encoder = TemperatureEncoder()

    def encode(self, data: MetarDataInput):
        response_data = {
            "encoders": {},
        }

        try:
            if data.temperature_c is not None:
                temperature_sdr = self.temperature_encoder.encode(data.temperature_c)
                temperature_sdr_active_indices = [
                    int(bit) for bit in temperature_sdr.sparse
                ]

                response_data["encoders"]["temperature"] = {
                    "value_encoded": data.temperature_c,
                    "size": self.temperature_encoder.output_size,
                    "active_bits": temperature_sdr_active_indices,
                    "active_count": len(temperature_sdr_active_indices),
                }

            if data.pressure_hpa is not None:
                pressure_sdr = self.pressure_encoder.encode(data.pressure_hpa)
                pressure_sdr_active_indices = [int(bit) for bit in pressure_sdr.sparse]

                response_data["encoders"]["pressure"] = {
                    "value_encoded": data.pressure_hpa,
                    "size": self.pressure_encoder.output_size,
                    "active_bits": pressure_sdr_active_indices,
                    "active_count": len(pressure_sdr_active_indices),
                }

            if data.dew_point_c is not None:
                dew_point_sdr = self.dew_point_encoder.encode(data.dew_point_c)
                dew_point_sdr_active_indices = [
                    int(bit) for bit in dew_point_sdr.sparse
                ]

                response_data["encoders"]["dew_point"] = {
                    "value_encoded": data.dew_point_c,
                    "size": self.dew_point_encoder.output_size,
                    "active_bits": dew_point_sdr_active_indices,
                    "active_count": len(dew_point_sdr_active_indices),
                }

            return response_data

        except Exception as e:
            logger.error(f"Visualizer Encoding Exception: {str(e)}")
            raise e
