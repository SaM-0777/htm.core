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
        self.temperature_encoder = TemperatureEncoder()
        self.pressure_encoder = PressureEncoder()
        self.dew_point_encoder = DewPointEncoder()
        self.visibility_encoder = VisibilityEncoder()
        self.wind_encoder = WindEncoder()
        self.cloud_encoder = CloudEncoder()

    def encode(self, data: MetarDataInput):
        response_data = {
            "encoders": {},
        }

        try:
            if data.time_recorded is not None:
                time_sdr = self.time_encoder.encode(data.time_recorded)
                time_sdr_active_indices = [int(bit) for bit in time_sdr.sparse]

                response_data["encoders"]["recorded_time"] = {
                    "value_encoded": data.temperature_c,
                    "size": self.time_encoder.output_size,
                    "active_bits": time_sdr_active_indices,
                    "active_count": len(time_sdr_active_indices),
                }

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

            if data.visibility is not None:
                visibility_sdr = self.visibility_encoder.encode(data.visibility)
                visibility_sdr_active_indices = [
                    int(bit) for bit in visibility_sdr.sparse
                ]

                response_data["encoders"]["visibility"] = {
                    "value_encoded": data.visibility,
                    "size": self.visibility_encoder.output_size,
                    "active_bits": visibility_sdr_active_indices,
                    "active_count": len(visibility_sdr_active_indices),
                }

            if data.wind_speed_kt is not None and data.wind_direction_deg is not None:
                is_wind_variable = (
                    data.is_wind_variable if data.is_wind_variable else False
                )
                wind_sdr = self.wind_encoder.encode(
                    data.wind_direction_deg,
                    data.wind_speed_kt,
                    data.wind_gust_kt,
                    is_wind_variable,
                )
                wind_sdr_active_indices = [int(bit) for bit in wind_sdr.sparse]

                response_data["encoders"]["wind"] = {
                    "value_encoded": None,
                    "size": self.wind_encoder.output_size,
                    "active_bits": wind_sdr_active_indices,
                    "active_count": len(wind_sdr_active_indices),
                }

            if data.cloud_layers is not None and len(data.cloud_layers) > 0:
                cloud_layers = [
                    (
                        dict(layer)
                        if isinstance(layer, dict)
                        else (
                            dict(layer.dict())
                            if hasattr(layer, "dict")
                            else dict(layer.__dict__)
                        )
                    )
                    for layer in data.cloud_layers
                ]
                cloud_sdr = self.cloud_encoder.encode(cloud_layers)
                cloud_sdr_active_indices = [int(bit) for bit in cloud_sdr.sparse]

                response_data["encoders"]["clouds"] = {
                    "value_encoded": None,
                    "size": self.cloud_encoder.total_size,
                    "active_bits": cloud_sdr_active_indices,
                    "active_count": len(cloud_sdr_active_indices),
                }

            return response_data

        except Exception as e:
            logger.error(f"Visualizer Encoding Exception: {str(e)}")
            raise e
