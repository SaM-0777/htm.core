from datetime import datetime

from htm.bindings.sdr import SDR

from model.metar import MetarRecord

from encoders.temperature_encoder import TemperatureEncoder
from encoders.time_encoder import TimeEncoder


class MetarEncoder:

    def __init__(self) -> None:

        self.temperature_encoder = TemperatureEncoder()

        self.time_encoder = TimeEncoder()

        self.output_width = (
            self.temperature_encoder.output_width + self.time_encoder.output_width
        )

    def encode(self, record: MetarRecord) -> SDR:
        dt = datetime.fromisoformat(record.recordedTime)

        temperature_sdr = self.temperature_encoder.encode(record.temperature)

        time_sdr = self.time_encoder.encode(dt)

        final_sdr = SDR(self.output_width)

        combined_sparse = []

        combined_sparse.extend(temperature_sdr.sparse)

        shifted_time_sparse = [
            i + self.temperature_encoder.output_width for i in time_sdr.sparse
        ]

        combined_sparse.extend(shifted_time_sparse)

        final_sdr.sparse = combined_sparse

        return final_sdr
