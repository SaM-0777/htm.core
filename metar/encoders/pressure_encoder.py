from dataclasses import dataclass

from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class PressureEncoderConfig:
    """
    Atmospheric pressure encoder configuration.

    Pressure is encoded as a continuous scalar manifold:
        nearby pressures -> high SDR overlap
        distant pressures -> low SDR overlap

    Designed for:
    - METAR atmospheric state encoding
    - HTM Spatial Pooler
    - HTM Temporal Memory
    - weather regime learning
    """

    minimum_hpa: float = 850.0
    maximum_hpa: float = 1100.0

    active_bits: int = 21
    sdr_size: int = 512

    resolution_hpa: float = 0.5

    clip_input: bool = True


class PressureEncoder:
    """
    Production-grade atmospheric pressure encoder.

    Uses HTM.core ScalarEncoder to preserve:
    - local continuity
    - semantic overlap
    - smooth topology
    - deterministic SDR geometry
    """

    def __init__(self, config: PressureEncoderConfig | None = None):
        self.config = config or PressureEncoderConfig()

        params = ScalarEncoderParameters()

        params.minimum = self.config.minimum_hpa
        params.maximum = self.config.maximum_hpa

        params.activeBits = self.config.active_bits
        params.size = self.config.sdr_size

        # params.radius = self.config.resolution_hpa * self.config.active_bits
        #params.resolution = self.config.resolution_hpa

        params.periodic = False

        self.encoder = ScalarEncoder(params)

        self.output_sdr = SDR(self.encoder.dimensions)

    @property
    def output_dimensions(self) -> tuple[int]:
        return tuple(self.encoder.dimensions)

    @property
    def output_size(self) -> int:
        return self.output_sdr.size

    def encode(self, pressure_hpa: float) -> SDR:
        """
        Encode atmospheric pressure into SDR.

        Args:
            pressure_hpa:
                Atmospheric pressure in hPa.

        Returns:
            HTM SDR object.
        """

        value = pressure_hpa

        if self.config.clip_input:
            value = max(self.config.minimum_hpa, value)
            value = min(self.config.maximum_hpa, value)

        self.encoder.encode(value, self.output_sdr)

        return self.output_sdr

    def encode_dense(self, pressure_hpa: float):
        """
        Encode pressure and return dense binary numpy vector.
        """

        sdr = self.encode(pressure_hpa)

        return sdr.dense

    def overlap(
        self,
        pressure_a: float,
        pressure_b: float,
    ) -> float:
        """
        Compute normalized SDR overlap between two pressures.
        """

        sdr_a = SDR(self.output_dimensions)
        sdr_b = SDR(self.output_dimensions)

        self.encoder.encode(pressure_a, sdr_a)
        self.encoder.encode(pressure_b, sdr_b)

        overlap = len(set(sdr_a.sparse) & set(sdr_b.sparse))

        return overlap / self.config.active_bits
