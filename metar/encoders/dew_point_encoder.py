from dataclasses import dataclass

from htm.bindings.encoders import (
    ScalarEncoder,
)

from htm.bindings.encoders import (
    ScalarEncoderParameters,
)

from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class DewPointEncoderConfig:
    """
    Atmospheric dew point encoder configuration.

    Dew point encodes atmospheric moisture state.

    Nearby dew points:
        -> high SDR overlap

    Distant dew points:
        -> low SDR overlap

    Designed for:
    - humidity regime learning
    - fog prediction
    - saturation dynamics
    - atmospheric moisture topology
    """

    minimum_celsius: float = -80.0
    maximum_celsius: float = 50.0

    active_bits: int = 21
    sdr_size: int = 512

    resolution_celsius: float = 0.25

    periodic: bool = False

    clip_input: bool = True


class DewPointEncoder:
    """
    Production-grade atmospheric dew point encoder.

    Preserves:
    - moisture continuity
    - semantic overlap geometry
    - smooth atmospheric topology
    - deterministic SDR structure

    Built directly on top of HTM.core ScalarEncoder.
    """

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

        # params.resolution = self.config.resolution_celsius

        params.periodic = self.config.periodic

        self.encoder = ScalarEncoder(params)

        self.output_sdr = SDR(self.encoder.dimensions)

        print()
        print("=" * 80)
        print("DEW POINT ENCODER")
        print("=" * 80)
        print()

        print(self.encoder.parameters)

        print()

        print(f"SDR Size: {self.encoder.size}")

        print(f"Dimensions: {self.encoder.dimensions}")

        print()

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
