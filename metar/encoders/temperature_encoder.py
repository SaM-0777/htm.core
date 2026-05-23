# from htm.bindings.encoders import (
#    ScalarEncoder,
#    ScalarEncoderParameters,
# )

# from htm.bindings.sdr import SDR


# class TemperatureEncoder:

#    def __init__(self) -> None:

#        params = ScalarEncoderParameters()

#        params.size = 128
#        params.activeBits = 8

#        params.minimum = -30.0
#        params.maximum = 50.0

#        params.clipInput = True

#        self.encoder = ScalarEncoder(params)

#        self.output_width = params.size

#    def encode(self, temperature: float) -> SDR:

#        return self.encoder.encode(float(temperature))


from dataclasses import dataclass

from htm.bindings.encoders import (
    ScalarEncoder,
)

from htm.bindings.encoders import (
    ScalarEncoderParameters,
)

from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class TemperatureEncoderConfig:
    """
    Atmospheric temperature encoder configuration.

    Temperature is encoded as a continuous thermal manifold:
        nearby temperatures -> high SDR overlap
        distant temperatures -> low SDR overlap

    Designed for:
    - atmospheric thermodynamics
    - HTM sequence learning
    - weather regime transitions
    - thermal continuity modeling
    """

    minimum_celsius: float = -80.0
    maximum_celsius: float = 80.0

    active_bits: int = 21
    sdr_size: int = 512

    resolution_celsius: float = 0.25

    periodic: bool = False

    clip_input: bool = True


class TemperatureEncoder:
    """
    Production-grade atmospheric temperature encoder.

    Preserves:
    - thermal continuity
    - semantic overlap geometry
    - smooth SDR transitions
    - deterministic topology

    Built directly on top of HTM.core ScalarEncoder.
    """

    def __init__(
        self,
        config: TemperatureEncoderConfig | None = None,
    ):

        self.config = config or TemperatureEncoderConfig()

        params = ScalarEncoderParameters()

        params.minimum = self.config.minimum_celsius
        params.maximum = self.config.maximum_celsius

        params.activeBits = self.config.active_bits
        params.size = self.config.sdr_size
        
        #params.resolution = self.config.resolution_celsius

        params.periodic = self.config.periodic

        self.encoder = ScalarEncoder(params)

        self.output_sdr = SDR(self.encoder.dimensions)

        print()
        print("=" * 80)
        print("TEMPERATURE ENCODER")
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
        temperature_celsius: float,
    ) -> SDR:

        value = temperature_celsius

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
        temperature_celsius: float,
    ):

        sdr = self.encode(temperature_celsius)

        return sdr.dense

    def overlap(
        self,
        temperature_a: float,
        temperature_b: float,
    ) -> float:

        sdr_a = SDR(self.output_dimensions)

        sdr_b = SDR(self.output_dimensions)

        self.encoder.encode(
            temperature_a,
            sdr_a,
        )

        self.encoder.encode(
            temperature_b,
            sdr_b,
        )

        overlap = len(set(sdr_a.sparse) & set(sdr_b.sparse))

        return overlap / self.config.active_bits
