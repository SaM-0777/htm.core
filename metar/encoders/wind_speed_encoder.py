# metar/encoders/wind_speed_encoder.py

from __future__ import annotations

from dataclasses import dataclass

from htm.bindings.encoders import (
    ScalarEncoder,
    ScalarEncoderParameters,
)
from htm.bindings.sdr import SDR


@dataclass(frozen=True)
class WindSpeedEncoderConfig:
    """
    Wind speed encoder configuration.

    Encodes sustained wind speed into a continuous SDR manifold
    using HTM.core ScalarEncoder.

    Semantic goals:
    ----------------
    - Nearby wind speeds produce high overlap
    - Distant wind speeds produce low overlap
    - Smooth atmospheric transitions preserve SDR continuity
    - No cyclic wraparound

    Meteorological goals:
    ---------------------
    Learn:
    - pressure gradient flow
    - frontal dynamics
    - storm strengthening
    - circulation evolution
    - temporal wind persistence
    """

    # Realistic global atmospheric range
    minimum_kt: float = 0.0
    maximum_kt: float = 150.0

    # SDR topology
    active_bits: int = 21

    # ONLY specify size OR resolution/radius/category
    # Never combine size with radius/resolution.
    sdr_size: int = 512

    # Approximate semantic resolution
    # Nearby speeds within ~1 kt maintain strong overlap
    resolution_kt: float = 1.0

    # Safety
    clip_input: bool = True


class WindSpeedEncoder:
    """
    Production-grade sustained wind speed encoder.

    Uses HTM.core ScalarEncoder.

    Properties:
    -----------
    - deterministic
    - continuous
    - topology-preserving
    - HTM-native SDR geometry
    - smooth overlap manifold
    """

    def __init__(
        self,
        config: WindSpeedEncoderConfig | None = None,
    ) -> None:

        self.config = config or WindSpeedEncoderConfig()

        params = ScalarEncoderParameters()

        params.minimum = self.config.minimum_kt
        params.maximum = self.config.maximum_kt

        params.activeBits = self.config.active_bits

        # IMPORTANT:
        # Use ONLY size.
        # HTM.core internally derives radius/resolution.
        params.size = self.config.sdr_size

        # Wind speed is NOT cyclic
        params.periodic = False

        self.encoder = ScalarEncoder(params)

        self.output_sdr = SDR(self.encoder.dimensions)

    @property
    def output_dimensions(self) -> tuple[int]:
        return tuple(self.encoder.dimensions)

    @property
    def output_size(self) -> int:
        return self.output_sdr.size

    @property
    def active_bits(self) -> int:
        return self.config.active_bits

    def encode(
        self,
        speed_kt: float,
    ) -> SDR:
        """
        Encode sustained wind speed into SDR.

        Parameters
        ----------
        speed_kt:
            Sustained wind speed in knots.

        Returns
        -------
        SDR
            HTM SDR representation.
        """

        value = float(speed_kt)

        if self.config.clip_input:
            value = max(self.config.minimum_kt, value)
            value = min(self.config.maximum_kt, value)

        self.encoder.encode(value, self.output_sdr)

        return self.output_sdr

    def encode_dense(
        self,
        speed_kt: float,
    ):
        """
        Encode wind speed into dense binary vector.

        Returns
        -------
        numpy.ndarray
        """

        sdr = self.encode(speed_kt)

        return sdr.dense.copy()

    def encode_sparse(
        self,
        speed_kt: float,
    ) -> list[int]:
        """
        Encode wind speed into sparse active indices.

        Returns
        -------
        list[int]
        """

        sdr = self.encode(speed_kt)

        return list(sdr.sparse)

    def overlap(
        self,
        speed_a: float,
        speed_b: float,
    ) -> float:
        """
        Compute normalized SDR overlap between two wind speeds.

        Returns
        -------
        float
            Range:
                0.0 -> no overlap
                1.0 -> identical SDRs
        """

        sdr_a = SDR(self.output_dimensions)
        sdr_b = SDR(self.output_dimensions)

        self.encoder.encode(float(speed_a), sdr_a)
        self.encoder.encode(float(speed_b), sdr_b)

        overlap = len(set(sdr_a.sparse) & set(sdr_b.sparse))

        return overlap / self.config.active_bits

    def similarity(
        self,
        speed_a: float,
        speed_b: float,
    ) -> float:
        """
        Alias for overlap() for semantic clarity.
        """

        return self.overlap(speed_a, speed_b)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"range=[{self.config.minimum_kt}, "
            f"{self.config.maximum_kt}], "
            f"size={self.output_size}, "
            f"active_bits={self.active_bits})"
        )
