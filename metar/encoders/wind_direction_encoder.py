from dataclasses import dataclass

from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.sdr import SDR
import numpy as np


@dataclass(frozen=True)
class WindDirectionEncoderConfig:
    """
    Wind direction encoder configuration (Cyclic / Periodic).

    Uses ScalarEncoder with periodic=True to preserve circular topology.

    This guarantees:
        - 359° ≈ 0°  → HIGH overlap
        - Smooth rotational continuity
        - Angular (circular) distance
        - Clear directional regimes (N/E/S/W)
    """

    sdr_size: int = 512
    active_bits: int = 21

    minimum: float = 0.0
    maximum: float = 360.0

    resolution: float = 3.0  # ~3° per bucket (smooth transitions)
    periodic: bool = True  # ← Most important setting

    clip_input: bool = True


class WindDirectionEncoder:
    """
    Production-grade cyclic wind direction encoder using periodic ScalarEncoder.

    This is the recommended approach in HTM for angles and directions.
    Much cleaner and more native than sin/cos projection.
    """

    def __init__(
        self,
        config: WindDirectionEncoderConfig | None = None,
    ):
        self.config = config or WindDirectionEncoderConfig()

        params = ScalarEncoderParameters()

        params.minimum = self.config.minimum
        params.maximum = self.config.maximum
        params.size = self.config.sdr_size
        params.activeBits = self.config.active_bits
        #params.resolution = self.config.resolution
        params.periodic = self.config.periodic  # ← Enables circular manifold

        self.encoder = ScalarEncoder(params)
        self.output_sdr = SDR(self.encoder.size)

        ## Debug information
        #print()
        #print("=" * 80)
        #print("WIND DIRECTION ENCODER (PERIODIC)")
        #print("=" * 80)
        #print(self.encoder.parameters)
        #print(f"SDR Size     : {self.encoder.size}")
        #print(f"Active Bits  : {self.config.active_bits}")
        #print(f"Resolution   : {self.config.resolution}° per bucket")
        #print(f"Periodic     : {self.config.periodic}")
        #print()

    @property
    def output_size(self) -> int:
        return self.encoder.size

    @property
    def output_dimensions(self) -> tuple[int]:
        return (self.encoder.size,)

    def encode(self, direction_degrees: float) -> SDR:
        """
        Encode wind direction in degrees (0-360).
        """
        value = float(direction_degrees)

        if self.config.clip_input:
            value = value % 360.0

        self.encoder.encode(value, self.output_sdr)
        return self.output_sdr

    def encode_dense(self, direction_degrees: float) -> np.ndarray:
        """Return dense numpy array (for validation suite compatibility)."""
        sdr = self.encode(direction_degrees)
        return sdr.dense.astype(np.float32)

    def overlap(self, direction_a: float, direction_b: float) -> float:
        """Normalized overlap between two wind directions."""
        sdr_a = SDR(self.output_size)
        sdr_b = SDR(self.output_size)

        self.encoder.encode(direction_a % 360.0, sdr_a)
        self.encoder.encode(direction_b % 360.0, sdr_b)

        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return float(intersection / self.config.active_bits)
