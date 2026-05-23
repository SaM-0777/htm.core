from dataclasses import dataclass
from typing import Any, List, Dict

from htm.bindings.sdr import SDR

from encoders.cloud_layer_encoder import CloudLayerEncoder


@dataclass(frozen=True)
class CloudEncoderConfig:
    max_layers: int = 3
    layer_size: int = 512


class CloudEncoder:
    """
    Top-level Cloud Encoder.
    Takes a list of cloud layer dictionaries and returns a single fixed-size SDR.
    """

    def __init__(self, config: CloudEncoderConfig | None = None):
        self.config = config or CloudEncoderConfig()
        self.layer_enc = CloudLayerEncoder()  # 512 bits per layer
        self.total_size = self.config.max_layers * self.config.layer_size  # 1536 bits

        self.output_sdr = SDR(self.total_size)

        print(
            f"CloudEncoder ready → Max {self.config.max_layers} layers, "
            f"Total size = {self.total_size} bits"
        )

    def encode(self, layers: List[Dict[str, Any]]) -> SDR:
        """
        Main method.

        Expected input format:
        [
            {"coverage": "SCT", "height_ft": 4800.0, "type": None},
            {"coverage": "BKN", "height_ft": 12000.0, "type": "CI"},
            ...
        ]

        If no clouds → pass empty list [] or [{"coverage": "NCD", "height_ft": None, "type": None}]
        """
        # Clear the output SDR
        self.output_sdr.sparse = []

        # Handle no cloud / NCD case
        if not layers or (
            len(layers) == 1
            and layers[0].get("coverage") in ["NCD", "CLR", "SKC", None]
        ):
            # All layers are clear → entire 1536 bits remain zero (clean "no cloud" representation)
            return self.output_sdr

        # Sort layers by height (lowest first) - important for consistency
        sorted_layers = sorted(layers, key=lambda x: x.get("height_ft") or 999999)

        # Take only up to max_layers
        for i, layer in enumerate(sorted_layers[: self.config.max_layers]):
            coverage = layer.get("coverage")
            height_ft = layer.get("height_ft")
            cloud_type = layer.get("type")

            # Encode single layer (512 bits)
            layer_sdr = self.layer_enc.encode(coverage, height_ft, cloud_type)

            # Place it in its region
            offset = i * self.config.layer_size
            for bit in layer_sdr.sparse:
                self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]

        return self.output_sdr

    def overlap(self, layers_a: List[Dict], layers_b: List[Dict]) -> float:
        """Utility to compare two sets of layers"""
        sdr_a = self.encode(layers_a)
        sdr_b = self.encode(layers_b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.total_size, 4)
