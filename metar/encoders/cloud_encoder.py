from dataclasses import dataclass
from typing import Any, List, Dict
from htm.bindings.sdr import SDR
from metar.encoders.cloud_layer_encoder import CloudLayerEncoder


@dataclass(frozen=True)
class CloudEncoderConfig:
    max_layers: int = 3
    layer_size: int = 512


class CloudEncoder:
    def __init__(self, config: CloudEncoderConfig | None = None):
        self.config = config or CloudEncoderConfig()
        self.layer_enc = CloudLayerEncoder()
        self.total_size = self.config.max_layers * self.config.layer_size
        self.output_sdr = SDR(self.total_size)

    def encode(self, layers: List[Dict[str, Any]]) -> SDR:
        self.output_sdr.sparse = []

        if not layers or (
            len(layers) == 1
            and layers[0].get("coverage") in ["NCD", "CLR", "SKC", None]
        ):
            return self.output_sdr

        sorted_layers = sorted(layers, key=lambda x: x.get("height_ft") or 999999)

        for i, layer in enumerate(sorted_layers[: self.config.max_layers]):
            coverage = layer.get("coverage")
            height_ft = layer.get("height_ft") or layer.get("altitude_ft")
            cloud_type = layer.get("type") or layer.get("cloud_type")
            layer_sdr = self.layer_enc.encode(coverage, height_ft, cloud_type)
            offset = i * self.config.layer_size
            for bit in layer_sdr.sparse:
                self.output_sdr.sparse = list(self.output_sdr.sparse) + [bit + offset]

        return self.output_sdr

    def overlap(self, layers_a: List[Dict], layers_b: List[Dict]) -> float:
        sdr_a = self.encode(layers_a)
        sdr_b = self.encode(layers_b)
        intersection = len(set(sdr_a.sparse) & set(sdr_b.sparse))
        return round(intersection / self.total_size, 4)
