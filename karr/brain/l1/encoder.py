from htm.bindings.sdr import SDR
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters


# Encoder for characters/symbols (expanded range for more symbols)
class SymbolEncoder:
    def __init__(self):
        self.params = ScalarEncoderParameters()
        self.params.activeBits = 25  # Increased for more symbols
        self.params.size = 1200  # Larger for distinction
        self.params.minimum = 32  # Space
        self.params.maximum = 126  # ~ (covers ASCII printable)
        self.params.periodic = False
        self.params.clipInput = True

        self.encoder = ScalarEncoder(self.params)
        self.width = self.params.size

    def encode(self, char):
        output_sdr = SDR(self.width)
        val = max(32, min(ord(char), 126))
        self.encoder.encode(val, output_sdr)
        return output_sdr
