from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler
from metar.encoders.temperature_encoder import TemperatureEncoder

temperature = 24

encoder = TemperatureEncoder()

input_dense = encoder.encode_dense(temperature)
input_sdr = SDR((encoder.output_size,))
input_sdr.dense = input_dense

active_input_bits = set(input_sdr.sparse)

sp = SpatialPooler(
    inputDimensions=(encoder.output_size,),
    columnDimensions=(256,),
    potentialPct=0.5,
    potentialRadius=encoder.output_size,
    globalInhibition=True,
    localAreaDensity=0,
    numActiveColumnsPerInhArea=12,
    synPermInactiveDec=0.008,
    synPermActiveInc=0.05,
    synPermConnected=0.1,
    boostStrength=0.0,
    seed=42,
    wrapAround=False,
)


active_columns = SDR(sp.getColumnDimensions())
sp.compute(input_sdr, learn=True, output=active_columns)
winning_columns = set(active_columns.sparse)
active_column_indices = set(active_columns.sparse)

num_columns = sp.getNumColumns()
