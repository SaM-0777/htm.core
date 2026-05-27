import numpy as np
import matplotlib.pyplot as plt

from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler

from metar.encoders.temperature_encoder import TemperatureEncoder

encoder = TemperatureEncoder()
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

training_temperatures = np.linspace(15, 35, 200)
for temp in training_temperatures:
    dense = encoder.encode_dense(temp)
    input_sdr = SDR((encoder.output_size,))
    input_sdr.dense = dense

    active_columns = SDR(sp.getColumnDimensions())
    sp.compute(input_sdr, True, active_columns)

base_temperature = 24

original_dense = encoder.encode_dense(base_temperature)
original_sdr = SDR((encoder.output_size,))
original_sdr.dense = original_dense

noisy_dense = original_dense.copy()
noise_bits = np.random.choice(len(noisy_dense), size=6, replace=False)
for bit in noise_bits:
    noisy_dense[bit] = 1 - noisy_dense[bit]

noisy_sdr = SDR((encoder.output_size,))
noisy_sdr.dense = noisy_dense

original_active = SDR(sp.getColumnDimensions())
sp.compute(original_sdr, False, original_active)
original_columns = set(original_active.sparse)
noisy_active = SDR(sp.getColumnDimensions())
sp.compute(noisy_sdr, False, noisy_active)
noisy_columns = set(noisy_active.sparse)

shared_columns = original_columns.intersection(noisy_columns)
similarity = len(shared_columns) / len(original_columns)
num_columns = sp.getNumColumns()
grid_size = int(np.ceil(np.sqrt(num_columns)))
original_grid = np.zeros((grid_size, grid_size))
noisy_grid = np.zeros((grid_size, grid_size))
shared_grid = np.zeros((grid_size, grid_size))


for idx in original_columns:
    row = idx // grid_size
    col = idx % grid_size

    original_grid[row, col] = 1


for idx in noisy_columns:
    row = idx // grid_size
    col = idx % grid_size

    noisy_grid[row, col] = 1


for idx in shared_columns:
    row = idx // grid_size
    col = idx % grid_size

    shared_grid[row, col] = 1


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(original_grid, cmap="inferno", interpolation="nearest")
axes[0].set_title("Original Input SDR", fontsize=16)
axes[0].set_xticks([])
axes[0].set_yticks([])


axes[1].imshow(noisy_grid, cmap="inferno", interpolation="nearest")
axes[1].set_title("Noisy Input SDR", fontsize=16)
axes[1].set_xticks([])
axes[1].set_yticks([])


axes[2].imshow(shared_grid, cmap="inferno", interpolation="nearest")
axes[2].set_title(f"Shared Active Columns\nSimilarity = {similarity:.2f}", fontsize=16)
axes[2].set_xticks([])
axes[2].set_yticks([])


plt.suptitle("Spatial Pooler Noise Robustness", fontsize=20)
plt.tight_layout()
plt.show()
