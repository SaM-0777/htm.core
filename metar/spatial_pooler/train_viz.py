import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler

from metar.encoders.temperature_encoder import TemperatureEncoder

encoder = TemperatureEncoder()
temperatures = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

sp = SpatialPooler(
    inputDimensions=(encoder.output_size,),
    columnDimensions=(64,),
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

num_columns = sp.getNumColumns()
num_inputs = sp.getNumInputs()
iterations = 40
permanence_history = []

for step in range(iterations):
    temp = temperatures[step % len(temperatures)]

    dense = encoder.encode_dense(temp)

    input_sdr = SDR((encoder.output_size,))
    input_sdr.dense = dense
    active_columns = SDR(sp.getColumnDimensions())
    sp.compute(input_sdr, True, active_columns)

    permanence_matrix = np.zeros((num_columns, num_inputs), dtype=np.float32)

    for col in range(num_columns):
        permanence = np.zeros(num_inputs, dtype=np.float32)
        sp.getPermanence(col, permanence)
        permanence_matrix[col] = permanence

    permanence_history.append(permanence_matrix.copy())


fig, ax = plt.subplots(figsize=(16, 10))


def update(frame):
    ax.clear()

    permanence_matrix = permanence_history[frame]
    im = ax.imshow(
        permanence_matrix,
        cmap="inferno",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax.set_title(f"Synaptic Permanence Evolution — Iteration {frame + 1}", fontsize=18)
    ax.set_xlabel("Encoder SDR Input Bits", fontsize=12)
    ax.set_ylabel("Spatial Pooler Columns", fontsize=12)

    return [im]


ani = animation.FuncAnimation(
    fig, update, frames=iterations, interval=300, blit=False, repeat=True
)


ani.save("synaptic_permanence_heatmap.gif", writer="pillow", fps=6)
plt.show()
