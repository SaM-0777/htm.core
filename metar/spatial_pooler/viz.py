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
sp.compute(input_sdr, True, active_columns)
winning_columns = set(active_columns.sparse)
active_column_indices = set(active_columns.sparse)

num_columns = sp.getNumColumns()
connected_synapses = []
overlap_scores = []

for col in range(num_columns):
    permanence = np.zeros(sp.getNumInputs(), dtype=np.float32)
    sp.getPermanence(col, permanence)

    connected = np.where(permanence >= sp.getSynPermConnected())[0]
    connected_synapses.append(connected)

    overlap = len(active_input_bits.intersection(connected))
    overlap_scores.append(overlap)

normalized_overlap = overlap_scores / np.max(overlap_scores)

fig, ax = plt.subplots(figsize=(18, 8))
input_y = 0

for i in range(len(input_dense)):
    if input_dense[i] == 1:
        color = "black"
        size = 80
    else:
        color = "lightgray"
        size = 20

    ax.scatter(i, input_y, color=color, s=size)


column_y = 12
column_positions = np.linspace(10, len(input_dense) - 10, num_columns)

for idx, x in enumerate(column_positions):
    if idx in active_column_indices:
        color = "black"
        size = 500
    else:
        color = "gray"
        size = 250

    ax.scatter(x, column_y, color=color, s=size)
    ax.text(x, column_y + 1.3, f"O={overlap_scores[idx]}", ha="center", fontsize=8)


for col_idx, synapses in enumerate(connected_synapses):
    col_x = column_positions[col_idx]

    for syn in synapses:
        if syn in active_input_bits:
            color = "black"
            linewidth = 2
            alpha = 0.9
        else:
            color = "gray"
            linewidth = 0.5
            alpha = 0.08

        ax.plot(
            [syn, col_x],
            [input_y, column_y],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


ax.text(
    len(input_dense) / 2,
    -2,
    f"Temperature Encoder SDR ({temperature}°C)",
    ha="center",
    fontsize=14,
)

ax.text(
    len(input_dense) / 2,
    column_y + 3,
    "HTM Spatial Pooler Columns",
    ha="center",
    fontsize=14,
)

ax.set_xlim(-5, len(input_dense) + 5)
ax.set_ylim(-4, 18)
ax.axis("off")

plt.title("Authentic HTM Spatial Pooler Receptive Fields", fontsize=18)
plt.tight_layout()
plt.show()


# Heatmap
grid_size = int(np.ceil(np.sqrt(num_columns)))
heatmap_before = np.zeros((grid_size, grid_size))
heatmap_after = np.zeros((grid_size, grid_size))
for idx, score in enumerate(normalized_overlap):

    row = idx // grid_size
    col = idx % grid_size

    heatmap_before[row, col] = score

    if idx in winning_columns:
        heatmap_after[row, col] = score


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(heatmap_before, cmap="inferno", interpolation="nearest")

for idx, score in enumerate(overlap_scores):
    row = idx // grid_size
    col = idx % grid_size
    ax.text(col, row, str(score), ha="center", va="center", color="white", fontsize=10)

plt.title("Spatial Pooler Overlap Score Heatmap", fontsize=18)

plt.xlabel("Spatial Pooler Columns")
plt.ylabel("Spatial Pooler Columns")

cbar = plt.colorbar(im)
cbar.set_label("Normalized Overlap Strength")

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

frames = [
    ("Before Inhibition", heatmap_before),
    ("After Inhibition", heatmap_after),
]


def update(frame_index):

    ax.clear()

    title, data = frames[frame_index]

    im = ax.imshow(data, cmap="inferno", interpolation="nearest", vmin=0, vmax=1)

    # Annotate overlap scores
    for idx, score in enumerate(overlap_scores):

        row = idx // grid_size
        col = idx % grid_size

        # Only annotate visible cells
        if data[row, col] > 0:

            ax.text(
                col,
                row,
                str(score),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    ax.set_title(f"Spatial Pooler {title}", fontsize=18)

    ax.set_xticks([])
    ax.set_yticks([])

    return [im]


ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=1800, blit=False, repeat=True
)
ani.save("spatial_pooler_inhibition.gif", writer="pillow", fps=1)

print("GIF saved as spatial_pooler_inhibition.gif")

plt.show()


activation_grid = np.zeros((grid_size, grid_size))
for idx in range(num_columns):

    row = idx // grid_size
    col = idx % grid_size

    if idx in active_column_indices:
        activation_grid[row, col] = 1.0

fig, ax = plt.subplots(figsize=(12, 12))

im = ax.imshow(activation_grid, cmap="inferno", interpolation="nearest", vmin=0, vmax=1)

for idx in active_column_indices:

    row = idx // grid_size
    col = idx % grid_size

    ax.text(col, row, "●", ha="center", va="center", color="white", fontsize=10)

plt.title("Spatial Pooler Active Column Map", fontsize=20)

plt.xlabel("Cortical Column Space")
plt.ylabel("Cortical Column Space")

cbar = plt.colorbar(im)
cbar.set_label("Column Activation")

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

plt.show()
