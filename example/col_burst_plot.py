import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# ------------------------------------
# Styling
# ------------------------------------

BG = "#050505"
INACTIVE = "#181818"
PREDICTED = "#f4d35e"
ACTIVE = "#fff6bf"
TEXT = "#d0d0d0"

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], hspace=0.25, wspace=0.15)

# ------------------------------------
# Input SDR
# ------------------------------------


def draw_input(ax, active_bits):

    ax.set_facecolor(BG)

    cols = 32
    rows = 4

    for i in range(cols):
        x = i

        color = ACTIVE if i in active_bits else INACTIVE

        ax.add_patch(Rectangle((x, 0), 0.85, 0.85, color=color, linewidth=0))

    ax.set_xlim(-0.5, cols)
    ax.set_ylim(-0.5, 1.5)

    ax.axis("off")


# ------------------------------------
# Cell Column
# ------------------------------------


def draw_column(ax, bursting=False, predicted_cell=8):

    ax.set_facecolor(BG)

    cells_per_col = 16

    ys = np.arange(cells_per_col)

    for i in range(cells_per_col):

        if bursting:
            color = ACTIVE
            size = 350

        else:
            if i == predicted_cell:
                color = PREDICTED
                size = 450
            else:
                color = INACTIVE
                size = 220

        ax.scatter(0, cells_per_col - i, s=size, color=color, edgecolors="none")

    ax.axis("off")

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, cells_per_col + 2)


# ------------------------------------
# Left Input
# ------------------------------------

ax0 = fig.add_subplot(gs[0, 0])

draw_input(ax0, active_bits=[12, 13, 14, 15, 16, 17, 18, 19])

ax0.set_title("Predicted Input Pattern", color="white", fontsize=18)

# ------------------------------------
# Right Input
# ------------------------------------

ax1 = fig.add_subplot(gs[0, 1])

draw_input(ax1, active_bits=[5, 6, 7, 8, 9, 10, 11, 12])

ax1.set_title("Unexpected Input Pattern", color="white", fontsize=18)

# ------------------------------------
# Prediction Success
# ------------------------------------

ax2 = fig.add_subplot(gs[1, 0])

draw_column(ax2, bursting=False, predicted_cell=7)

ax2.set_title(
    "Prediction Success\nOnly Predicted Cell Activates", color="white", fontsize=20
)

# ------------------------------------
# Burst
# ------------------------------------

ax3 = fig.add_subplot(gs[1, 1])

draw_column(ax3, bursting=True)

ax3.set_title("Column Bursting\nAll Cells Become Active", color="white", fontsize=20)

fig.suptitle(
    "Temporal Memory Predicted Activation vs Column Bursting",
    color="white",
    fontsize=26,
    y=0.97,
)

plt.savefig("tm_column_bursting.png", dpi=300, facecolor=BG, bbox_inches="tight")

plt.show()
