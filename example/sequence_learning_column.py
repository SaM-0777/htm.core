import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# =====================================================
# CONFIG
# =====================================================

BG = "#050505"

CELL_OFF = "#1A1A1A"

CELL_ACTIVE = "#F5EDB5"

CELL_PREDICT = "#F4C95D"

TEXT = "#EAEAEA"

ARROW = "#555555"

SAVE_PATH = "tm_sequence_learning_rcells.gif"

# =====================================================
# TEMPORAL SEQUENCE
# =====================================================

columns = [12, 28, 7, 41]

cells_per_column = 6

# -----------------------------------------------------
# Learned cell-to-cell transitions
# -----------------------------------------------------

transition_cells = {
    (12, 28): (4, 1),
    (28, 7): (1, 5),
    (7, 41): (5, 2),
}

# -----------------------------------------------------
# Animation frames
# -----------------------------------------------------

frames = [
    {"active": (12, 4), "predict": (28, 1)},
    {"active": (28, 1), "predict": (7, 5)},
    {"active": (7, 5), "predict": (41, 2)},
    {"active": (41, 2), "predict": None},
]

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)

# =====================================================
# DRAW COLUMN
# =====================================================


def draw_column(x, column_id, active_cell=None, predictive_cell=None):

    ax.text(
        x,
        7.2,
        f"Column {column_id}",
        ha="center",
        color=TEXT,
        fontsize=16,
        fontweight="bold",
    )

    for cell in range(cells_per_column):

        y = 5 - cell

        color = CELL_OFF

        if active_cell == cell:
            color = CELL_ACTIVE

        elif predictive_cell == cell:
            color = CELL_PREDICT

        circle = Circle((x, y), 0.25, facecolor=color, edgecolor=color, linewidth=2)

        ax.add_patch(circle)


# =====================================================
# FRAME UPDATE
# =====================================================


def update(frame_idx):

    ax.clear()

    ax.set_facecolor(BG)

    state = frames[frame_idx]

    active_col, active_cell = state["active"]

    pred = state["predict"]

    xs = [0, 3, 6, 9]

    x_map = {12: xs[0], 28: xs[1], 7: xs[2], 41: xs[3]}

    # ---------------------------------------------
    # Draw columns
    # ---------------------------------------------

    for col, x in zip(columns, xs):

        active = None
        predict = None

        if col == active_col:
            active = active_cell

        if pred is not None:

            pred_col, pred_cell = pred

            if col == pred_col:
                predict = pred_cell

        draw_column(x, col, active_cell=active, predictive_cell=predict)

    # ---------------------------------------------
    # Draw learned synaptic path
    # ---------------------------------------------

    for (src_col, dst_col), (src_cell, dst_cell) in transition_cells.items():

        x1 = x_map[src_col]
        y1 = 5 - src_cell

        x2 = x_map[dst_col]
        y2 = 5 - dst_cell

        ax.plot([x1, x2], [y1, y2], color="#303030", linewidth=2, alpha=0.6, zorder=0)

    # ---------------------------------------------
    # Highlight active transition
    # ---------------------------------------------

    if pred is not None:

        pred_col, pred_cell = pred

        x1 = x_map[active_col]
        y1 = 5 - active_cell

        x2 = x_map[pred_col]
        y2 = 5 - pred_cell

        ax.plot([x1, x2], [y1, y2], color=CELL_PREDICT, linewidth=4, alpha=1.0)

    # ---------------------------------------------
    # Titles
    # ---------------------------------------------

    ax.text(
        4.5,
        8.8,
        "Temporal Memory Sequence Learning",
        ha="center",
        color="white",
        fontsize=30,
        fontweight="bold",
    )

    ax.text(
        4.5,
        8.0,
        "Predictions emerge through learned cell-to-cell transitions",
        ha="center",
        color="#A0A0A0",
        fontsize=15,
    )

    # ---------------------------------------------
    # Legend
    # ---------------------------------------------

    ax.scatter([-0.5], [-1.5], s=250, c=CELL_ACTIVE)

    ax.text(-0.1, -1.5, "Active Cell", color=TEXT, va="center", fontsize=12)

    ax.scatter([2.2], [-1.5], s=250, c=CELL_PREDICT)

    ax.text(2.6, -1.5, "Predictive Cell", color=TEXT, va="center", fontsize=12)

    ax.set_xlim(-1, 10)

    ax.set_ylim(-2, 10)

    ax.axis("off")

    return ax.artists


# =====================================================
# ANIMATION
# =====================================================

anim = FuncAnimation(fig, update, frames=len(frames), interval=1400, repeat=True)

anim.save(SAVE_PATH, writer="pillow", fps=1)

plt.close()

print(f"Saved: {SAVE_PATH}")
