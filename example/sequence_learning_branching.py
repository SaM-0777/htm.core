import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# =====================================================
# STYLE
# =====================================================

BG = "#050505"

CELL_OFF = "#1A1A1A"

CELL_ACTIVE = "#F5EDB5"

CELL_PREDICT = "#F4C95D"

TEXT = "#EAEAEA"

CONNECTION = "#303030"

SAVE_PATH = "tm_branching_futures_4.gif"

# =====================================================
# COLUMNS
# =====================================================

cells_per_column = 6

positions = {
    12: (0, 0),
    28: (4, 0),
    41: (8, 3),
    57: (8, 0),
    83: (8, -3),
}

# learned transitions

transition_cells = {
    (12, 28): (4, 1),
    (28, 41): (1, 2),
    (28, 57): (1, 4),
    (28, 83): (1, 0),
}

# =====================================================
# FRAMES
# =====================================================

# frames = [
#    {
#        "active": (12, 4),
#        "predict": [(28, 1)],
#    },
#    {
#        "active": (28, 1),
#        "predict": [
#            (41, 2),
#            (57, 4),
#            (83, 0),
#        ],
#    },
#    {
#        "active": (41, 2),
#        "predict": [],
#    },
# ]

frames = [
    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------
    {
        "active": (12, 4),
        "predict": [
            (28, 1),
        ],
    },
    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------
    {
        "active": (28, 1),
        "predict": [
            (41, 2),
            (57, 4),
            (83, 0),
        ],
    },
    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------
    {
        "active": (41, 2),
        "predict": [],
    },
]

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)

# =====================================================
# DRAW COLUMN
# =====================================================


def draw_column(
    column_id,
    active_cell=None,
    predictive_cells=None,
):

    if predictive_cells is None:
        predictive_cells = []

    x, y_offset = positions[column_id]

    ax.text(
        x,
        y_offset + 4.3,
        f"Column {column_id}",
        ha="center",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
    )

    for cell in range(cells_per_column):

        y = y_offset + (5 - cell)

        color = CELL_OFF

        # ------------------------------------
        # Predictive cell
        # ------------------------------------

        if cell in predictive_cells:
            color = CELL_PREDICT

        # ------------------------------------
        # Active cell overrides prediction
        # ------------------------------------

        if active_cell == cell:
            color = CELL_ACTIVE

        # ------------------------------------
        # Cell
        # ------------------------------------

        circle = Circle(
            (x, y), 0.22, facecolor=color, edgecolor=color, linewidth=1.5, zorder=5
        )

        ax.add_patch(circle)

        # ------------------------------------
        # Predictive glow
        # ------------------------------------

        if cell in predictive_cells:

            glow = Circle(
                (x, y),
                0.40,
                facecolor=CELL_PREDICT,
                edgecolor="none",
                alpha=0.25,
                zorder=4,
            )

            ax.add_patch(glow)

        # ------------------------------------
        # Active glow
        # ------------------------------------

        if active_cell == cell:

            glow = Circle(
                (x, y),
                0.50,
                facecolor=CELL_ACTIVE,
                edgecolor="none",
                alpha=0.30,
                zorder=4,
            )

            ax.add_patch(glow)


# =====================================================
# FRAME UPDATE
# =====================================================


def update(frame_idx):

    ax.clear()

    ax.set_facecolor(BG)

    state = frames[frame_idx]

    active_col, active_cell = state["active"]

    predictive = state["predict"]

    # ----------------------------------------
    # Draw learned connections
    # ----------------------------------------

    for (src_col, dst_col), (src_cell, dst_cell) in transition_cells.items():

        x1, y1_base = positions[src_col]
        x2, y2_base = positions[dst_col]

        y1 = y1_base + (5 - src_cell)
        y2 = y2_base + (5 - dst_cell)

        ax.plot([x1, x2], [y1, y2], color="#2A2A2A", linewidth=2, alpha=0.6)

    # ----------------------------------------
    # Highlight current predictions
    # ----------------------------------------

    for pred_col, pred_cell in predictive:

        x1, y1_base = positions[active_col]

        y1 = y1_base + (5 - active_cell)

        x2, y2_base = positions[pred_col]

        y2 = y2_base + (5 - pred_cell)

        ax.plot([x1, x2], [y1, y2], color=CELL_PREDICT, linewidth=4, alpha=1.0)

    # ----------------------------------------
    # Draw columns
    # ----------------------------------------

    for col in positions.keys():

        active = None
        predictive_cells = []

        # Active cell
        if col == active_col:
            active = active_cell

        # Predictive cells
        for pred_col, pred_cell in predictive:
            if pred_col == col:
                predictive_cells.append(pred_cell)

        draw_column(
            column_id=col, active_cell=active, predictive_cells=predictive_cells
        )

    # ----------------------------------------
    # Title
    # ----------------------------------------

    ax.text(
        4,
        10,
        "Multiple Predictions and Branching Futures",
        ha="center",
        color="white",
        fontsize=26,
        fontweight="bold",
    )

    ax.text(
        4,
        9,
        "A single learned context can predict multiple possible future states simultaneously.",
        ha="center",
        color="#9a9a9a",
        fontsize=14,
    )

    # ----------------------------------------
    # Legend
    # ----------------------------------------

    ax.scatter([-1], [-6], s=250, c=CELL_ACTIVE)

    ax.text(-0.5, -6, "Active Cell", color=TEXT, va="center")

    ax.scatter([2], [-6], s=250, c=CELL_PREDICT)

    ax.text(2.5, -6, "Predictive Cell", color=TEXT, va="center")

    ax.set_xlim(-2, 10)

    ax.set_ylim(-7, 11)

    ax.axis("off")

    return ax.artists


# =====================================================
# GIF
# =====================================================

anim = FuncAnimation(fig, update, frames=len(frames), interval=1800, repeat=True)

anim.save(SAVE_PATH, writer="pillow", fps=1)

plt.close()

print(f"Saved -> {SAVE_PATH}")
