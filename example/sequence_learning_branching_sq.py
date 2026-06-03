import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch

# =====================================================
# THEME
# =====================================================

BG = "#050505"

CARD = "#101114"

ACTIVE = "#F5EDB5"

PREDICT = "#F4C95D"

TEXT = "#ECECEC"

SUBTEXT = "#8A8A8A"

ARROW = "#343434"

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)

# =====================================================
# POSITIONS
# =====================================================

positions = {
    "A": (0, 0),
    "B": (4, 0),
    "C": (8, 2),
    "D": (8, 0),
    "E": (8, -2),
}

# =====================================================
# STATES
# =====================================================

frames = [
    {"active": ["A"], "predict": []},
    {"active": ["A"], "predict": ["B"]},
    {"active": ["B"], "predict": []},
    {"active": ["B"], "predict": ["C", "D", "E"]},
    {"active": ["C"], "predict": []},
]

# =====================================================
# DRAW NODE
# =====================================================


def draw_node(label, x, y, active=False, predictive=False):

    color = CARD

    if predictive:
        color = PREDICT

    if active:
        color = ACTIVE

    box = FancyBboxPatch(
        (x - 0.8, y - 0.6),
        1.6,
        1.2,
        boxstyle="round,pad=0.15,rounding_size=0.15",
        facecolor=color,
        edgecolor="none",
    )

    ax.add_patch(box)

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="black" if (active or predictive) else TEXT,
    )


# =====================================================
# DRAW CONNECTION
# =====================================================


def draw_connection(a, b, highlight=False):

    x1, y1 = positions[a]
    x2, y2 = positions[b]

    color = ARROW
    lw = 3

    if highlight:
        color = PREDICT
        lw = 5

    ax.arrow(
        x1 + 0.9,
        y1,
        x2 - x1 - 1.8,
        y2 - y1,
        width=0.03,
        length_includes_head=True,
        head_width=0.18,
        head_length=0.18,
        color=color,
    )


# =====================================================
# FRAME UPDATE
# =====================================================


def update(frame_idx):

    ax.clear()

    ax.set_facecolor(BG)

    state = frames[frame_idx]

    active = state["active"]

    predictive = state["predict"]

    # -------------------------------------------------
    # Static connections
    # -------------------------------------------------

    draw_connection("A", "B", highlight=("B" in predictive))

    draw_connection("B", "C", highlight=("C" in predictive))

    draw_connection("B", "D", highlight=("D" in predictive))

    draw_connection("B", "E", highlight=("E" in predictive))

    # -------------------------------------------------
    # Nodes
    # -------------------------------------------------

    for node, (x, y) in positions.items():

        draw_node(node, x, y, active=node in active, predictive=node in predictive)

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    ax.text(
        4,
        4,
        "Multiple Predictions and Branching Futures",
        ha="center",
        color=TEXT,
        fontsize=30,
        fontweight="bold",
    )

    ax.text(
        4,
        3.3,
        "A learned context can predict several valid futures simultaneously.",
        ha="center",
        color=SUBTEXT,
        fontsize=16,
    )

    # -------------------------------------------------

    ax.set_xlim(-2, 10)

    ax.set_ylim(-4, 5)

    ax.axis("off")
    
    return ax.artists


# =====================================================
# ANIMATION
# =====================================================

anim = FuncAnimation(fig, update, frames=len(frames), interval=1800, repeat=True)

anim.save("tm_branching_futures_letters.gif", writer="pillow", fps=1)

plt.close()

print("Saved: tm_branching_futures_letters.gif")
