import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch

# ============================================================
# STYLE
# ============================================================

BG = "#050505"

INACTIVE = "#151515"

ACTIVE = "#FFF6BF"

PREDICTED = "#F4D35E"

TEXT = "#E0E0E0"

plt.style.use("dark_background")

# ============================================================
# PATTERNS
# ============================================================

patterns = ["A", "B", "C", "D"]

# Sequence evolution
frames_data = [
    {"active": ["A"], "predictive": []},
    {"active": ["A", "B"], "predictive": []},
    {"active": ["A", "B"], "predictive": ["C"]},
    {"active": ["A", "B", "C"], "predictive": []},
    {"active": ["B", "C"], "predictive": ["D"]},
    {"active": ["B", "C", "D"], "predictive": []},
]

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)

ax.set_facecolor(BG)

# ============================================================
# DRAW
# ============================================================


def draw(frame):

    ax.clear()

    ax.set_facecolor(BG)

    state = frames_data[frame]

    active = state["active"]

    predictive = state["predictive"]

    for i, pattern in enumerate(patterns):

        x = i * 2.5

        color = INACTIVE

        if pattern in predictive:
            color = PREDICTED

        if pattern in active:
            color = ACTIVE

        rect = FancyBboxPatch(
            (x, 0),
            1.8,
            1.8,
            boxstyle="round,pad=0.15",
            linewidth=0,
            facecolor=color,
        )

        ax.add_patch(rect)

        ax.text(
            x + 0.9,
            0.9,
            pattern,
            ha="center",
            va="center",
            fontsize=28,
            color="black" if color != INACTIVE else TEXT,
            fontweight="bold",
        )

    # arrows

    for i in range(len(patterns) - 1):

        ax.arrow(
            i * 2.5 + 1.9,
            0.9,
            0.45,
            0,
            width=0.02,
            head_width=0.15,
            head_length=0.15,
            color="#444444",
            length_includes_head=True,
        )

    ax.text(
        0,
        2.8,
        "Temporal Memory Sequence Learning",
        fontsize=22,
        color="white",
        weight="bold",
    )

    ax.text(
        0,
        2.2,
        "Active patterns illuminate while future patterns enter a predictive state.",
        fontsize=12,
        color="#AAAAAA",
    )

    ax.set_xlim(-0.5, 9)

    ax.set_ylim(-0.3, 3.5)

    ax.axis("off")

    return ax.artists


# ============================================================
# ANIMATION
# ============================================================

anim = FuncAnimation(
    fig,
    draw,
    frames=len(frames_data),
    interval=1200,
    repeat=True,
)

# ============================================================
# SAVE
# ============================================================

anim.save(
    "tm_sequence_learning.gif",
    writer="pillow",
    fps=1,
)

plt.close()
