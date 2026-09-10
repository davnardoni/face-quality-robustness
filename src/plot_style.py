import matplotlib.pyplot as plt


# Target size: one column in a two-column report
COLUMN_WIDTH = 3.35
COLUMN_HEIGHT = 2.25


# Restrained palette for scientific figures
COLORS = {
    "Eigenfaces": "#4C78A8",
    "LBPH": "#F58518",
    "FaceNet": "#54A24B",
}


def apply_paper_style():
    plt.rcParams.update({
        # Figure
        "figure.figsize": (COLUMN_WIDTH, COLUMN_HEIGHT),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.facecolor": "white",

        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.65,
        "axes.titlesize": 8.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 7.0,
        "axes.labelpad": 3,

        # Typography
        "font.family": "sans-serif",
        "font.size": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Lines
        "lines.linewidth": 1.3,

        # Grid
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.45,

        # Legend
        "legend.fontsize": 5.8,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CCCCCC",
        "legend.framealpha": 0.95,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
    })


def plot_method_curve(x, y, method):
    """
    Plot a method using a simple continuous line.
    No markers are used, so the legend contains only colored lines.
    """
    plt.plot(
        x,
        y,
        color=COLORS[method],
        linestyle="-",
        linewidth=1.3,
        label=method,
    )


def finish_plot():
    """
    Apply the final visual formatting.
    """
    ax = plt.gca()

    # Keep the grid behind the curves
    ax.set_axisbelow(True)

    # Light scientific grid
    ax.grid(
        True,
        linestyle="-",
        linewidth=0.45,
        alpha=0.45,
    )

    # Keep all four borders, as in the example report
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#444444")

    ax.tick_params(
        axis="both",
        direction="out",
        width=0.6,
        length=2.5,
    )

    plt.tight_layout(pad=0.35)


def save_paper_figure(path):
    """
    Save at high resolution with minimal surrounding whitespace.
    """
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )