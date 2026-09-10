import matplotlib.pyplot as plt


# Size designed for one column of a two-column scientific report
COLUMN_WIDTH = 3.35
COLUMN_HEIGHT = 2.35


def apply_paper_style():
    """
    Common visual style for figures used in the report.
    This changes only plot appearance.
    """
    plt.rcParams.update({
        # Figure
        "figure.figsize": (COLUMN_WIDTH, COLUMN_HEIGHT),
        "figure.dpi": 120,
        "savefig.dpi": 300,

        # Fonts
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,

        # Lines
        "lines.linewidth": 1.4,
        "lines.markersize": 3.8,

        # Axes
        "axes.linewidth": 0.7,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.5,
    })


def finish_plot():
    """
    Final formatting before saving.
    """
    ax = plt.gca()

    ax.grid(
        True,
        linestyle="-",
        linewidth=0.45,
        alpha=0.25,
    )

    ax.tick_params(
        axis="both",
        width=0.6,
        length=2.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    plt.tight_layout(pad=0.4)


def save_paper_figure(path):
    """
    Save a publication-quality figure with minimal empty margins.
    """
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )