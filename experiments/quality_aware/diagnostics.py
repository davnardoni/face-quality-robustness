import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plot_style import (
    apply_paper_style,
    finish_plot,
    save_paper_figure,
)

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from experiments.quality_aware.common import (
    REJECTION_RATES,
)

apply_paper_style()

def analyze_quality_by_blur(
    quality_scores,
    blur_levels,
):
    """
    Analyze how the sharpness quality score
    changes as Gaussian blur increases.
    """

    quality_df = pd.DataFrame(
        {
            "blur_sigma": blur_levels,
            "quality": quality_scores,
        }
    )

    statistics = (
        quality_df
        .groupby("blur_sigma")["quality"]
        .agg(["mean", "std"])
        .reset_index()
    )

    print()
    print(
        "=== SHARPNESS QUALITY BY BLUR ==="
    )

    print(
        statistics.to_string(
            index=False
        )
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics.to_csv(
        METRICS_DIR
        / "sharpness_by_blur.csv",
        index=False,
    )

    plt.figure()

    plt.errorbar(
        statistics["blur_sigma"],
        statistics["mean"],
        yerr=statistics["std"],
        color="#4C78A8",
        markersize=3.2,
        markeredgewidth=0,
        linewidth=1.1,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
    )

    plt.xlabel(r"Gaussian blur $\sigma$")
    plt.ylabel("Mean sharpness")
    plt.title("Sharpness vs Gaussian blur")

    plt.xticks(statistics["blur_sigma"])

    finish_plot()

    save_paper_figure(
        PLOTS_DIR / "sharpness_by_blur.png"
    )

    plt.show()


def analyze_rejected_blur_levels(
    quality_scores,
    blur_levels,
):
    """
    Analyze which blur levels are rejected
    as the quality rejection rate increases.
    """

    quality_scores = np.asarray(
        quality_scores
    )

    blur_levels = np.asarray(
        blur_levels
    )

    sorted_indices = np.argsort(
        quality_scores
    )

    n_samples = len(
        quality_scores
    )

    unique_blur_levels = np.unique(
        blur_levels
    )

    rows = []

    for rejection_rate in REJECTION_RATES:

        n_reject = int(
            np.floor(
                rejection_rate
                * n_samples
            )
        )

        rejected_indices = (
            sorted_indices[:n_reject]
        )

        rejected_blur_levels = (
            blur_levels[
                rejected_indices
            ]
        )

        for sigma in unique_blur_levels:

            total_at_level = np.sum(
                blur_levels == sigma
            )

            rejected_at_level = np.sum(
                rejected_blur_levels == sigma
            )

            rejected_percent = (
                100.0
                * rejected_at_level
                / total_at_level
            )

            rows.append(
                {
                    "rejection_rate": (
                        rejection_rate
                    ),
                    "blur_sigma": sigma,
                    "rejected_count": (
                        rejected_at_level
                    ),
                    "total_samples": (
                        total_at_level
                    ),
                    "rejected_percent": (
                        rejected_percent
                    ),
                }
            )

    results_df = pd.DataFrame(
        rows
    )

    print()
    print(
        "=== REJECTED SAMPLES BY BLUR LEVEL ==="
    )

    pivot = results_df.pivot(
        index="rejection_rate",
        columns="blur_sigma",
        values="rejected_percent",
    )

    print(
        pivot.round(2).to_string()
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        METRICS_DIR
        / "rejected_samples_by_blur.csv",
        index=False,
    )

    plt.figure()

    for sigma in unique_blur_levels:
        sigma_df = results_df[
            results_df["blur_sigma"] == sigma
        ]

        plt.plot(
            sigma_df["rejection_rate"] * 100,
            sigma_df["rejected_percent"],
            linestyle="-",
            linewidth=1.3,
            marker=None,
            label=rf"$\sigma={sigma:g}$",
        )

    plt.xlabel("Overall rejection (%)")
    plt.ylabel("Rejected within level (%)")
    plt.title("Rejected samples by blur level")

    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 20, 40, 60, 80, 100])

    plt.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
    )

    finish_plot()

    save_paper_figure(
        PLOTS_DIR / "rejected_samples_by_blur.png"
    )

    plt.show()

    return results_df