import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from src.degradations import (
    reduce_resolution,
)

from experiments.robustness.common import (
    prepare_robustness_experiment,
    evaluate_degraded_probes,
)


RESOLUTION_LEVELS = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.125,
]


def run_resolution_experiment():
    """
    Evaluate robustness to resolution reduction.

    The probe image is first downsampled according
    to the selected scale factor and then restored
    to its original dimensions.
    """

    context = prepare_robustness_experiment()

    probe_rgb = context[
        "probe_rgb"
    ]

    results = []

    for scale in RESOLUTION_LEVELS:

        print()
        print(
            f"=== Resolution scale={scale} ==="
        )

        degraded_probe_rgb = np.asarray(
            [
                reduce_resolution(
                    image,
                    scale,
                )
                for image in probe_rgb
            ]
        )

        evaluation = evaluate_degraded_probes(
            context,
            degraded_probe_rgb,
        )

        for method, metrics in evaluation.items():

            eer = metrics["eer"]
            threshold = metrics["threshold"]

            print(
                f"{method:<10} EER: "
                f"{eer * 100:.2f}%"
            )

            results.append(
                {
                    "method": method,
                    "degradation": "resolution",
                    "level": scale,
                    "eer": eer,
                    "threshold": threshold,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    save_resolution_results(
        results_df
    )

    return results_df


def save_resolution_results(results_df):
    """
    Save resolution robustness results
    and generate the EER plot.
    """

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
        / "resolution_robustness.csv",
        index=False,
    )

    plt.figure(
        figsize=(9, 6)
    )

    for method in results_df[
        "method"
    ].unique():

        method_results = results_df[
            results_df["method"] == method
        ]

        plt.plot(
            method_results["level"],
            method_results["eer"] * 100,
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Resolution scale"
    )

    plt.ylabel(
        "EER (%)"
    )

    plt.title(
        "Robustness to Resolution Reduction"
    )

    plt.axvline(
        1.0,
        linestyle="--",
        label="Original resolution",
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Resolution decreases from left to right.
    plt.gca().invert_xaxis()

    plt.savefig(
        PLOTS_DIR
        / "resolution_robustness.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":

    run_resolution_experiment()