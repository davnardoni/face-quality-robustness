import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from src.degradations import (
    gaussian_blur,
)

from experiments.robustness.common import (
    prepare_robustness_experiment,
    evaluate_degraded_probes,
)


BLUR_LEVELS = [
    0.0,
    0.5,
    1.0,
    2.0,
    4.0,
]


def run_blur_experiment():
    """
    Evaluate robustness to Gaussian blur.
    """

    context = prepare_robustness_experiment()

    probe_rgb = context[
        "probe_rgb"
    ]

    results = []

    for sigma in BLUR_LEVELS:

        print()
        print(
            f"=== Gaussian blur sigma={sigma} ==="
        )

        degraded_probe_rgb = np.asarray(
            [
                gaussian_blur(
                    image,
                    sigma,
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
                    "degradation": "blur",
                    "level": sigma,
                    "eer": eer,
                    "threshold": threshold,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    save_blur_results(
        results_df
    )

    return results_df


def save_blur_results(results_df):
    """
    Save Gaussian blur results and plot EER.
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
        / "blur_robustness.csv",
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
        "Gaussian blur sigma"
    )

    plt.ylabel(
        "EER (%)"
    )

    plt.title(
        "Robustness to Gaussian Blur"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "blur_robustness.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":

    run_blur_experiment()