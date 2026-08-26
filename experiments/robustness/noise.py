import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from src.degradations import (
    gaussian_noise,
)

from experiments.robustness.common import (
    prepare_robustness_experiment,
    evaluate_degraded_probes,
)


NOISE_LEVELS = [
    0.0,
    0.02,
    0.05,
    0.10,
    0.20,
]


def run_noise_experiment():
    """
    Evaluate robustness to Gaussian noise.
    """

    context = prepare_robustness_experiment()

    probe_rgb = context[
        "probe_rgb"
    ]

    results = []

    for sigma in NOISE_LEVELS:

        print()
        print(
            f"=== Gaussian noise sigma={sigma} ==="
        )

        # Same random sequence for reproducibility.
        rng = np.random.default_rng(42)

        degraded_probe_rgb = np.asarray(
            [
                gaussian_noise(
                    image,
                    sigma,
                    rng=rng,
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
                    "degradation": "noise",
                    "level": sigma,
                    "eer": eer,
                    "threshold": threshold,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    save_noise_results(
        results_df
    )

    return results_df


def save_noise_results(results_df):
    """
    Save Gaussian noise results and plot EER.
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
        / "noise_robustness.csv",
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
        "Gaussian noise sigma"
    )

    plt.ylabel(
        "EER (%)"
    )

    plt.title(
        "Robustness to Gaussian Noise"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "noise_robustness.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":

    run_noise_experiment()