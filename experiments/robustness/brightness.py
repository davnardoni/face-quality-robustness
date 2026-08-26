import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from src.degradations import (
    adjust_brightness,
)

from experiments.robustness.common import (
    prepare_robustness_experiment,
    evaluate_degraded_probes,
)


BRIGHTNESS_LEVELS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
]


def run_brightness_experiment():
    """
    Evaluate robustness to brightness changes.
    """

    context = prepare_robustness_experiment()

    probe_rgb = context[
        "probe_rgb"
    ]

    results = []

    for factor in BRIGHTNESS_LEVELS:

        print()
        print(
            f"=== Brightness factor={factor} ==="
        )

        degraded_probe_rgb = np.asarray(
            [
                adjust_brightness(
                    image,
                    factor,
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
                    "degradation": "brightness",
                    "level": factor,
                    "eer": eer,
                    "threshold": threshold,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    save_brightness_results(
        results_df
    )

    return results_df


def save_brightness_results(results_df):
    """
    Save brightness results and plot EER.
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
        / "brightness_robustness.csv",
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
        "Brightness factor"
    )

    plt.ylabel(
        "EER (%)"
    )

    plt.title(
        "Robustness to Brightness Changes"
    )

    # Clean/original image
    plt.axvline(
        1.0,
        linestyle="--",
        label="Original brightness",
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "brightness_robustness.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":

    run_brightness_experiment()