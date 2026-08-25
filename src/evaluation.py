import numpy as np
import matplotlib.pyplot as plt

from config import PLOTS_DIR


def compute_far_frr(
    genuine_scores,
    impostor_scores,
    thresholds=None,
    num_thresholds=1000,
):
    """
    Compute FAR and FRR for distance-based biometric scores.

    Acceptance rule:
        distance <= threshold

    Parameters
    ----------
    genuine_scores : array-like
        Distances obtained from same-identity comparisons.

    impostor_scores : array-like
        Distances obtained from different-identity comparisons.

    thresholds : array-like or None
        Thresholds to evaluate.

    num_thresholds : int
        Number of automatically generated thresholds.

    Returns
    -------
    thresholds : ndarray
    far : ndarray
    frr : ndarray
    """

    genuine_scores = np.asarray(
        genuine_scores,
        dtype=np.float64,
    )

    impostor_scores = np.asarray(
        impostor_scores,
        dtype=np.float64,
    )

    if thresholds is None:

        minimum = min(
            genuine_scores.min(),
            impostor_scores.min(),
        )

        maximum = max(
            genuine_scores.max(),
            impostor_scores.max(),
        )

        thresholds = np.linspace(
            minimum,
            maximum,
            num_thresholds,
        )

    far = np.empty(len(thresholds))
    frr = np.empty(len(thresholds))

    for i, threshold in enumerate(thresholds):

        # Impostor accepted incorrectly
        far[i] = np.mean(
            impostor_scores <= threshold
        )

        # Genuine rejected incorrectly
        frr[i] = np.mean(
            genuine_scores > threshold
        )

    return thresholds, far, frr


def compute_eer(
    thresholds,
    far,
    frr,
):
    """
    Estimate the Equal Error Rate.

    The selected operating point is the threshold
    where |FAR - FRR| is minimum.
    """

    index = np.argmin(
        np.abs(far - frr)
    )

    eer = (
        far[index] + frr[index]
    ) / 2.0

    eer_threshold = thresholds[index]

    return (
        eer,
        eer_threshold,
        index,
    )


def print_evaluation_results(
    genuine_scores,
    impostor_scores,
    eer,
    eer_threshold,
):
    """
    Print summary biometric evaluation information.
    """

    print()
    print("=== BIOMETRIC EVALUATION ===")

    print(
        f"Genuine comparisons: "
        f"{len(genuine_scores)}"
    )

    print(
        f"Impostor comparisons: "
        f"{len(impostor_scores)}"
    )

    print()

    print(
        f"Mean genuine distance: "
        f"{np.mean(genuine_scores):.4f}"
    )

    print(
        f"Std genuine distance:  "
        f"{np.std(genuine_scores):.4f}"
    )

    print(
        f"Mean impostor distance: "
        f"{np.mean(impostor_scores):.4f}"
    )

    print(
        f"Std impostor distance:  "
        f"{np.std(impostor_scores):.4f}"
    )

    print()

    print(
        f"EER:                  "
        f"{eer:.4f} "
        f"({eer * 100:.2f}%)"
    )

    print(
        f"EER threshold:        "
        f"{eer_threshold:.4f}"
    )


def plot_score_distributions(
    genuine_scores,
    impostor_scores,
    eer_threshold=None,
    save=True,
):
    """
    Plot genuine and impostor distance distributions.
    """

    plt.figure(figsize=(9, 6))

    plt.hist(
        genuine_scores,
        bins=50,
        density=True,
        alpha=0.6,
        label="Genuine",
    )

    plt.hist(
        impostor_scores,
        bins=50,
        density=True,
        alpha=0.6,
        label="Impostor",
    )

    if eer_threshold is not None:

        plt.axvline(
            eer_threshold,
            linestyle="--",
            label="EER threshold",
        )

    plt.xlabel("Euclidean distance")
    plt.ylabel("Density")

    plt.title(
        "Genuine and Impostor Score Distributions"
    )

    plt.legend()
    plt.tight_layout()

    if save:

        PLOTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            PLOTS_DIR
            / "eigenfaces_score_distributions.png",
            dpi=200,
        )

    plt.show()


def plot_far_frr(
    thresholds,
    far,
    frr,
    eer_threshold=None,
    save=True,
):
    """
    Plot FAR and FRR as functions of the threshold.
    """

    plt.figure(figsize=(9, 6))

    plt.plot(
        thresholds,
        far,
        label="FAR",
    )

    plt.plot(
        thresholds,
        frr,
        label="FRR",
    )

    if eer_threshold is not None:

        plt.axvline(
            eer_threshold,
            linestyle="--",
            label="EER threshold",
        )

    plt.xlabel("Threshold")
    plt.ylabel("Error rate")

    plt.title("FAR / FRR")

    plt.legend()
    plt.tight_layout()

    if save:

        PLOTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            PLOTS_DIR
            / "eigenfaces_far_frr.png",
            dpi=200,
        )

    plt.show()


def plot_roc(
    far,
    frr,
    save=True,
):
    """
    Plot biometric ROC:

        x = FAR
        y = GAR = 1 - FRR
    """

    gar = 1.0 - frr

    plt.figure(figsize=(7, 7))

    plt.plot(
        far,
        gar,
    )

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("Genuine Acceptance Rate (GAR)")

    plt.title("ROC Curve")

    plt.grid(True)
    plt.tight_layout()

    if save:

        PLOTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            PLOTS_DIR
            / "eigenfaces_roc.png",
            dpi=200,
        )

    plt.show()