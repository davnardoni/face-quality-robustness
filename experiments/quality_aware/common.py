import numpy as np
import pandas as pd

from src.degradations import gaussian_blur

from src.matching import (
    extract_verification_scores,
)

from src.evaluation import (
    compute_far_frr,
    compute_eer,
)


BLUR_LEVELS = [
    0.0,
    0.5,
    1.0,
    2.0,
    4.0,
]


REJECTION_RATES = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


def create_mixed_blur_probes(
    probe_rgb,
    probe_labels,
):
    """
    Generate probe samples at different
    Gaussian blur levels.
    """

    images = []
    labels = []
    levels = []

    for sigma in BLUR_LEVELS:

        degraded_images = np.asarray(
            [
                gaussian_blur(
                    image,
                    sigma,
                )
                for image in probe_rgb
            ]
        )

        images.append(
            degraded_images
        )

        labels.append(
            probe_labels.copy()
        )

        levels.append(
            np.full(
                len(probe_labels),
                sigma,
                dtype=np.float32,
            )
        )

    return (
        np.concatenate(
            images,
            axis=0,
        ),
        np.concatenate(
            labels,
            axis=0,
        ),
        np.concatenate(
            levels,
            axis=0,
        ),
    )


def compute_rejection_curve(
    distance_matrix,
    probe_labels,
    gallery_labels,
    quality_scores,
):
    """
    Progressively reject the lowest-quality
    probe samples and recompute the EER.
    """

    results = []

    sorted_indices = np.argsort(
        quality_scores
    )

    n_samples = len(
        quality_scores
    )

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

        keep_mask = np.ones(
            n_samples,
            dtype=bool,
        )

        keep_mask[
            rejected_indices
        ] = False

        retained_distances = (
            distance_matrix[
                keep_mask
            ]
        )

        retained_labels = (
            probe_labels[
                keep_mask
            ]
        )

        genuine_scores, impostor_scores = (
            extract_verification_scores(
                retained_distances,
                retained_labels,
                gallery_labels,
            )
        )

        thresholds, far, frr = (
            compute_far_frr(
                genuine_scores,
                impostor_scores,
            )
        )

        eer, threshold, _ = compute_eer(
            thresholds,
            far,
            frr,
        )

        results.append(
            {
                "rejection_rate": (
                    n_reject
                    / n_samples
                ),
                "retained_samples": (
                    np.sum(keep_mask)
                ),
                "eer": eer,
                "threshold": threshold,
            }
        )

    return pd.DataFrame(
        results
    )