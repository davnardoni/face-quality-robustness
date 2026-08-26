import pandas as pd
import matplotlib.pyplot as plt

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from src.quality import (
    compute_sharpness_scores,
)

from src.matching import (
    euclidean_distance_matrix,
    chi_square_distance_matrix,
)

from experiments.robustness.common import (
    prepare_robustness_experiment,
    preprocess_classical_batch,
)

from experiments.quality_aware.common import (
    create_mixed_blur_probes,
    compute_rejection_curve,
)

from experiments.quality_aware.diagnostics import (
    analyze_quality_by_blur,
    analyze_rejected_blur_levels,
)


def run_quality_aware_blur_experiment():
    """
    Evaluate sharpness-based quality rejection
    using a mixed Gaussian-blur probe set.
    """

    context = (
        prepare_robustness_experiment()
    )

    probe_rgb = context[
        "probe_rgb"
    ]

    probe_labels = context[
        "probe_labels"
    ]

    gallery_labels = context[
        "gallery_labels"
    ]

    print()
    print(
        "Creating mixed-quality probe set..."
    )

    (
        mixed_probe_rgb,
        mixed_probe_labels,
        blur_levels,
    ) = create_mixed_blur_probes(
        probe_rgb,
        probe_labels,
    )

    print(
        f"Mixed probe samples: "
        f"{len(mixed_probe_rgb)}"
    )

    # -------------------------------------------------
    # QUALITY ESTIMATION
    # -------------------------------------------------

    print(
        "Computing sharpness quality scores..."
    )

    quality_scores = (
        compute_sharpness_scores(
            mixed_probe_rgb
        )
    )

    analyze_quality_by_blur(
        quality_scores,
        blur_levels,
    )

    # Diagnostic:
    # Which blur levels are actually rejected?
    analyze_rejected_blur_levels(
        quality_scores,
        blur_levels,
    )

    # -------------------------------------------------
    # CLASSICAL PREPROCESSING
    # -------------------------------------------------

    mixed_probe_classical = (
        preprocess_classical_batch(
            mixed_probe_rgb
        )
    )

    all_results = []

    # -------------------------------------------------
    # EIGENFACES
    # -------------------------------------------------

    print()
    print(
        "Computing Eigenfaces embeddings..."
    )

    eigenfaces_embeddings = context[
        "eigenfaces"
    ].transform(
        mixed_probe_classical
    )

    eigenfaces_distances = (
        euclidean_distance_matrix(
            eigenfaces_embeddings,
            context[
                "gallery_eigenfaces"
            ],
        )
    )

    eigenfaces_results = (
        compute_rejection_curve(
            eigenfaces_distances,
            mixed_probe_labels,
            gallery_labels,
            quality_scores,
        )
    )

    eigenfaces_results[
        "method"
    ] = "Eigenfaces"

    all_results.append(
        eigenfaces_results
    )

    # -------------------------------------------------
    # LBPH
    # -------------------------------------------------

    print(
        "Computing LBPH embeddings..."
    )

    lbph_embeddings = context[
        "lbph"
    ].transform(
        mixed_probe_classical
    )

    lbph_distances = (
        chi_square_distance_matrix(
            lbph_embeddings,
            context[
                "gallery_lbph"
            ],
        )
    )

    lbph_results = (
        compute_rejection_curve(
            lbph_distances,
            mixed_probe_labels,
            gallery_labels,
            quality_scores,
        )
    )

    lbph_results[
        "method"
    ] = "LBPH"

    all_results.append(
        lbph_results
    )

    # -------------------------------------------------
    # FACENET
    # -------------------------------------------------

    print(
        "Computing FaceNet embeddings..."
    )

    facenet_embeddings = context[
        "facenet"
    ].transform(
        mixed_probe_rgb
    )

    facenet_distances = (
        euclidean_distance_matrix(
            facenet_embeddings,
            context[
                "gallery_facenet"
            ],
        )
    )

    facenet_results = (
        compute_rejection_curve(
            facenet_distances,
            mixed_probe_labels,
            gallery_labels,
            quality_scores,
        )
    )

    facenet_results[
        "method"
    ] = "FaceNet"

    all_results.append(
        facenet_results
    )

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    save_quality_aware_results(
        results_df
    )

    return results_df


def save_quality_aware_results(
    results_df,
):
    """
    Save quality-aware results and generate
    EER versus rejection-rate plot.
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
        / "quality_aware_blur.csv",
        index=False,
    )

    print()
    print(
        "=== QUALITY-AWARE REJECTION ==="
    )

    print(
        results_df[
            [
                "method",
                "rejection_rate",
                "retained_samples",
                "eer",
            ]
        ].to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    for method in results_df[
        "method"
    ].unique():

        method_df = results_df[
            results_df["method"]
            == method
        ]

        plt.plot(
            method_df[
                "rejection_rate"
            ] * 100,
            method_df[
                "eer"
            ] * 100,
            marker="o",
            label=method,
        )

    plt.xlabel(
        "Rejected probe samples (%)"
    )

    plt.ylabel(
        "EER (%)"
    )

    plt.title(
        "Sharpness-Based Quality-Aware Rejection"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "quality_aware_blur.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":

    run_quality_aware_blur_experiment()