import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    METRICS_DIR,
    PLOTS_DIR,
)

from src.dataset import load_lfw_dataset

from src.preprocessing import (
    load_split,
    get_split_images,
    get_preprocessed_images,
    preprocess_classical,
)

from src.degradations import (
    gaussian_blur,
    gaussian_noise,
    adjust_brightness,
)

from src.eigenfaces import EigenfacesRecognizer
from src.lbph import LBPHRecognizer
from src.deep_recognizer import FaceNetRecognizer

from src.matching import (
    euclidean_distance_matrix,
    chi_square_distance_matrix,
    extract_verification_scores,
)

from src.evaluation import (
    compute_far_frr,
    compute_eer,
)

def evaluate_embeddings(
    probe_embeddings,
    gallery_embeddings,
    probe_labels,
    gallery_labels,
    distance_type,
):
    """
    Evaluate a set of probe and gallery embeddings
    and return the EER.
    """

    if distance_type == "euclidean":

        distance_matrix = euclidean_distance_matrix(
            probe_embeddings,
            gallery_embeddings,
        )

    elif distance_type == "chi_square":

        distance_matrix = chi_square_distance_matrix(
            probe_embeddings,
            gallery_embeddings,
        )

    else:
        raise ValueError(
            f"Unknown distance type: {distance_type}"
        )

    genuine_scores, impostor_scores = (
        extract_verification_scores(
            distance_matrix,
            probe_labels,
            gallery_labels,
        )
    )

    thresholds, far, frr = compute_far_frr(
        genuine_scores,
        impostor_scores,
    )

    eer, eer_threshold, _ = compute_eer(
        thresholds,
        far,
        frr,
    )

    return eer, eer_threshold

def preprocess_classical_batch(images):
    """
    Apply the classical preprocessing pipeline
    to a collection of RGB images.
    """

    return np.asarray(
        [
            preprocess_classical(image)
            for image in images
        ]
    )

def run_blur_experiment():
    """
    Evaluate the robustness of Eigenfaces,
    LBPH and FaceNet to Gaussian blur.
    """

    dataset = load_lfw_dataset()
    split = load_split()

    # -------------------------------------------------
    # TRAINING DATA FOR EIGENFACES
    # -------------------------------------------------

    train_images, _ = get_preprocessed_images(
        dataset,
        split,
        "train",
    )

    # -------------------------------------------------
    # CLEAN GALLERY
    # -------------------------------------------------

    gallery_classical, gallery_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "gallery",
        )
    )

    gallery_rgb, _ = get_split_images(
        dataset,
        split,
        "gallery",
    )

    # -------------------------------------------------
    # ORIGINAL PROBES
    # -------------------------------------------------

    probe_rgb, probe_labels = get_split_images(
        dataset,
        split,
        "probe",
    )

    # -------------------------------------------------
    # INITIALIZE RECOGNIZERS
    # -------------------------------------------------

    print("Initializing recognizers...")

    eigenfaces = EigenfacesRecognizer()
    eigenfaces.fit(train_images)

    lbph = LBPHRecognizer()

    facenet = FaceNetRecognizer()

    # -------------------------------------------------
    # CLEAN GALLERY EMBEDDINGS
    # -------------------------------------------------

    print("Computing clean gallery embeddings...")

    gallery_eigenfaces = eigenfaces.transform(
        gallery_classical
    )

    gallery_lbph = lbph.transform(
        gallery_classical
    )

    gallery_facenet = facenet.transform(
        gallery_rgb
    )

    # Blur severity
    blur_levels = [
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
    ]

    results = []

    # -------------------------------------------------
    # BLUR EXPERIMENT
    # -------------------------------------------------

    for sigma in blur_levels:

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

        degraded_probe_classical = (
            preprocess_classical_batch(
                degraded_probe_rgb
            )
        )

        # ---------------------------------------------
        # Eigenfaces
        # ---------------------------------------------

        probe_eigenfaces = eigenfaces.transform(
            degraded_probe_classical
        )

        eer, threshold = evaluate_embeddings(
            probe_eigenfaces,
            gallery_eigenfaces,
            probe_labels,
            gallery_labels,
            distance_type="euclidean",
        )

        print(
            f"Eigenfaces EER: "
            f"{eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "Eigenfaces",
                "degradation": "blur",
                "level": sigma,
                "eer": eer,
                "threshold": threshold,
            }
        )

        # ---------------------------------------------
        # LBPH
        # ---------------------------------------------

        probe_lbph = lbph.transform(
            degraded_probe_classical
        )

        eer, threshold = evaluate_embeddings(
            probe_lbph,
            gallery_lbph,
            probe_labels,
            gallery_labels,
            distance_type="chi_square",
        )

        print(
            f"LBPH EER:       "
            f"{eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "LBPH",
                "degradation": "blur",
                "level": sigma,
                "eer": eer,
                "threshold": threshold,
            }
        )

        # ---------------------------------------------
        # FaceNet
        # ---------------------------------------------

        probe_facenet = facenet.transform(
            degraded_probe_rgb
        )

        eer, threshold = evaluate_embeddings(
            probe_facenet,
            gallery_facenet,
            probe_labels,
            gallery_labels,
            distance_type="euclidean",
        )

        print(
            f"FaceNet EER:    "
            f"{eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "FaceNet",
                "degradation": "blur",
                "level": sigma,
                "eer": eer,
                "threshold": threshold,
            }
        )

    results_df = pd.DataFrame(results)

    save_blur_results(results_df)

    return results_df


def run_noise_experiment():
    """
    Evaluate the robustness of Eigenfaces,
    LBPH and FaceNet to Gaussian noise.
    """

    dataset = load_lfw_dataset()
    split = load_split()

    # Training data for Eigenfaces
    train_images, _ = get_preprocessed_images(
        dataset,
        split,
        "train",
    )

    # Clean gallery
    gallery_classical, gallery_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "gallery",
        )
    )

    gallery_rgb, _ = get_split_images(
        dataset,
        split,
        "gallery",
    )

    # Original probes
    probe_rgb, probe_labels = get_split_images(
        dataset,
        split,
        "probe",
    )

    print("Initializing recognizers...")

    eigenfaces = EigenfacesRecognizer()
    eigenfaces.fit(train_images)

    lbph = LBPHRecognizer()

    facenet = FaceNetRecognizer()

    print("Computing clean gallery embeddings...")

    gallery_eigenfaces = eigenfaces.transform(
        gallery_classical
    )

    gallery_lbph = lbph.transform(
        gallery_classical
    )

    gallery_facenet = facenet.transform(
        gallery_rgb
    )

    noise_levels = [
        0.0,
        0.02,
        0.05,
        0.10,
        0.20,
    ]

    results = []

    for sigma in noise_levels:

        print()
        print(
            f"=== Gaussian noise sigma={sigma} ==="
        )

        # Fixed seed -> reproducible experiment
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

        degraded_probe_classical = (
            preprocess_classical_batch(
                degraded_probe_rgb
            )
        )

        # Eigenfaces
        probe_eigenfaces = eigenfaces.transform(
            degraded_probe_classical
        )

        eer, threshold = evaluate_embeddings(
            probe_eigenfaces,
            gallery_eigenfaces,
            probe_labels,
            gallery_labels,
            distance_type="euclidean",
        )

        print(
            f"Eigenfaces EER: {eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "Eigenfaces",
                "degradation": "noise",
                "level": sigma,
                "eer": eer,
                "threshold": threshold,
            }
        )

        # LBPH
        probe_lbph = lbph.transform(
            degraded_probe_classical
        )

        eer, threshold = evaluate_embeddings(
            probe_lbph,
            gallery_lbph,
            probe_labels,
            gallery_labels,
            distance_type="chi_square",
        )

        print(
            f"LBPH EER:       {eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "LBPH",
                "degradation": "noise",
                "level": sigma,
                "eer": eer,
                "threshold": threshold,
            }
        )

        # FaceNet
        probe_facenet = facenet.transform(
            degraded_probe_rgb
        )

        eer, threshold = evaluate_embeddings(
            probe_facenet,
            gallery_facenet,
            probe_labels,
            gallery_labels,
            distance_type="euclidean",
        )

        print(
            f"FaceNet EER:    {eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "FaceNet",
                "degradation": "noise",
                "level": sigma,
                "eer": eer,
                "threshold": threshold,
            }
        )

    results_df = pd.DataFrame(results)

    save_noise_results(results_df)

    return results_df


def run_brightness_experiment():
    """
    Evaluate the robustness of Eigenfaces,
    LBPH and FaceNet to brightness changes.
    """

    dataset = load_lfw_dataset()
    split = load_split()

    # Training data for Eigenfaces
    train_images, _ = get_preprocessed_images(
        dataset,
        split,
        "train",
    )

    # Clean gallery
    gallery_classical, gallery_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "gallery",
        )
    )

    gallery_rgb, _ = get_split_images(
        dataset,
        split,
        "gallery",
    )

    # Original probes
    probe_rgb, probe_labels = get_split_images(
        dataset,
        split,
        "probe",
    )

    print("Initializing recognizers...")

    eigenfaces = EigenfacesRecognizer()
    eigenfaces.fit(train_images)

    lbph = LBPHRecognizer()

    facenet = FaceNetRecognizer()

    print("Computing clean gallery embeddings...")

    gallery_eigenfaces = eigenfaces.transform(
        gallery_classical
    )

    gallery_lbph = lbph.transform(
        gallery_classical
    )

    gallery_facenet = facenet.transform(
        gallery_rgb
    )

    brightness_levels = [
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ]

    results = []

    for factor in brightness_levels:

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

        degraded_probe_classical = (
            preprocess_classical_batch(
                degraded_probe_rgb
            )
        )

        # Eigenfaces
        probe_eigenfaces = eigenfaces.transform(
            degraded_probe_classical
        )

        eer, threshold = evaluate_embeddings(
            probe_eigenfaces,
            gallery_eigenfaces,
            probe_labels,
            gallery_labels,
            distance_type="euclidean",
        )

        print(
            f"Eigenfaces EER: {eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "Eigenfaces",
                "degradation": "brightness",
                "level": factor,
                "eer": eer,
                "threshold": threshold,
            }
        )

        # LBPH
        probe_lbph = lbph.transform(
            degraded_probe_classical
        )

        eer, threshold = evaluate_embeddings(
            probe_lbph,
            gallery_lbph,
            probe_labels,
            gallery_labels,
            distance_type="chi_square",
        )

        print(
            f"LBPH EER:       {eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "LBPH",
                "degradation": "brightness",
                "level": factor,
                "eer": eer,
                "threshold": threshold,
            }
        )

        # FaceNet
        probe_facenet = facenet.transform(
            degraded_probe_rgb
        )

        eer, threshold = evaluate_embeddings(
            probe_facenet,
            gallery_facenet,
            probe_labels,
            gallery_labels,
            distance_type="euclidean",
        )

        print(
            f"FaceNet EER:    {eer * 100:.2f}%"
        )

        results.append(
            {
                "method": "FaceNet",
                "degradation": "brightness",
                "level": factor,
                "eer": eer,
                "threshold": threshold,
            }
        )

    results_df = pd.DataFrame(results)

    save_brightness_results(
        results_df
    )

    return results_df


def save_brightness_results(results_df):
    """
    Save brightness robustness results
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
        / "brightness_robustness.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    for method in results_df["method"].unique():

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

    # Original brightness
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


def save_blur_results(results_df):
    """
    Save Gaussian blur robustness results
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
        / "blur_robustness.csv",
        index=False,
    )

    plt.figure(
        figsize=(9, 6)
    )

    for method in results_df["method"].unique():

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


def save_noise_results(results_df):
    """
    Save Gaussian noise robustness results
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
        / "noise_robustness.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    for method in results_df["method"].unique():

        method_results = results_df[
            results_df["method"] == method
        ]

        plt.plot(
            method_results["level"],
            method_results["eer"] * 100,
            marker="o",
            label=method,
        )

    plt.xlabel("Gaussian noise sigma")
    plt.ylabel("EER (%)")

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

    #run_blur_experiment()
    #run_noise_experiment()
    run_brightness_experiment()