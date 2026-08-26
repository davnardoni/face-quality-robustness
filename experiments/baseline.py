from src.dataset import load_lfw_dataset

from src.preprocessing import (
    load_split,
    get_preprocessed_images,
    get_split_images,
)

from src.eigenfaces import (
    EigenfacesRecognizer,
    print_pca_info,
    show_eigenfaces,
)

from src.matching import (
    euclidean_distance_matrix,
    chi_square_distance_matrix,
    extract_verification_scores,
)

from src.evaluation import (
    compute_far_frr,
    compute_eer,
    print_evaluation_results,
    plot_score_distributions,
    plot_far_frr,
    plot_roc,
)

from src.lbph import LBPHRecognizer

from src.deep_recognizer import (
    FaceNetRecognizer,
)

def run_eigenfaces_baseline():
    """
    Run the clean Eigenfaces verification baseline.
    """

    dataset = load_lfw_dataset()

    split = load_split()

    train_images, _ = get_preprocessed_images(
        dataset,
        split,
        "train",
    )

    gallery_images, gallery_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "gallery",
        )
    )

    probe_images, probe_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "probe",
        )
    )

    recognizer = EigenfacesRecognizer()

    # PCA is learned only from training images
    recognizer.fit(train_images)

    train_embeddings = recognizer.transform(
        train_images
    )

    gallery_embeddings = recognizer.transform(
        gallery_images
    )

    probe_embeddings = recognizer.transform(
        probe_images
    )

    print_pca_info(
        recognizer,
        train_images,
        train_embeddings,
    )

    distance_matrix = euclidean_distance_matrix(
        probe_embeddings,
        gallery_embeddings,
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

    print_evaluation_results(
        genuine_scores,
        impostor_scores,
        eer,
        eer_threshold,
    )

    plot_score_distributions(
        genuine_scores,
        impostor_scores,
        eer_threshold,
        method_name="Eigenfaces",
    )

    plot_far_frr(
        thresholds,
        far,
        frr,
        eer_threshold,
        method_name="Eigenfaces",
    )

    plot_roc(
        far,
        frr,
        method_name="Eigenfaces",
    )

    show_eigenfaces(
        recognizer
    )

def run_lbph_baseline():
    """
    Run the clean LBP/LBPH verification baseline.
    """

    dataset = load_lfw_dataset()

    split = load_split()

    gallery_images, gallery_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "gallery",
        )
    )

    probe_images, probe_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "probe",
        )
    )

    recognizer = LBPHRecognizer()

    gallery_embeddings = recognizer.transform(
        gallery_images
    )

    probe_embeddings = recognizer.transform(
        probe_images
    )

    print()
    print("=== LBP / LBPH ===")

    print(
        f"Gallery embeddings:     "
        f"{gallery_embeddings.shape}"
    )

    print(
        f"Probe embeddings:       "
        f"{probe_embeddings.shape}"
    )

    print(
        f"Template dimensions:    "
        f"{gallery_embeddings.shape[1]}"
    )

    distance_matrix = chi_square_distance_matrix(
        probe_embeddings,
        gallery_embeddings,
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

    print_evaluation_results(
        genuine_scores,
        impostor_scores,
        eer,
        eer_threshold,
    )

    plot_score_distributions(
        genuine_scores,
        impostor_scores,
        eer_threshold,
        method_name="LBPH",
    )

    plot_far_frr(
        thresholds,
        far,
        frr,
        eer_threshold,
        method_name="LBPH",
    )

    plot_roc(
        far,
        frr,
        method_name="LBPH",
    )

def run_deep_baseline():
    """
    Run the clean deep face-recognition baseline.
    """

    dataset = load_lfw_dataset()

    split = load_split()

    gallery_images, gallery_labels = (
        get_split_images(
            dataset,
            split,
            "gallery",
        )
    )

    probe_images, probe_labels = (
        get_split_images(
            dataset,
            split,
            "probe",
        )
    )

    recognizer = FaceNetRecognizer()

    print()
    print("=== DEEP / FACENET ===")

    print(
        f"Device:                 "
        f"{recognizer.device}"
    )

    gallery_embeddings = recognizer.transform(
        gallery_images
    )

    probe_embeddings = recognizer.transform(
        probe_images
    )

    print(
        f"Gallery embeddings:     "
        f"{gallery_embeddings.shape}"
    )

    print(
        f"Probe embeddings:       "
        f"{probe_embeddings.shape}"
    )

    print(
        f"Template dimensions:    "
        f"{gallery_embeddings.shape[1]}"
    )

    distance_matrix = euclidean_distance_matrix(
        probe_embeddings,
        gallery_embeddings,
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

    print_evaluation_results(
        genuine_scores,
        impostor_scores,
        eer,
        eer_threshold,
    )

    plot_score_distributions(
        genuine_scores,
        impostor_scores,
        eer_threshold,
        method_name="FaceNet",
    )

    plot_far_frr(
        thresholds,
        far,
        frr,
        eer_threshold,
        method_name="FaceNet",
    )

    plot_roc(
        far,
        frr,
        method_name="FaceNet",
    )

if __name__ == "__main__":
    #run_eigenfaces_baseline()
    #run_lbph_baseline()
    run_deep_baseline()