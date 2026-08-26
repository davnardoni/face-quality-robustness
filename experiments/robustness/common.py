import numpy as np

from src.dataset import load_lfw_dataset

from src.preprocessing import (
    load_split,
    get_split_images,
    get_preprocessed_images,
    preprocess_classical,
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
    Evaluate probe and gallery embeddings
    and return EER and EER threshold.
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
    Apply classical preprocessing to a batch
    of RGB face images.
    """

    return np.asarray(
        [
            preprocess_classical(image)
            for image in images
        ]
    )


def prepare_robustness_experiment():
    """
    Prepare data, recognizers and clean gallery
    embeddings for a robustness experiment.

    The gallery always remains clean.

    Only probe images will be degraded.
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

    eigenfaces.fit(
        train_images
    )

    lbph = LBPHRecognizer()

    facenet = FaceNetRecognizer()

    # -------------------------------------------------
    # CLEAN GALLERY EMBEDDINGS
    # -------------------------------------------------

    print(
        "Computing clean gallery embeddings..."
    )

    gallery_eigenfaces = eigenfaces.transform(
        gallery_classical
    )

    gallery_lbph = lbph.transform(
        gallery_classical
    )

    gallery_facenet = facenet.transform(
        gallery_rgb
    )

    return {
        "probe_rgb": probe_rgb,
        "probe_labels": probe_labels,
        "gallery_labels": gallery_labels,

        "eigenfaces": eigenfaces,
        "lbph": lbph,
        "facenet": facenet,

        "gallery_eigenfaces": gallery_eigenfaces,
        "gallery_lbph": gallery_lbph,
        "gallery_facenet": gallery_facenet,
    }


def evaluate_degraded_probes(
    context,
    degraded_probe_rgb,
):
    """
    Evaluate the three recognizers using the
    same degraded probe images.

    Returns
    -------
    dict
        EER and threshold for each recognizer.
    """

    degraded_probe_classical = (
        preprocess_classical_batch(
            degraded_probe_rgb
        )
    )

    probe_labels = context[
        "probe_labels"
    ]

    gallery_labels = context[
        "gallery_labels"
    ]

    results = {}

    # -------------------------------------------------
    # EIGENFACES
    # -------------------------------------------------

    probe_eigenfaces = context[
        "eigenfaces"
    ].transform(
        degraded_probe_classical
    )

    eer, threshold = evaluate_embeddings(
        probe_eigenfaces,
        context["gallery_eigenfaces"],
        probe_labels,
        gallery_labels,
        distance_type="euclidean",
    )

    results["Eigenfaces"] = {
        "eer": eer,
        "threshold": threshold,
    }

    # -------------------------------------------------
    # LBPH
    # -------------------------------------------------

    probe_lbph = context[
        "lbph"
    ].transform(
        degraded_probe_classical
    )

    eer, threshold = evaluate_embeddings(
        probe_lbph,
        context["gallery_lbph"],
        probe_labels,
        gallery_labels,
        distance_type="chi_square",
    )

    results["LBPH"] = {
        "eer": eer,
        "threshold": threshold,
    }

    # -------------------------------------------------
    # FACENET
    # -------------------------------------------------

    probe_facenet = context[
        "facenet"
    ].transform(
        degraded_probe_rgb
    )

    eer, threshold = evaluate_embeddings(
        probe_facenet,
        context["gallery_facenet"],
        probe_labels,
        gallery_labels,
        distance_type="euclidean",
    )

    results["FaceNet"] = {
        "eer": eer,
        "threshold": threshold,
    }

    return results