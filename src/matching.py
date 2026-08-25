import numpy as np


def euclidean_distance_matrix(
    probe_embeddings,
    gallery_embeddings,
):
    """
    Compute all Euclidean distances between
    probe and gallery embeddings.

    Returns
    -------
    distances : ndarray
        Matrix with shape:

            n_probes x n_gallery
    """

    probe_embeddings = np.asarray(
        probe_embeddings,
        dtype=np.float32,
    )

    gallery_embeddings = np.asarray(
        gallery_embeddings,
        dtype=np.float32,
    )

    differences = (
        probe_embeddings[:, np.newaxis, :]
        - gallery_embeddings[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    return distances


def extract_verification_scores(
    distance_matrix,
    probe_labels,
    gallery_labels,
):
    """
    Separate genuine and impostor distances.

    Genuine:
        probe and gallery belong to the same subject.

    Impostor:
        probe and gallery belong to different subjects.
    """

    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)

    same_identity = (
        probe_labels[:, np.newaxis]
        == gallery_labels[np.newaxis, :]
    )

    genuine_scores = distance_matrix[
        same_identity
    ]

    impostor_scores = distance_matrix[
        ~same_identity
    ]

    return genuine_scores, impostor_scores

def chi_square_distance_matrix(
    probe_embeddings,
    gallery_embeddings,
    eps=1e-10,
):
    """
    Compute Chi-square distances between
    histogram-based descriptors using a
    memory-efficient implementation.

    Returns
    -------
    distances : ndarray
        Matrix with shape:
            n_probes x n_gallery
    """

    probe_embeddings = np.asarray(
        probe_embeddings,
        dtype=np.float32,
    )

    gallery_embeddings = np.asarray(
        gallery_embeddings,
        dtype=np.float32,
    )

    n_probes = probe_embeddings.shape[0]
    n_gallery = gallery_embeddings.shape[0]

    distances = np.empty(
        (n_probes, n_gallery),
        dtype=np.float32,
    )

    for i, probe in enumerate(probe_embeddings):

        difference = (
            gallery_embeddings - probe
        )

        denominator = (
            gallery_embeddings
            + probe
            + eps
        )

        distances[i] = 0.5 * np.sum(
            (difference ** 2)
            / denominator,
            axis=1,
        )

    return distances