import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from config import (
    PCA_EXPLAINED_VARIANCE,
    N_EIGENFACES_TO_SHOW,
    CLASSICAL_IMAGE_SIZE,
)

from src.matching import (
    euclidean_distance_matrix,
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

from src.dataset import load_lfw_dataset
from src.preprocessing import (
    preprocess_classical,
    load_split,
)


class EigenfacesRecognizer:
    """
    Face representation based on PCA / Eigenfaces.
    """

    def __init__(
        self,
        explained_variance=PCA_EXPLAINED_VARIANCE,
    ):
        self.variance_threshold = explained_variance

        self.pca = PCA(
            n_components=explained_variance,
            svd_solver="full",
            whiten=False, 
        )

        self.is_fitted = False

    def fit(self, images):
        """
        Fit PCA using training face images.

        Parameters
        ----------
        images : array-like
            Preprocessed grayscale face images.
        """

        X = self._flatten_images(images)

        self.pca.fit(X)

        self.is_fitted = True

    def transform(self, images):
        """
        Project face images into the Eigenfaces space.

        Returns
        -------
        embeddings : ndarray
            PCA representations of the input faces.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "The Eigenfaces recognizer must be fitted first."
            )

        X = self._flatten_images(images)

        return self.pca.transform(X)

    @staticmethod
    def _flatten_images(images):
        """
        Convert images from:

            N x H x W

        to:

            N x (H*W)
        """

        images = np.asarray(images, dtype=np.float32)

        return images.reshape(len(images), -1)

    @property
    def n_components(self):
        """
        Number of PCA components selected.
        """

        if not self.is_fitted:
            return None

        return self.pca.n_components_

    @property
    def explained_variance(self):
        """
        Total variance represented by PCA components.
        """

        if not self.is_fitted:
            return None

        return np.sum(
            self.pca.explained_variance_ratio_
        )


def get_preprocessed_images(
    lfw,
    split_df,
    split_name,
):
    """
    Retrieve and preprocess all images belonging
    to a specific experimental split.
    """

    rows = split_df[
        split_df["split"] == split_name
    ]

    images = []

    labels = []

    for _, row in rows.iterrows():

        image_index = int(row["image_index"])

        image = lfw.images[image_index]

        processed = preprocess_classical(image)

        images.append(processed)

        labels.append(int(row["label"]))

    return (
        np.asarray(images),
        np.asarray(labels),
    )


def show_eigenfaces(recognizer):
    """
    Display the PCA mean face and first eigenfaces.
    """

    width, height = CLASSICAL_IMAGE_SIZE

    mean_face = recognizer.pca.mean_.reshape(
        height,
        width,
    )

    eigenfaces = recognizer.pca.components_[
        :N_EIGENFACES_TO_SHOW
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8, 8),
    )

    axes = axes.ravel()

    axes[0].imshow(
        mean_face,
        cmap="gray",
    )

    axes[0].set_title("Mean face")

    axes[0].axis("off")

    for i, eigenface in enumerate(
        eigenfaces,
        start=1,
    ):

        eigenface_image = eigenface.reshape(
            height,
            width,
        )

        axes[i].imshow(
            eigenface_image,
            cmap="gray",
        )

        axes[i].set_title(
            f"Eigenface {i}"
        )

        axes[i].axis("off")

    plt.tight_layout()

    plt.show()


def print_pca_info(
    recognizer,
    train_images,
    train_embeddings,
):
    """
    Print PCA sanity-check information.
    """

    print("=== EIGENFACES / PCA ===")

    print(
        f"Training images:         "
        f"{len(train_images)}"
    )

    print(
        f"Original image shape:    "
        f"{train_images[0].shape}"
    )

    print(
        f"Original dimensions:     "
        f"{train_images[0].size}"
    )

    print(
        f"PCA components:          "
        f"{recognizer.n_components}"
    )

    print(
        f"Explained variance:      "
        f"{recognizer.explained_variance:.4f}"
    )

    print(
        f"Embedding shape:         "
        f"{train_embeddings.shape}"
    )


if __name__ == "__main__":

    dataset = load_lfw_dataset()

    split = load_split()

    train_images, train_labels = (
        get_preprocessed_images(
            dataset,
            split,
            "train",
        )
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

    eer, eer_threshold, eer_index = compute_eer(
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
    )

    plot_far_frr(
        thresholds,
        far,
        frr,
        eer_threshold,
    )

    plot_roc(
        far,
        frr,
    )

    print()

    show_eigenfaces(recognizer)