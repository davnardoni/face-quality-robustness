import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from config import (
    PCA_EXPLAINED_VARIANCE,
    N_EIGENFACES_TO_SHOW,
    CLASSICAL_IMAGE_SIZE,
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
