import numpy as np

from config import (
    LBP_GRID_X,
    LBP_GRID_Y,
    LBP_HIST_BINS,
)


def compute_lbp(image):
    """
    Compute the Local Binary Pattern map using
    the 8 neighbours of each pixel.
    """

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    center = image[1:-1, 1:-1]

    lbp = np.zeros_like(
        center,
        dtype=np.uint8,
    )

    neighbours = [
        image[:-2, :-2],    # top-left
        image[:-2, 1:-1],   # top
        image[:-2, 2:],     # top-right
        image[1:-1, 2:],    # right
        image[2:, 2:],      # bottom-right
        image[2:, 1:-1],    # bottom
        image[2:, :-2],     # bottom-left
        image[1:-1, :-2],   # left
    ]

    for bit, neighbour in enumerate(neighbours):

        comparison = neighbour >= center

        lbp |= (
            comparison.astype(np.uint8)
            << bit
        )

    return lbp


def compute_lbph_descriptor(
    image,
    grid_x=LBP_GRID_X,
    grid_y=LBP_GRID_Y,
):
    """
    Compute an LBPH descriptor.

    Pipeline:
        image
        -> LBP map
        -> spatial grid
        -> histogram for each region
        -> concatenation
    """

    lbp = compute_lbp(image)

    height, width = lbp.shape

    cell_height = height // grid_y
    cell_width = width // grid_x

    histograms = []

    for row in range(grid_y):

        for col in range(grid_x):

            y_start = row * cell_height
            x_start = col * cell_width

            if row == grid_y - 1:
                y_end = height
            else:
                y_end = (row + 1) * cell_height

            if col == grid_x - 1:
                x_end = width
            else:
                x_end = (col + 1) * cell_width

            cell = lbp[
                y_start:y_end,
                x_start:x_end,
            ]

            histogram, _ = np.histogram(
                cell,
                bins=LBP_HIST_BINS,
                range=(0, LBP_HIST_BINS),
            )

            histogram = histogram.astype(
                np.float32
            )

            histogram /= (
                histogram.sum() + 1e-10
            )

            histograms.append(histogram)

    return np.concatenate(histograms)


class LBPHRecognizer:
    """
    Face representation based on Local Binary
    Pattern Histograms.
    """

    def transform(self, images):
        """
        Extract the LBPH descriptor for each image.
        """

        embeddings = []

        for image in images:

            descriptor = compute_lbph_descriptor(
                image
            )

            embeddings.append(descriptor)

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )