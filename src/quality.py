import cv2
import numpy as np


def to_gray_float(image):
    """
    Convert an RGB image to grayscale float32
    in the range [0, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.max() > 1.0:
        image = image / 255.0

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    return image.astype(
        np.float32
    )


def sharpness_estimation(image):
    """
    Compute the Sharpness Estimation quality index.

    Higher values indicate stronger local
    intensity variations and, in general,
    a sharper image.
    """

    gray = to_gray_float(
        image
    )

    horizontal_differences = np.abs(
        gray[:, 1:]
        - gray[:, :-1]
    )

    vertical_differences = np.abs(
        gray[1:, :]
        - gray[:-1, :]
    )

    horizontal_mean = np.mean(
        horizontal_differences
    )

    vertical_mean = np.mean(
        vertical_differences
    )

    sharpness = 0.5 * (
        horizontal_mean
        + vertical_mean
    )

    return float(sharpness)


def compute_sharpness_scores(images):
    """
    Compute sharpness quality scores
    for a collection of images.
    """

    return np.asarray(
        [
            sharpness_estimation(image)
            for image in images
        ],
        dtype=np.float32,
    )