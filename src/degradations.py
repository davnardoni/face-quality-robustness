import cv2
import numpy as np


def ensure_float_image(image):
    """
    Convert an image to float32 in the range [0, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(
        image,
        0.0,
        1.0,
    )


def gaussian_blur(image, sigma):
    """
    Apply Gaussian blur.

    Parameters
    ----------
    image : ndarray
        RGB image.

    sigma : float
        Standard deviation of the Gaussian kernel.
        sigma = 0 means no degradation.
    """

    image = ensure_float_image(image)

    if sigma <= 0:
        return image.copy()

    blurred = cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return np.clip(
        blurred,
        0.0,
        1.0,
    )


def gaussian_noise(
    image,
    sigma,
    rng=None,
):
    """
    Add zero-mean Gaussian noise.

    sigma is expressed in normalized [0, 1]
    image intensity units.
    """

    image = ensure_float_image(image)

    if sigma <= 0:
        return image.copy()

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    )

    noisy = image + noise

    return np.clip(
        noisy,
        0.0,
        1.0,
    ).astype(np.float32)


def adjust_brightness(
    image,
    factor,
):
    """
    Modify image brightness.

    factor = 1.0:
        unchanged image

    factor < 1.0:
        darker image

    factor > 1.0:
        brighter image
    """

    image = ensure_float_image(image)

    adjusted = image * factor

    return np.clip(
        adjusted,
        0.0,
        1.0,
    ).astype(np.float32)


def reduce_resolution(
    image,
    scale,
):
    """
    Simulate resolution loss by downsampling
    and then restoring the original dimensions.

    scale = 1.0:
        original resolution

    scale < 1.0:
        lower resolution
    """

    image = ensure_float_image(image)

    if scale >= 1.0:
        return image.copy()

    height, width = image.shape[:2]

    low_width = max(
        1,
        int(width * scale),
    )

    low_height = max(
        1,
        int(height * scale),
    )

    low_resolution = cv2.resize(
        image,
        (low_width, low_height),
        interpolation=cv2.INTER_AREA,
    )

    restored = cv2.resize(
        low_resolution,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    return np.clip(
        restored,
        0.0,
        1.0,
    ).astype(np.float32)