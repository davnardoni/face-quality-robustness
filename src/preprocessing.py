import cv2
import numpy as np
import pandas as pd

from config import (
    SPLITS_DIR,
    CLASSICAL_IMAGE_SIZE,
)

def to_uint8(image):
    """
    Convert an image to uint8 in the range [0, 255].
    """

    if image.dtype == np.uint8:
        return image.copy()

    image = np.asarray(image)

    # Handle images represented in [0, 1]
    if image.max() <= 1.0:
        image = image * 255.0

    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def rgb_to_gray(image):
    """
    Convert an RGB image to grayscale.
    """

    image = to_uint8(image)

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_image(image, size=CLASSICAL_IMAGE_SIZE):
    """
    Resize an image to a fixed size.

    OpenCV expects size as:
        (width, height)
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA,
    )


def preprocess_classical(image):
    """
    Common preprocessing for classical recognizers
    such as Eigenfaces and LBP/LBPH.

    Pipeline:
        RGB image
            -> uint8
            -> grayscale
            -> resize
    """

    gray = rgb_to_gray(image)

    resized = resize_image(gray)

    return resized


def load_split():
    """
    Load the previously generated experimental split.
    """

    split_path = SPLITS_DIR / "lfw_split.csv"

    if not split_path.exists():
        raise FileNotFoundError(
            "Experimental split not found. "
            "Run 'python -m src.protocol' first."
        )

    return pd.read_csv(split_path)


def get_split_images(
    lfw,
    split_df,
    split_name,
):
    """
    Retrieve original RGB images belonging
    to a specific experimental split.
    """

    rows = split_df[
        split_df["split"] == split_name
    ]

    images = []
    labels = []

    for _, row in rows.iterrows():

        image_index = int(row["image_index"])

        images.append(
            lfw.images[image_index]
        )

        labels.append(
            int(row["label"])
        )

    return (
        np.asarray(images),
        np.asarray(labels),
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
