import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    SPLITS_DIR,
    CLASSICAL_IMAGE_SIZE,
)

from src.dataset import load_lfw_dataset


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


def show_split_examples(lfw, split_df):
    """
    Display one training, gallery and probe example.
    """

    split_names = ["train", "gallery", "probe"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for column, split_name in enumerate(split_names):

        row = split_df[
            split_df["split"] == split_name
        ].iloc[0]

        image_index = int(row["image_index"])
        subject = row["subject"]

        original = lfw.images[image_index]
        processed = preprocess_classical(original)

        axes[0, column].imshow(
            to_uint8(original)
        )

        axes[0, column].set_title(
            f"{split_name.upper()}\n{subject}"
        )

        axes[0, column].axis("off")

        axes[1, column].imshow(
            processed,
            cmap="gray",
        )

        axes[1, column].set_title(
            f"Processed {processed.shape}"
        )

        axes[1, column].axis("off")

    plt.tight_layout()
    plt.show()


def print_preprocessing_info(lfw, split_df):
    """
    Print a small sanity check for preprocessing.
    """

    row = split_df.iloc[0]

    image_index = int(row["image_index"])

    original = lfw.images[image_index]
    processed = preprocess_classical(original)

    print("=== PREPROCESSING CHECK ===")

    print(
        f"Original shape:   {original.shape}"
    )

    print(
        f"Original dtype:   {original.dtype}"
    )

    print(
        f"Original range:   "
        f"[{original.min()}, {original.max()}]"
    )

    print()

    print(
        f"Processed shape:  {processed.shape}"
    )

    print(
        f"Processed dtype:  {processed.dtype}"
    )

    print(
        f"Processed range:  "
        f"[{processed.min()}, {processed.max()}]"
    )


if __name__ == "__main__":

    dataset = load_lfw_dataset()

    split = load_split()

    print_preprocessing_info(
        dataset,
        split,
    )

    show_split_examples(
        dataset,
        split,
    )