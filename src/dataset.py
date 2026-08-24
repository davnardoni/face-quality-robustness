from collections import Counter

from sklearn.datasets import fetch_lfw_people

from config import (
    RAW_DATA_DIR,
    MIN_FACES_PER_PERSON,
    LFW_RESIZE,
    LFW_COLOR,
    LFW_FUNNELED,
)


def load_lfw_dataset():
    """
    Download (if necessary) and load the LFW dataset.

    Returns
    -------
    lfw : sklearn.utils.Bunch
        Object containing:
        - images
        - data
        - target
        - target_names
    """

    lfw = fetch_lfw_people(
        data_home=RAW_DATA_DIR,
        min_faces_per_person=MIN_FACES_PER_PERSON,
        resize=LFW_RESIZE,
        color=LFW_COLOR,
        funneled=LFW_FUNNELED,
        download_if_missing=True,
    )

    return lfw


def print_dataset_info(lfw):
    """
    Print general information about the loaded LFW dataset.
    """

    images = lfw.images
    labels = lfw.target
    names = lfw.target_names

    counts = Counter(labels)

    print("=== LFW DATASET ===")
    print(f"Number of images:     {len(images)}")
    print(f"Number of subjects:   {len(names)}")
    print(f"Image shape:          {images[0].shape}")
    print()

    print("Images per subject:")
    for label, count in sorted(counts.items()):
        print(f"{names[label]}: {count}")


if __name__ == "__main__":
    dataset = load_lfw_dataset()
    print_dataset_info(dataset)