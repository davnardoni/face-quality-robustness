import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    SPLITS_DIR,
    SAMPLES_PER_SUBJECT,
    TRAIN_SAMPLES_PER_SUBJECT,
    GALLERY_SAMPLES_PER_SUBJECT,
    PROBE_SAMPLES_PER_SUBJECT,
)

from src.dataset import load_lfw_dataset


def create_balanced_split(lfw):
    """
    Create a balanced experimental split.

    For each subject:
        - select exactly SAMPLES_PER_SUBJECT images
        - assign some to training
        - assign some to gallery
        - assign some to probe

    Returns
    -------
    pandas.DataFrame
        Columns:
        - image_index
        - label
        - subject
        - split
    """

    total_required = (
        TRAIN_SAMPLES_PER_SUBJECT
        + GALLERY_SAMPLES_PER_SUBJECT
        + PROBE_SAMPLES_PER_SUBJECT
    )

    if total_required != SAMPLES_PER_SUBJECT:
        raise ValueError(
            "Train + gallery + probe samples must equal "
            "SAMPLES_PER_SUBJECT."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for label, subject_name in enumerate(lfw.target_names):

        # All images belonging to this subject
        subject_indices = np.where(lfw.target == label)[0]

        if len(subject_indices) < SAMPLES_PER_SUBJECT:
            continue

        # Randomly select exactly N images
        selected_indices = rng.choice(
            subject_indices,
            size=SAMPLES_PER_SUBJECT,
            replace=False,
        )

        # Shuffle before assigning roles
        rng.shuffle(selected_indices)

        train_end = TRAIN_SAMPLES_PER_SUBJECT

        gallery_end = (
            train_end
            + GALLERY_SAMPLES_PER_SUBJECT
        )

        train_indices = selected_indices[:train_end]

        gallery_indices = selected_indices[
            train_end:gallery_end
        ]

        probe_indices = selected_indices[gallery_end:]

        for index in train_indices:
            rows.append(
                {
                    "image_index": int(index),
                    "label": int(label),
                    "subject": subject_name,
                    "split": "train",
                }
            )

        for index in gallery_indices:
            rows.append(
                {
                    "image_index": int(index),
                    "label": int(label),
                    "subject": subject_name,
                    "split": "gallery",
                }
            )

        for index in probe_indices:
            rows.append(
                {
                    "image_index": int(index),
                    "label": int(label),
                    "subject": subject_name,
                    "split": "probe",
                }
            )

    return pd.DataFrame(rows)


def save_split(split_df):
    """
    Save the experimental split to CSV.
    """

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SPLITS_DIR / "lfw_split.csv"

    split_df.to_csv(output_path, index=False)

    print(f"Split saved to: {output_path}")


def print_split_info(split_df):
    """
    Print information about the experimental split.
    """

    print("=== EXPERIMENTAL PROTOCOL ===")

    print(f"Subjects: {split_df['label'].nunique()}")
    print(f"Total selected images: {len(split_df)}")

    print()

    counts = split_df["split"].value_counts()

    print(f"Training images: {counts.get('train', 0)}")
    print(f"Gallery images:  {counts.get('gallery', 0)}")
    print(f"Probe images:    {counts.get('probe', 0)}")


if __name__ == "__main__":

    dataset = load_lfw_dataset()

    split = create_balanced_split(dataset)

    print_split_info(split)

    save_split(split)