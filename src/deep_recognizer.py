import cv2
import numpy as np
import torch

from facenet_pytorch import InceptionResnetV1

from config import (
    DEEP_IMAGE_SIZE,
    DEEP_BATCH_SIZE,
    DEEP_PRETRAINED_MODEL,
)


class FaceNetRecognizer:
    """
    Deep face representation based on a pretrained
    InceptionResnetV1 model.

    The model produces 512-dimensional,
    L2-normalized face embeddings.
    """

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = InceptionResnetV1(
            pretrained=DEEP_PRETRAINED_MODEL
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    def transform(
        self,
        images,
        batch_size=DEEP_BATCH_SIZE,
    ):
        """
        Extract deep embeddings from RGB face images.

        Parameters
        ----------
        images : array-like
            RGB images from LFW.

        Returns
        -------
        embeddings : ndarray
            Array with shape:
                n_images x 512
        """

        embeddings = []

        for start in range(
            0,
            len(images),
            batch_size,
        ):

            batch_images = images[
                start:start + batch_size
            ]

            batch = self._preprocess_batch(
                batch_images
            )

            batch = batch.to(
                self.device
            )

            with torch.no_grad():

                batch_embeddings = self.model(
                    batch
                )

            embeddings.append(
                batch_embeddings
                .cpu()
                .numpy()
            )

        return np.concatenate(
            embeddings,
            axis=0,
        )

    @staticmethod
    def _preprocess_batch(images):
        """
        Prepare a batch for InceptionResnetV1.

        Pipeline:
            RGB image
            -> uint8 [0,255]
            -> resize to 160x160
            -> CHW tensor
            -> fixed image standardization
        """

        tensors = []

        for image in images:

            image = np.asarray(image)

            if image.dtype != np.uint8:

                if image.max() <= 1.0:
                    image = image * 255.0

                image = np.clip(
                    image,
                    0,
                    255,
                ).astype(np.uint8)

            resized = cv2.resize(
                image,
                (
                    DEEP_IMAGE_SIZE,
                    DEEP_IMAGE_SIZE,
                ),
                interpolation=cv2.INTER_AREA,
            )

            tensor = torch.from_numpy(
                resized.copy()
            )

            tensor = tensor.permute(
                2,
                0,
                1,
            ).float()

            # Same standardization used by
            # facenet-pytorch's MTCNN pipeline.
            tensor = (
                tensor - 127.5
            ) / 128.0

            tensors.append(tensor)

        return torch.stack(tensors)