"""
ImageNet dataset implementation for PyDTNN.

Provides functionality to load, parse, and iterate over the ILSVRC 2012
ImageNet dataset, including support for nested TAR archives and label mapping.
"""

from __future__ import annotations

import copy
import io
import logging
import tarfile
import typing
from collections import abc
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Generator

import numpy as np
from scipy.io import loadmat

from pydtnn.datasets.abstract import Dataset
from pydtnn.utils import random

__all__ = ("ImageNet",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TRAIN_NSAMPLES = 1281167
TEST_NSAMPLES = 50000
INPUT_SHAPE = (3, 227, 227)
OUTPUT_SHAPE = (1000,)


def list_archive(root_path: Path) -> typing.Iterator[tuple[str, ...]]:
    """
    Recursively walk through a TAR archive or nested TAR archives.

    Args:
        root_path: Path to the root TAR file.

    Yields:
        A tuple of strings representing the path hierarchy to a file.
    """
    path = str(root_path)

    def is_tar(path: PurePath) -> bool:
        """Check if the given path has a .tar extension."""
        return path.suffix == ".tar"

    if not is_tar(root_path):
        yield (path,)
        return

    with ExitStack() as fp_stack:
        stack: list[tuple[tarfile.TarFile, tuple[str, ...]]] = []

        tar = fp_stack.enter_context(tarfile.open(path, "r:"))
        stack.append((tar, (path,)))

        while stack:
            tar, base_path = stack.pop()

            for member in tar.getmembers():
                if not member.isfile():
                    continue

                path = Path(member.name)
                full_path = (*base_path, str(path))

                if is_tar(path):
                    sub_file = fp_stack.enter_context(tar.extractfile(member))  # type: ignore
                    sub_tar = fp_stack.enter_context(tarfile.open(fileobj=sub_file, mode="r:"))
                    stack.append((sub_tar, full_path))
                else:
                    yield full_path


@contextmanager
def load_archive(*paths: str) -> abc.Generator[typing.IO[bytes]]:
    """
    Open a file located inside nested TAR archives.

    Args:
        *paths: A sequence of paths representing the nesting hierarchy.

    Yields:
        A file-like object for the target file.
    """
    with ExitStack() as stack:
        # First: on disk
        file = stack.enter_context(open(paths[0], "rb"))

        if len(paths) <= 1:
            yield file
            return

        tar = stack.enter_context(tarfile.open(fileobj=file, mode="r:"))

        # Intermediate: nested tars
        for fp in paths[1:-1]:
            file = stack.enter_context(tar.extractfile(fp))  # type: ignore
            tar = stack.enter_context(tarfile.open(fileobj=file, mode="r:"))

        # Last: return
        file = stack.enter_context(tar.extractfile(paths[-1]))  # type: ignore
        yield file


class ImageNet(Dataset):
    """
    ImageNet Dataset

    The most highly-used subset of ImageNet is the ImageNet Large
    Scale Visual Recognition Challenge (ILSVRC) 2012-2017 image
    classification and localization dataset. This dataset spans
    1000 object classes and contains 1,281, 167 training images,
    50,000 validation images and 100,000 test images.

    Source (SHA1): https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
    43eda4fe35c1705d6606a6a7a633bc965d194284 ILSVRC2012_img_train.tar
    5f3f73da3395154b60528b2b2a2caf2374f5f178 ILSVRC2012_img_val.tar
    092a94ed6a05454b8b72d1c4ecf336fa48d37fda ILSVRC2012_devkit_t12.tar.gz

    Normalize (z-score):
    offset: -0.451
    scale:  +3.471
    """

    def __init__(self, model: Model, force_test_as_validation=False, debug=False):
        """Initialize the ImageNet dataset."""
        super().__init__(
            model,
            TRAIN_NSAMPLES,
            TEST_NSAMPLES,
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            force_test_as_validation=force_test_as_validation,
            debug=debug,
        )

    def _get_label(self, code: int, labels: dict[int, int]) -> np.ndarray:
        """Transform a code (int) into a label (ndarray 1D uint8)."""
        label = labels[code] - 1
        mask = np.zeros(len(labels), dtype=np.uint8)
        mask[label] = 1
        return mask

    def _get_train_code(self, name: str) -> int:
        """Transform a file-name (str) to code (int)."""
        label = name
        label = label.replace("_", ".")
        label = label.split(".")[0]
        label = label.lstrip("n")
        label = int(label)
        return label

    def _get_val_code(self, name: str) -> int:
        """Transform a file-name (str) to code (int)."""
        label = name
        label = label.replace("_", ".")
        label = label.split(".")[2]
        label = int(label)
        return label

    def _get_train_labels(self, path: Path) -> dict[int, int]:
        """Get label mappings from archive."""
        member = "ILSVRC2012_devkit_t12/data/meta.mat"
        with tarfile.open(path) as tar, tar.extractfile(member) as fp:  # type: ignore
            meta = loadmat(file_name=fp, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        labels, codes, class_name = list(zip(*meta))[:3]
        return {int(code.lstrip("n")): int(label) for code, label in zip(codes, labels)}

    def _get_val_labels(self, path: Path) -> dict[int, int]:
        """Get label mappings from archive."""
        member = "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        with (
            tarfile.open(path) as tar,
            tar.extractfile(member) as fp,
            io.TextIOWrapper(buffer=fp) as lines,
        ):  # type: ignore
            return {i: int(line) for i, line in enumerate(lines, 1)}

    def _model_init(self):
        """Initialize dataset metadata and file paths."""
        if not self.model.augment_scale:
            raise ValueError("Model augment_resize must be enabled for dataset!")

        meta = Path(self.model.dataset_path) / "ILSVRC2012_devkit_t12.tar.gz"
        train = Path(self.model.dataset_path) / "ILSVRC2012_img_train.tar"
        test = Path(self.model.dataset_path) / "ILSVRC2012_img_val.tar"

        train_lables = self._get_train_labels(meta)

        if len(train_lables) != OUTPUT_SHAPE[0]:
            raise ValueError(
                f"Mismatch class shape (got: {len(train_lables)}, expect: {OUTPUT_SHAPE[0]})"
            )

        train_xy = [
            (path, self._get_label(self._get_train_code(path[-1]), train_lables))
            for path in list_archive(train)
        ]

        if len(train_xy) != TRAIN_NSAMPLES:
            raise ValueError(
                f"Mismatch train samples (got: {len(train_xy)}, expect: {TRAIN_NSAMPLES})"
            )

        val_lables = self._get_val_labels(meta)
        val_xy = [
            (path, self._get_label(self._get_val_code(path[-1]), val_lables))
            for path in list_archive(test)
        ]

        if len(val_lables) != TEST_NSAMPLES:
            raise ValueError(
                f"Mismatch test samples (got: {len(val_lables)}, expect: {TEST_NSAMPLES})"
            )

        self._xy_filenames = [
            train_xy,
            copy.copy(val_xy if self.test_as_validation else train_xy),
            val_xy,
        ]

    def _data_generator(self, part: Dataset.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate data batches for the specified dataset part.

        Args:
            part: The dataset partition (TRAIN, VALIDATION, or TEST).
        """
        offset = self._local_offset[part]
        nsamples = self._local_nsamples[part]
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN and self.model.augment_shuffle:
            # type: ignore (numpy shuffle's typing wasn't well defined.)
            random.shuffle(xy_filenames)

        xy_filenames = xy_filenames[offset: offset + nsamples]

        for path, y in xy_filenames:
            with load_archive(*path) as fp:
                x = self._load_rgb_image(fp)

            # Add N dimension
            x = x[None, ...]
            y = y[None, ...]

            # Set tensor format
            x = self.model.encode_tensor(x)

            # Set dtype and order (normalize)
            x = np.divide(x, 255.0, dtype=self.model.dtype, casting="unsafe")
            y = y.astype(dtype=self.model.dtype)

            yield x, y
