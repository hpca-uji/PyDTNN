import typing
from pathlib import Path

import numpy as np
from PIL import Image
import csv

from itertools import islice
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.datasets.dataset import Dataset
from pydtnn.utils import random
from pydtnn.utils.archive import load_archive, list_archive, list_directory
from math import ceil

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

SYNTHETIC_INPUT_SHAPE = (3, 600, 600)
CSV_DELIMETER = ','
CSV_LABELS_DELIMETER = '|'
CSV_IMAGES_FIELD = "Image Index"
CSV_LABELS_FIELD = "Finding Labels"
type Class = np.ndarray[int]

# TODO: move to parser
SPLIT_PERCENTAGE_TEST = 0.2

# ----------- #
# -- UTILS -- #
def get_dict_file_labels(path: Path) -> dict[str, list[str]]:
    with open(file=path, mode="r") as file:
        reader = csv.DictReader(file, delimiter=CSV_DELIMETER)
        dict_file_labels = dict[str, str]()
        for row in reader:
            image = row[CSV_IMAGES_FIELD]
            labels = row[CSV_LABELS_FIELD].split(CSV_LABELS_DELIMETER)
            dict_file_labels[image] = labels
    return dict_file_labels
# ---

# ----------- #

class ChestXRay14(Dataset):

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):

        self._xy_filenames: list[tuple[Path, Class]]
        self._dict_images_labels: dict[str, list[str]]
        self.labels2classes: dict[str, int]
        self.files: Path

        self.model = model

        if not self.model.resize:
            raise ValueError("Model resize must be enabled for dataset!")
        
        # TODO: add variables to parser for dataset_path

        csv = Path(self.model.dataset_path)
        self.files = Path(self.model.dataset_train_path)

        self._dict_images_labels = get_dict_file_labels(csv)
        self._xy_filenames = list(Dataset.Part)

        # Splitting the dataset.
        _total_samples = len(self._dict_images_labels)
        test_samples = ceil(_total_samples * SPLIT_PERCENTAGE_TEST)
        train_samples = _total_samples - test_samples

        # Getting the labels and equivalence class - label
        labels = sorted(list({elem for list_elems in self._dict_images_labels.values() for elem in list_elems}))
        self.labels2classes = {labels[_class]: _class for _class in range(len(labels))}
        output_shape = (len(labels), )

        super().__init__(model, train_samples, test_samples, input_shape=SYNTHETIC_INPUT_SHAPE, output_shape=output_shape, 
                         max_prefetch=model.batch_size, force_test_as_validation=force_test_as_validation, debug=debug)
        # ------

    def _get_labels(self, labels: list[str]) -> np.ndarray:
        mask = np.zeros(len(labels), dtype=np.uint8)
        for label in labels:
            mask[self.labels2classes[label]] = 1
        return mask

    def _load_image(self, fp: typing.IO[bytes]) -> np.ndarray:
        """Transform a file-like (image) to array (ndarray CHW uint8)"""
        with Image.open(fp=fp) as image:
            image = image.convert("RGB")
            array = np.asarray(image)
            # NOTE: PIL mode RGB is WHC in unit8
            array = array.transpose(2, 1, 0)
        return array

    def _init_actual_data(self):
        files = list_directory(self.files)

        train_files = islice(files, self._nsamples[Dataset.Part.TRAIN])
        test_files = islice(files, self._nsamples[Dataset.Part.TRAIN], self._nsamples[Dataset.Part.TEST])
        val_files = islice(files, self._nsamples[Dataset.Part.TEST], self._nsamples[Dataset.Part.VAL])

        self._xy_filenames[Dataset.Part.TRAIN] = [(im, self._dict_images_labels[im[-1]]) for im in train_files]
        self._xy_filenames[Dataset.Part.TEST] = [(im, self._dict_images_labels[im[-1]]) for im in test_files]
        self._xy_filenames[Dataset.Part.VAL] = [(im, self._dict_images_labels[im[-1]]) for im in val_files]
    # ----

    def _actual_data_generator(self, part):
        offset = self._local_offset[part]
        nsamples = self._local_nsamples[part]
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN:
            random.shuffle(xy_filenames)

        xy_filenames = xy_filenames[offset:offset + nsamples]

        for path, y in xy_filenames:
            with load_archive(*path) as fp:
                x = self._load_image(fp)

            # Add N dimension
            x = x[None, ...]
            y = y[None, ...]

            # Set tensor format
            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    x = self._nchw2nhwc(x)
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    pass
                case _:
                    raise ValueError("Unsupported tensor format")

            # Set dtype and order
            x = x.astype(dtype=self.model.dtype, order="C")
            y = y.astype(dtype=self.model.dtype, order="C")

            # Inplace normalization
            x /= 255.0

            yield x, y
