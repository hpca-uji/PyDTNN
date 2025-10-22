from pathlib import Path

import numpy as np
import csv

from itertools import islice
from pydtnn.datasets.dataset import Dataset
from pydtnn.datasets.fromTar import DatasetFromTar
from pydtnn.utils import random
from pydtnn.utils.archive import load_archive, list_directory
from math import ceil

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

SYNTHETIC_INPUT_SHAPE = (3, 600, 600)
CSV_DELIMETER = ','
CSV_LABELS_DELIMETER = '|'
CSV_IMAGES_FIELD = "Image Index"
CSV_LABELS_FIELD = "Finding Labels"
type Class = np.ndarray

# ----------- #
# -- UTILS -- #
def get_dict_file_labels(path: Path) -> dict[str, list[str]]:
    with open(file=path, mode="r") as file:
        reader = csv.DictReader(file, delimiter=CSV_DELIMETER)
        dict_file_labels = dict[str, list[str]]()
        for row in reader:
            image = row[CSV_IMAGES_FIELD]
            labels = row[CSV_LABELS_FIELD].split(CSV_LABELS_DELIMETER)
            dict_file_labels[image] = labels
    return dict_file_labels
# ---
# ----------- #

class ChestXRay14(DatasetFromTar):

    """
    Dataset checksum:
        fef95a7a789bcb0013fbf966cb92c4d92c90becd  images_01.tar.gz
        23d2f2cd62d0271b16869abcdd8a00a1fd2492b5  images_02.tar.gz
        69935ac7886c18246446899cb2b75443195847c4  images_03.tar.gz
        fb86b3adaad0e9ff1405154cf6521e180063af10  images_04.tar.gz
        baa8155f0285edb4a07e717f79682713416eb205  images_05.tar.gz
        3a4252d82143757600885121bb57b0ef4e482532  images_06.tar.gz
        cd3cd855acb4e12ca11608be6aae99414d4bc22b  images_07.tar.gz
        d8891e0079e88fc04dab45253b86a2214ff499b6  images_08.tar.gz
        84661300d777e07be9ae7d5f37fb82721202f0bc  images_09.tar.gz
        30216f59778f259db91d77bcd3d0495c8fce88ef  images_10.tar.gz
        97985118ba36f18c27d62371d28c1698478cecfa  images_11.tar.gz
        cb2865369f434a9deea11e2d5222b8472890681b  images_12.tar.gz
    """

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):

        self._xy_filenames: list[list[tuple[tuple[str, ...], Class]]]
        self._dict_images_labels: dict[str, list[str]]
        self.labels2classes: dict[str, int]
        self.files: Path

        self.model = model

        if not self.model.resize:
            raise ValueError("Model resize must be enabled for dataset!")

        csv = Path(self.model.dataset_metadata_path)
        self.files = Path(self.model.dataset_path)
        split_percentage_test = self.model.test_split

        self._dict_images_labels = get_dict_file_labels(csv)
        self._xy_filenames = [[((), np.empty((0,)))] for _ in Dataset.Part]

        # Splitting the dataset.
        _total_samples = len(self._dict_images_labels)
        test_samples = ceil(_total_samples * split_percentage_test)
        train_samples = _total_samples - test_samples

        # Getting the labels and equivalence class - label
        labels = sorted(list({elem for list_elems in self._dict_images_labels.values() for elem in list_elems}))
        self.labels2classes = {labels[_class]: _class for _class in range(len(labels))}
        output_shape = (len(labels), )

        super().__init__(model, train_samples, test_samples, input_shape=SYNTHETIC_INPUT_SHAPE, output_shape=output_shape, 
                         max_prefetch=model.batch_size, force_test_as_validation=force_test_as_validation, debug=debug)
    # ----

    def _get_labels(self, image_file: str) -> Class:
        labels = self._dict_images_labels[image_file]
        mask = np.zeros(self.output_shape, dtype=np.uint8)
        for label in labels:
            mask[self.labels2classes[label]] = 1
        return mask
    # ----

    def _init_actual_data(self):
        files = list_directory(self.files)

        train_files = islice(files, self._nsamples[Dataset.Part.TRAIN])
        test_files = islice(files, self._nsamples[Dataset.Part.TRAIN], self._nsamples[Dataset.Part.TEST])
        val_files = islice(files, self._nsamples[Dataset.Part.TEST], self._nsamples[Dataset.Part.VAL])

        self._xy_filenames[Dataset.Part.TRAIN] = [(data, self._get_labels(Path(data[-1]).name)) for data in train_files]
        self._xy_filenames[Dataset.Part.TEST] = [(data, self._get_labels(Path(data[-1]).name)) for data in test_files]
        self._xy_filenames[Dataset.Part.VAL] = [(data, self._get_labels(Path(data[-1]).name)) for data in val_files]
    # ----