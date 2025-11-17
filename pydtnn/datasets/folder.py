import os
import copy
import numpy as np

from pydtnn.datasets.dataset import Dataset
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils import random

from typing import TYPE_CHECKING, override, Generator
if TYPE_CHECKING:
    from pydtnn.model import Model

type DataPath = str
type ClassName = int


class Folder(Dataset):
    """
    This class will receive the path to a dataset divided in different sub-folders where every sub-folder is a different data class, and will
    generate the samples.
    For example:
    - Dataset:
        - A: img1, img2
        - B: img3, img4, img5
        - C: img6

    The Dataset is composed by img1 and img2, which belongs to the class A; img3, img4 and img5, which belong to class the class B; and img6, which belongs to class C.
    """

    def __init__(self, model: "Model", force_test_as_validation=False, debug=False):
        """
        Args:
            model (Model): Model's object.
            force_test_as_validation (bool): True to force the use of the test dataset as validation. default: False.
            debug (bool): True to show debug prints. default: False.
        """

        # NOTE: Validation dataset is extracted from the Test one.
        self.model = model
        if not os.path.isdir(self.model.dataset_path):
            raise NotADirectoryError(f"{self.model.dataset_path!r} should be a directory.")
        dataset_train_path = os.path.join(self.model.dataset_path, "train")
        dataset_test_path = os.path.join(self.model.dataset_path, "test")

        # self.new_size = (new_size, new_size) if isinstance(new_size, int) else new_size
        self._nsamples = [0, 0, 0]  # train, val, test
        self.labels_and_images = dict[Dataset.Part, list[tuple[ClassName, DataPath]]]()

        self.labels_and_images[Dataset.Part.TRAIN], num_classes_train, self._nsamples[Dataset.Part.TRAIN] = self._get_dict_class_and_file(path=dataset_train_path)
        self.labels_and_images[Dataset.Part.TEST], num_classes_test, self._nsamples[Dataset.Part.TEST] = self._get_dict_class_and_file(path=dataset_test_path)

        if num_classes_train != num_classes_test:
            raise ValueError(f"The number of train classes ({num_classes_train}) must be the same as the number of test classes {num_classes_test}.")

        input_shape = (3, 10, 10)  # synthetic
        output_shape = (num_classes_train, )

        super().__init__(model=model, train_nsamples=self._nsamples[Dataset.Part.TRAIN],
                         test_nsamples=self._nsamples[Dataset.Part.TEST],
                         input_shape=input_shape, output_shape=output_shape,
                         force_test_as_validation=force_test_as_validation,
                         debug=debug)

        self.labels_and_images[Dataset.Part.VAL] = copy.copy(self.labels_and_images[Dataset.Part.TEST] if self.test_as_validation else self.labels_and_images[Dataset.Part.TRAIN])

    def _get_dict_class_and_file(self, path: str) -> tuple[list[tuple[ClassName, DataPath]], int, int]:
        dict_class_file = dict[ClassName, set[DataPath]]()
        num_images = 0
        list_dir = sorted(os.listdir(path))
        num_classes = len(list_dir)
        for class_name in range(num_classes):
            file = list_dir[class_name]
            path_folder = os.path.join(path, file)
            if os.path.isdir(path_folder):
                data_set = set(file for file in [os.path.join(path_folder, file) for file in sorted(os.listdir(path_folder))] if os.path.isfile(file))
                dict_class_file[class_name] = data_set
                num_images += len(data_set)

        if len(dict_class_file.values()) == 0:
            raise ValueError(f"There are no directories in \'{path}\'.")

        labels_and_images = [(class_name, path_image) for class_name, set_path_image in dict_class_file.items() for path_image in set_path_image]

        return (labels_and_images, num_classes, num_images)
    # ---

    def _prepare_label(self, label: int, num_classes: ArrayShape) -> np.ndarray:
        """Transform class numer into class mask (ndarray 1D unit8)"""
        np_label = np.zeros(shape=num_classes, dtype=np.uint8, order="C")
        np_label[label] = 1
        return np_label

    @override
    def _init_actual_data(self):
        if not self.model.transform_resize:
            raise ValueError("Model transform_resize must be enabled for dataset!")
    # ---

    @override
    def _actual_data_generator(self, part: Dataset.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        offset = self._local_offset[part]
        nsamples = self._local_nsamples[part]
        labels_and_images = self.labels_and_images[part]

        if part is Dataset.Part.TRAIN and self.model.augment_shuffle:
            random.shuffle(labels_and_images)

        labels_and_images = labels_and_images[offset:offset + nsamples]

        for label, path_image in labels_and_images:
            x = self._load_rgb_image(path_image)
            y = self._prepare_label(label, self.output_shape)

            # Add N dimension
            x = x[None, ...]
            y = y[None, ...]

            # Set tensor format
            x = self.model.encode_tensor(x)

            # Set dtype and order
            x = x.astype(dtype=self.model.dtype, order="C")
            y = y.astype(dtype=self.model.dtype, order="C")

            # Inplace normalization
            x /= 255.0

            yield x, y
