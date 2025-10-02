#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#
import os
import numpy as np
from random import shuffle
from PIL import Image

from .dataset import Dataset, shape_t, DatasetEnum, Array
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

from typing import TYPE_CHECKING, override, Generator
if TYPE_CHECKING:
    from pydtnn.model import Model

type DataPath = str
type ClassName = np.number

# TODO: Why is the output_shape in dataset??
INPUT_SHAPE = (3,600,600)
OUTPUT_SHAPE = (5,)

class DatasetFolderLoader(Dataset):
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
        
    def __init__(self, model: "Model", train_nsamples:int=-1, test_nsamples:int=-1, input_shape:shape_t = INPUT_SHAPE, output_shape:shape_t= OUTPUT_SHAPE, 
                 max_batches_online = 2, force_test_as_validation=False, debug=False):
        """
        Args:
            model (Model): Model's object.
            train_nsamples (int): number of train samples. This value will be ignored, the real value will be obtained later.
            test_nsamples (int): number of test samples. This value will be ignored, the real value will be obtained later.
            input_shape (shape_t): input's shape.
            output_shape (shape_t): output's shape.
            max_batches_online (int): The maximum number of batches in memory. default: 40.
            force_test_as_validation (bool): True to force the use of the test dataset as validation. default: False.
            debug (bool): True to show debug prints. default: False.
        """
        # TODO: add all the transformations.
        
        # NOTE: Validation dataset is extracted from the Test one.
        self.model = model
        assert os.path.isdir(self.model.dataset_train_path), f"\'{self.model.dataset_train_path}\' should be a directory."
        assert os.path.isdir(self.model.dataset_test_path), f"\'{self.model.dataset_test_path}\' should be a directory."
        
        #self.new_size = (new_size, new_size) if isinstance(new_size, int) else new_size
        self._nsamples:list[int, int, int] = [0,0,0] # train, val, test
        self.labels_and_images = dict[DatasetEnum, list[tuple[ClassName, DataPath]]]()
        self.max_nsamples_online = max_batches_online * self.model.batch_size

        self.labels_and_images[DatasetEnum.TRAIN], self._nsamples[DatasetEnum.TRAIN] = self._get_dict_class_and_file(path = self.model.dataset_train_path)
        self.labels_and_images[DatasetEnum.TEST], self._nsamples[DatasetEnum.TEST] = self._get_dict_class_and_file(path = self.model.dataset_test_path)

        super().__init__(model=model, train_nsamples=self._nsamples[DatasetEnum.TRAIN],
                         test_nsamples=self._nsamples[DatasetEnum.TEST], 
                         input_shape=input_shape, output_shape=output_shape, 
                         max_batches_online=max_batches_online, 
                         force_test_as_validation=force_test_as_validation, 
                         debug=debug)
        
        if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
            x_shape = (input_shape[1], input_shape[2], input_shape[0])
        else:
            x_shape = input_shape

        self.x = np.ndarray(shape=(self.max_nsamples_online, *x_shape), dtype=self.model.dtype)
        self.y = np.ndarray(shape=(self.max_nsamples_online, *output_shape), dtype=self.model.dtype)

        # Splitting the Train and the Validation dataset.
        if self.test_as_validation:
            self.labels_and_images[DatasetEnum.VAL] = self.labels_and_images[DatasetEnum.TEST]
        else:
            shuffle(self.labels_and_images[DatasetEnum.TRAIN])
            self.labels_and_images[DatasetEnum.VAL] = self.labels_and_images[DatasetEnum.TRAIN][:self._nsamples[DatasetEnum.VAL]]
            self.labels_and_images[DatasetEnum.TRAIN] = self.labels_and_images[DatasetEnum.TRAIN][self._nsamples[DatasetEnum.VAL]:]
    # --- END __init__ --- #

    def _get_dict_class_and_file(self, path: str) -> tuple[list[tuple[ClassName, DataPath]], int]:
        dict_class_file = dict[ClassName, set[DataPath]]()
        num_images = 0
        list_dir = sorted(os.listdir(path))
        for class_name in range(len(list_dir)):
            file = list_dir[class_name]
            path_folder = os.path.join(path, file)
            if os.path.isdir(path_folder):                
                data_set = set(file for file in [os.path.join(path_folder, file) for file in sorted(os.listdir(path_folder))] if os.path.isfile(file))
                dict_class_file[class_name] = data_set
                num_images += len(data_set)
        assert len(dict_class_file) != 0, f"There are no directories in \'{path}\'."

        labels_and_images = [(class_name, path_image) for class_name, set_path_image in dict_class_file.items() for path_image in set_path_image]

        return (labels_and_images, num_images)
    # ---

    
    def get_number_element_per_class(self, dict_class_file: dict[ClassName, set[DataPath]]) -> dict[ClassName, int]:
        return {key: len(dict_class_file[key]) for key in dict_class_file.keys()}
    # ---


    def set_class_names(self, dict_class_file: dict[ClassName, set[DataPath]], 
                        new_names: list[str] | dict[ClassName, str]) -> None:
        """
        This method set classes names with the values passed as parameters, that can be a dictionary or a list.
        - new_names (list[str]): the new names will be set with the same order as you get from dict.keys().

        OR

        - new_names (dict[ClassName, str]): will change the name from the dictionary's key to it's value (ClassName -> new_names[ClassName]).
        
        Notes: 
        - ClassName is a string that represents a name of a class.
        - The number of elements of the list or the number of keys of the dictionary must be the same as classes; if not, the method will raise an AssertionError.
        - If the parameter is a dictionary and there is a key that is not a class name, then it will raise an KeyError.

        Args:
            dict_class_file (dict[ClassName, set[DataPath]]): the dataset with the original names.
            new_names (list[str] | dict[ClassName, str]): the new names.
        Returns:
            Nothing. The changes will be updated in \'dict_class_file\'.
        """
        num_classes = len(dict_class_file)
        assert num_classes == len(new_names), f"The number of classes ({num_classes}) is not the same as the number of elements passed as parameter ({len(new_names)})."
        
        if isinstance(new_names, list):
            list_keys = dict_class_file.keys()
            for i in range(num_classes):
                dict_class_file[new_names[i]] = dict_class_file[list_keys[i]]
                del dict_class_file[list_keys[i]]
        else: # isinstance(new_names, dict):
            for old_key in new_names.keys():
                dict_class_file[old_key] = new_names[old_key] # new_key=new_names[old_key]
                del dict_class_file[old_key]
    # --- End set_class_names --- #


    def _get_image_as_np_ndarray(self, path_image:str) -> np.ndarray:
        image = Image.open(path_image)
        image = image.convert("RGB")
        np_array = np.asarray(image, dtype=self.model.dtype, order="C")
        # NOTE: base image format is HWC.

        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                np_array = self._hwc2chw(np_array)
            case PYDTNN_TENSOR_FORMAT.NHWC:
                pass # The format is correct.
            case _:
                raise TypeError(f"{self.model.tensor_format} format is not supported")
        return np_array
    # --- END _get_image_as_np_ndarray --- #

    def _prepare_label(self, label:np.number, output_shape:shape_t) -> np.ndarray:
        np_label = np.zeros(shape=output_shape, dtype=self.model.dtype, order="C")
        np_label[label] = 1
        return np_label
    # --- END _prepare_label ---#


    @override
    def _init_actual_data(self):
        # There is no initialization, as the data is huge, it will be read from the corresponding files as required
        pass
    # ---


    @override
    def _actual_data_generator(self, part: DatasetEnum) -> Generator[tuple[Array, Array]]:

        if part is DatasetEnum.TRAIN: 
            shuffle(self.labels_and_images[part])
        
        images = list[np.ndarray]()
        labels = list[ClassName]()

        for label, path_image in self.labels_and_images[part]:
            image = self._get_image_as_np_ndarray(path_image)
            label = self._prepare_label(label, self.output_shape)

            if len(images) < self.max_nsamples_online:
                images.append(image)
                labels.append(label)
            else:
                np.stack(images, out=self.x)
                np.stack(labels, out=self.y)
                images.clear()
                labels.clear()
                yield self.x, self.y
        #} - for
        
        num_not_processed_images = len(images)
        if num_not_processed_images != 0:
            np.stack(images, out=self.x[:num_not_processed_images])
            np.stack(labels, out=self.y[:num_not_processed_images])
            images.clear()
            labels.clear()
            yield self.x[:num_not_processed_images], self.y[:num_not_processed_images]
        #else: Since all the data was already yielded inside the for, do nothing.
    # --- END _actual_data_generator --- #


# --- END FolderLoader --- #
