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

from time import time
from typing import Iterable
type DataPath = str
type ClassName = str

class FolderLoader():
    """
    This class will receive the path to a dataset divided in different sub-folders where every sub-folder is a different data class, and will
    generate [COMPLETAR].
    For example:    
    - Dataset:
        - A: img1, img2
        - B: img3, img4, img5
        - C: img6

    The Dataset is composed by img1 and img2, which belongs to the class A; img3, img4 and img5, which belong to class the class B; and img6, which belongs to class C.
    """
    def __init__(self, dataset_path:str, files_extention: None | str | list[str] = None, preserve_class_name:bool = True):
        """
        - dataset_path (str): Path to the dataset.
        - files_extention (None, str, list[str]): The supported file extetions. If it is None, then all extentions are supported.
        - preserve_class_name (bool): If it's \'True\', the name of the classes will be the same as the directories. If it's \'False\', then the classes's name will be numbers starting from zero.
        - type_data_loading (Literal["static", "dynamic"]): If it's \'static\', all the data will be loaded. If it's \'dinamic\', the data will be loaded as thery are used.
        - shuffle_data_set (bool): If it's \'True\' it will give the dataset's data in a random order.
        """
        assert os.path.isdir(dataset_path), f"\'{dataset_path}\' should be a directory."
        
        self.path = dataset_path
        self._dict_class_file = dict[ClassName, set[DataPath]]()
        self._data_load_funcion: list[tuple[ClassName, Image.Image]] | Iterable[tuple[ClassName, Image.Image]] = None        
        self.num_images = 0       
                
        for file in os.listdir(self.path):
            path_folder = os.path.join(self.path, file)
            if os.path.isdir(path_folder): 
                # TODO/NOTE: Should I short this?                 
                class_name = file if preserve_class_name else len(self._dict_class_file)                
                data_set = set(file for file in [os.path.join(path_folder, file) for file in os.listdir(path_folder)] if os.path.isfile(file) and (files_extention is None or file.endswith(files_extention)))                
                assert len(data_set) != 0, f"There are not files in \'{path_folder}\'{'.' if files_extention is None else f' with any of the following extensions: {str(files_extention).replace('[', '').replace(']', '')}.'}"
                self._dict_class_file[class_name] = data_set
                self.num_images += len(data_set)
        assert len(self._dict_class_file), f"There are no directories in \'{self.path}\'."        
    # --- END __init__ --- #
    
    def get_number_element_per_class(self) -> dict[ClassName, int]:
        return {key: len(self._dict_class_file[key]) for key in self._dict_class_file.keys()}

    def set_class_names(self, new_names: list[str] | dict[ClassName, str]) -> None:
        """
        This method set classes names with the values passed as parameters, that can be a dictionary or a list.
        - new_names (list[str]): the new names will be set with the same order as you get from dict.keys().

        OR        

        - new_names (dict[ClassName, str]): will change the name from the dictionary's key to it's value (ClassName -> new_names[ClassName]).
        
        Notes: 
        - ClassName is a string that represents a name of a class.
        - The number of elements of the list or the number of keys of the dictionary must be the same as classes; if not, the method will raise an AssertionError.
        - If the parameter is a dictionary and there is a key that is not a class name, then it will raise an KeyError.
        """
        num_classes = len(self._dict_class_file)
        assert num_classes == len(new_names), f"The number of classes ({num_classes}) is not the same as the number of elements passed as parameter ({len(new_names)})."
        
        if isinstance(new_names, list):
            list_keys = self._dict_class_file.keys()
            for i in range(num_classes):
                self._dict_class_file[new_names[i]] = self._dict_class_file[list_keys[i]]
                del self._dict_class_file[list_keys[i]]
        else: # isinstance(new_names, dict):
            for old_key in new_names.keys():
                self._dict_class_file[old_key] = new_names[old_key] # new_key=new_names[old_key]
                del self._dict_class_file[old_key]
    # --- End set_class_names --- #

    def static_data_load(self, shuffle_data_set:bool = False) -> list[tuple[ClassName, np.ndarray]]:
        
        
        labels_and_images = list()

        for class_name, set_images in self._dict_class_file.items():
            for path_image in set_images:
                image = Image.open(path_image)
                labels_and_images.append((class_name, np.array(image)))

        if shuffle_data_set:
            shuffle(labels_and_images)        
        labels, images = zip(*labels_and_images)
            
        return np.ndarray(labels), np.ndarray(images)
    # --- END _static_data_load --- #

    def dynamic_data_load(self, shuffle_data_set:bool = False) -> Iterable[tuple[ClassName, np.ndarray]]:
        labels_and_images = [(class_name, path_image) for class_name, set_path_image in self._dict_class_file.items() for path_image in set_path_image]
        
        if shuffle_data_set:
            shuffle(labels_and_images)

        # TODO: Mirar el cargador de Mnist

        for _ in range(len(labels_and_images)):
            label, path_image = labels_and_images.pop()
            image = Image.open(path_image)
            yield (label, np.array(image))
    # --- END _dynamic_data_load --- #

# --- END FolderLoader --- #