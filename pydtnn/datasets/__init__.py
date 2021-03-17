#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

import importlib
import sys


def get_dataset(model):
    try:
        dataset_name = {"mnist": "MNIST", "cifar10": "CIFAR10", "imagenet": "ImageNet"}
        dataset_mod = importlib.import_module("pydtnn.datasets.dataset")
        dataset_obj = getattr(dataset_mod, dataset_name[model.dataset_name])
        dataset = dataset_obj(train_path=model.dataset_train_path,
                              test_path=model.dataset_test_path,
                              model=model.model_name,
                              test_as_validation=model.test_as_validation,
                              flip_images=model.flip_images,
                              flip_images_prob=model.flip_images_prob,
                              crop_images=model.crop_images,
                              crop_images_size=model.crop_images_size,
                              crop_images_prob=model.crop_images_prob,
                              dtype=model.dtype,
                              use_synthetic_data=model.use_synthetic_data)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return dataset
