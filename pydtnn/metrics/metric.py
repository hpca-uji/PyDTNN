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
from abc import ABC

import numpy as np

from .. import model as model_module


class Metric(ABC):

    def __new__(cls, *args, **kwargs):
        if not model_module.enable_cudnn:
            new_cls = cls
        else:
            # If GPU is requested, return a GPU-related object instead
            module = importlib.import_module(f"gpu_backend.losses")
            new_cls = getattr(module, f"{cls.__name__}GPU")
        return super(Metric, new_cls).__new__(new_cls)

    def __init__(self, shape, model, eps=1e-8):
        self.shape = shape
        self.b, self.n = shape
        self.model = model
        self.eps = eps
