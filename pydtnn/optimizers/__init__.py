"""
PyDTNN optimizers

If you want to add a new optimizer:
    1) create a new Python file in this directory,
    2) define your optimizer class as derived from Optimizer,
    3) and, optionally, import your optimizer on this file.
"""

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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import importlib

from .adam import Adam
from .nadam import Nadam
from .optimizer import Optimizer
from .rmsprop import RMSProp
from .sgd import SGD
from ..utils import get_derived_classes

# Search this module for Optimizer derived classes and expose them
get_derived_classes(Optimizer, locals())

# Aliases
adam = Adam
nadam = Nadam
rmsprop = RMSProp
sgd = SGD


def get_optimizer(model):
    """Get optimizer object from model attributes"""
    if not model.enable_cudnn:
        optimizers_module = importlib.import_module("pydtnn.optimizers")
        optimizer_ = getattr(optimizers_module, model.optimizer_name)
    else:
        optimizers_module = importlib.import_module("pydtnn.gpu_backend.optimizers")
        optimizer_ = getattr(optimizers_module, f"{model.optimizer_name}_gpu")
    if model.optimizer_name == "rmsprop":
        opt = optimizer_(learning_rate=model.learning_rate,
                         rho=model.rho,
                         epsilon=model.epsilon,
                         decay=model.decay,
                         dtype=model.dtype)
    elif model.optimizer_name == "adam":
        opt = optimizer_(learning_rate=model.learning_rate,
                         beta1=model.beta1,
                         beta2=model.beta2,
                         epsilon=model.epsilon,
                         decay=model.decay,
                         dtype=model.dtype)
    elif model.optimizer_name == "nadam":
        opt = optimizer_(learning_rate=model.learning_rate,
                         beta1=model.beta1,
                         beta2=model.beta2,
                         epsilon=model.epsilon,
                         decay=model.decay,
                         dtype=model.dtype)
    elif model.optimizer_name == "sgd":
        opt = optimizer_(learning_rate=model.learning_rate,
                         momentum=model.momentum,
                         nesterov=model.nesterov,
                         decay=model.decay,
                         dtype=model.dtype)
    else:
        raise ValueError(f"Optimizer '{model.optimizer}' not recognized.")
    return opt
