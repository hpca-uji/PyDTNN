"""
GPU optimizers

If you want to add a new GPU optimizer:
    1) create a new Python file in this directory,
    2) define your optimizer class as derived from Optimizer and, optionally, other Optimizer derived class,
    3) and, optionally, import your optimizer on this file.
"""

from .adam_gpu import AdamGPU
from .nadam_gpu import NadamGPU
from .rmsprop_gpu import RMSPropGPU
from .sgd_gpu import SGDGPU

from pydtnn.optimizers import Optimizer
from pydtnn.utils import get_derived_classes

# Search this module for Optimizer derived classes and expose them
get_derived_classes(Optimizer, locals())

# Aliases
adam_gpu = AdamGPU
nadam_gpu = NadamGPU
rmsprop_gpu = RMSPropGPU
sgd_gpu = SGDGPU
