"""
PyDTNN GPU Layers

If you want to add a new GPU layer:
    1) create a new Python file in this directory,
    2) define your layer class as derived from LayerGPU and, optionally, other Layer derived class,
    3) and, optionally, import your GPU layer on this file.
"""

from . import memory_allocation

from .layer_gpu import LayerGPU
from .addition_block_gpu import AdditionBlockGPU
from .average_pool_2d_gpu import AveragePool2DGPU
from .batch_normalization_gpu import BatchNormalizationGPU
from .concatenation_block_gpu import ConcatenationBlockGPU
from .conv_2d_gpu import Conv2DGPU
from .dropout_gpu import DropoutGPU
from .fc_gpu import FCGPU
from .flatten_gpu import FlattenGPU
from .input_gpu import InputGPU
from .max_pool_2d_gpu import MaxPool2DGPU
from .adaptive_average_pool_2d_gpu import AdaptiveAveragePool2DGPU
from pydtnn.utils import get_derived_classes

# Search this module for LayerGPU derived classes and expose them
get_derived_classes(LayerGPU, locals())
