"""
PyDTNN GPU Layers

If you want to add a new GPU layer:
    1) create a new Python file in this directory,
    2) define your layer class as derived from LayerGPU and, optionally, other Layer derived class,
    3) and, optionally, import your GPU layer on this file.
"""

from pydtnn.backends.gpu.layers import memory_allocation

from pydtnn.backends.gpu.layers.layer_gpu import LayerGPU
from pydtnn.backends.gpu.layers.addition_block_gpu import AdditionBlockGPU
from pydtnn.backends.gpu.layers.average_pool_2d_gpu import AveragePool2DGPU
from pydtnn.backends.gpu.layers.batch_normalization_gpu import BatchNormalizationGPU
from pydtnn.backends.gpu.layers.concatenation_block_gpu import ConcatenationBlockGPU
from pydtnn.backends.gpu.layers.conv_2d_gpu import Conv2DGPU
from pydtnn.backends.gpu.layers.dropout_gpu import DropoutGPU
from pydtnn.backends.gpu.layers.fc_gpu import FCGPU
from pydtnn.backends.gpu.layers.flatten_gpu import FlattenGPU
from pydtnn.backends.gpu.layers.input_gpu import InputGPU
from pydtnn.backends.gpu.layers.max_pool_2d_gpu import MaxPool2DGPU
from pydtnn.backends.gpu.layers.adaptive_average_pool_2d_gpu import AdaptiveAveragePool2DGPU
from pydtnn.utils import get_derived_classes

# Search this module for LayerGPU derived classes and expose them
get_derived_classes(LayerGPU, locals())
