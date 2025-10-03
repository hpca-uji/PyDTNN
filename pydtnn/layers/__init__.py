"""
PyDTNN Layers

If you want to add a new layer:
    1) create a new Python file in this directory,
    2) define your layer class as derived from Layer (or any Layer derived class),
    3) and, optionally, import your layer on this file.
"""

from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.batch_normalization_relu import BatchNormalizationRelu
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_relu import Conv2DRelu
from pydtnn.layers.conv_2d_batch_normalization import Conv2DBatchNormalization
from pydtnn.layers.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationRelu
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.layer import Layer
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.layers.abstract_pool_2d_layer import AbstractPool2DLayer
from pydtnn.utils import get_derived_classes

from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D

# Search this module for Layer derived classes and expose them
get_derived_classes(Layer, locals())
