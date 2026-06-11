"""
MobileNetV2 implementation for the PyDTNN framework.
"""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.arctanh import Arctanh
from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.activations.log import Log
from pydtnn.activations.relu6 import Relu6
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.activations.softmax import Softmax
from pydtnn.activations.tanh import Tanh
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape

__all__ = ("testNet",)


def testNet(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs one of the models to test all the library's layers.
    The objective is not to make a "good" model, only a model that has all the layers to test them.

    Args:
        input_shape: The shape of the input tensor.
        output_shape: The shape of the output tensor.

    Returns:
        A sequence of Layerable objects representing the model.
    """
    nfilters = 8

    model = []
    _ = model.append
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=2, use_bias=False))
    _(Relu6())
    _(AveragePool2D())

    _(ConcatenationBlock(
        [
        Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, use_bias=False),
        Log(),
        AdaptiveAveragePool2D((2, 2))
        ],

        [
        Conv2D(nfilters=nfilters, filter_shape=(1, 1), use_bias=False),
        Tanh(),
        AdaptiveAveragePool2D((2, 2))
        ]
    ))

    _(AdditionBlock(
        [Conv2DDepthwise(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, use_bias=False),
         LeakyRelu(),
         Conv2DPointwise(nfilters=nfilters, filter_shape=(1, 1), use_bias=False),
         BatchNormalization(),
         Sigmoid(),
         ],
        [Conv2DDepthwise(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, use_bias=False),
         LeakyRelu(),
         Conv2DPointwise(nfilters=nfilters, filter_shape=(1, 1), use_bias=False),
         BatchNormalization(),
         Sigmoid()]
    ))
    _(MaxPool2D())
    _(Flatten())
    _(Dropout(0.2))
    _(FC(shape=output_shape, activation=Softmax))

    return model
