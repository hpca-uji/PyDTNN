"""
Module for defining a standard simple convolutional neural network architecture.
"""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape

__all__ = ("simplecnn",)


def simplecnn(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a simple CNN architecture consisting of convolutional, pooling, and fully connected layers.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the CNN model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=4, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=8, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=Softmax))

    return model
