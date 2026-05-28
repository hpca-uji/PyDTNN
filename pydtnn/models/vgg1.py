"""VGG1 model architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

__all__ = ("vgg1",)


def vgg1(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a VGG1-style convolutional neural network architecture.

    Args:
        input_shape: The shape of the input data (channels, height, width).
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers defining the VGG1 model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=input_shape))
    _(
        Conv2D(
            nfilters=32,
            filter_shape=(3, 3),
            padding=1,
            activation=Relu,
            weights_initializer=he_uniform,
        )
    )
    _(
        Conv2D(
            nfilters=32,
            filter_shape=(3, 3),
            padding=1,
            activation=Relu,
            weights_initializer=he_uniform,
        )
    )
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation=Relu, weights_initializer=he_uniform))
    _(FC(shape=output_shape, activation=Softmax))

    return model
