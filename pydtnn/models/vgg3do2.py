"""VGG-style 3-block deep neural network architecture implementation."""
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
from pydtnn.utils.initializers import he_uniform

__all__ = ("vgg3do2",)


def vgg3do2(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a VGG-style neural network with 3 convolutional blocks and dropout.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers defining the model architecture.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Dropout(rate=0.2))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Dropout(rate=0.3))
    _(Conv2D(nfilters=128, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=128, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Dropout(rate=0.4))
    _(Flatten())
    _(FC(shape=(128,), activation=Relu, weights_initializer=he_uniform))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=Softmax))

    return model