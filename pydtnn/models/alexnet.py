"""AlexNet architecture implementation for the PyDTNN framework."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.activations.relu import Relu
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape

__all__ = ("alexnet",)


def alexnet(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs an AlexNet model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers representing the AlexNet model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Identity(shape=input_shape))
    _(Conv2D(nfilters=96, filter_shape=(11, 11), padding=0, stride=2, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=256, filter_shape=(5, 5), padding=2, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
