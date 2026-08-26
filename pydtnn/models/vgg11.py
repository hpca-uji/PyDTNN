"""VGG11 model architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape

__all__ = ("vgg11",)


def vgg11(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a VGG11 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers representing the VGG11 model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=input_shape))
    conv_pattern = [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
