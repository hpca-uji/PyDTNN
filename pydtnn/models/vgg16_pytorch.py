"""VGG architecture implementations for PyDTNN."""

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

__all__ = (
    "vgg16",
    "vgg6",
    "vgg8",
)


def vgg16(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Constructs a VGG16 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the VGG16 model.
    """
    model = []
    _ = model.append
    _(Identity(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128], [3, 256], [3, 512], [3, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(
            MaxPool2D(pool_shape=(2, 2), stride=2, padding=1)
        )  # NOTE: Model breaks with initial input size < (32, 32), as input size < pool shape

    _(Flatten())
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(0.5))
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(0.5))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model


def vgg8(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Constructs a VGG8 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the VGG8 model.
    """
    model = []
    _ = model.append
    _(Identity(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128], [2, 256]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model


def vgg6(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Constructs a VGG6 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the VGG6 model.
    """
    model = []
    _ = model.append
    _(Identity(shape=input_shape))

    conv_pattern = [[1, 64], [1, 128], [2, 256]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model


vgg16_pytorch = vgg16
