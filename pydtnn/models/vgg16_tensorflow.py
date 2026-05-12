"""VGG model architectures implemented using TensorFlow-style configurations."""
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

# NOTE: PyDTNN follows PyTorch's definitions
# NOTE: TensorFlow does not includes Dropout layers after final ReLUs


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
    model = list[Layerable]()
    _ = model.append
    _(Input(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128], [3, 256], [3, 512], [3, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2, padding=1))  # NOTE: Model breaks with initial input size < (32, 32), as input size < pool shape

    _(Flatten())
    _(FC(shape=(4096,), activation=Relu))
    _(FC(shape=(4096,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model


def vgg8(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Constructs a VGG8 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the VGG8 model.
    """
    model = list[Layerable]()
    _ = model.append
    _(Input(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model


def vgg6(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Constructs a VGG6 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the VGG6 model.
    """
    model = list[Layerable]()
    _ = model.append
    _(Input(shape=input_shape))

    conv_pattern = [[1, 64], [1, 128]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(256,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model


vgg16_tensorflow = vgg16