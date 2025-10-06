from pydtnn.layers import *
from pydtnn.activations import *

from collections.abc import Sequence
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase

# NOTE: PyDTNN follows PyTorch's definitions
# NOTE: TensorFlow does not includes Dropout layers after final ReLUs


def vgg16(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append
    _(Input(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128], [3, 256], [3, 512], [3, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2, padding=1))  # NOTE: Model breaks with initial input size < (32, 32), as input size < pool shape

    _(Flatten())
    _(FC(shape=(4096,), activation=relu))
    _(FC(shape=(4096,), activation=relu))
    _(FC(shape=output_shape, activation=softmax))

    return model


def vgg8(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append
    _(Input(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=output_shape, activation=softmax))

    return model


def vgg6(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append
    _(Input(shape=input_shape))

    conv_pattern = [[1, 64], [1, 128]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(256,), activation=relu))
    _(FC(shape=output_shape, activation=softmax))

    return model


create_vgg = vgg16
