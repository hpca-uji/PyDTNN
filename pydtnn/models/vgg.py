from collections.abc import Sequence
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase
from pydtnn.layers.max_pool_2d import MaxPool2D

# NOTE: PyDTNN follows PyTorch's definitions
# NOTE: TensorFlow does not includes Dropout layers after final ReLUs


def vgg16(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
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


def vgg8(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
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


def vgg6(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
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


vgg = vgg16
