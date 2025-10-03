from collections.abc import Sequence

from ..activations import *
from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.initializers import he_uniform


def create_vgg3dobn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    for n_filt, do_rate in zip([32, 64, 128], [0.2, 0.3, 0.4]):
        for i in range(2):
            _(Conv2D(nfilters=n_filt, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))
            _(Relu())
            _(BatchNormalization())
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
        _(Dropout(rate=do_rate))
    _(Flatten())
    _(FC(shape=(512,), weights_initializer=he_uniform))
    _(Relu())
    _(BatchNormalization())
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, weights_initializer=he_uniform))
    _(Softmax())

    return model
