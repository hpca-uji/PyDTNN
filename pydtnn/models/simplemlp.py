from collections.abc import Sequence

from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_and_activation_base import LayerAndActivationBase
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax


def simplemlp(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model
