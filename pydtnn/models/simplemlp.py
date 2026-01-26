from collections.abc import Sequence

from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.utils.constants import ArrayShape

def simplemlp(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model
