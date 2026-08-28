"""Module for defining a simple Multi-Layer Perceptron (MLP) architecture."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.utils.constants import ArrayShape

__all__ = ("simplemlp",)


def simplemlp(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a simple MLP model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers forming the MLP.
    """
    model = list[Layerable]()
    _ = model.append

    _(Identity(shape=input_shape))
    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
