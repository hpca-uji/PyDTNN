"""Module providing a identity model implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.utils.constants import ArrayShape

__all__ = ("identity",)


def identity(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Identity model without weights"""
    model = list[Layerable]()
    _ = model.append

    _(Identity(shape=input_shape))
    _(Flatten())

    return model
