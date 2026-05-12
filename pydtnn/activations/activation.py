"""
Module for defining base activation layer functionality and component selection.
"""

import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array, ArrayShape

__all__ = (
    "Activation",
    "select",
)

logger = logging.getLogger(__name__)


class Activation[T: Array](Layerable[T]):
    """
    Base class for all activation layers in the PyDTNN framework.
    """

    def __init__(self, shape: ArrayShape = (1,)):
        """
        Initializes the activation layer with a specified shape.

        Args:
            shape: The expected shape of the input data.
        """
        super().__init__(shape)

    def _model_init(self, prev_shape: ArrayShape, x: T | None = None):
        """
        Initializes the layer parameters based on the previous layer's shape.

        Args:
            prev_shape: The shape of the output from the preceding layer.
            x: Optional input data for initialization.
        """
        super()._model_init(prev_shape, x)
        self.shape = prev_shape


def select(name: str) -> type[Activation]:
    """
    Retrieves an activation class by its name from the package.

    Args:
        name: The name of the activation component to retrieve.

    Returns:
        The class type of the requested activation.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)
