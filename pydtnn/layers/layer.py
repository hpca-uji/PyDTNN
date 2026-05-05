"""
PyDTNN Layer base class
"""
import logging

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array

__all__ = (
    "Layer",
    "LayerError",
    "ParameterException",
    "select",
)

logger = logging.getLogger(__name__)


class LayerError(ValueError):
    pass


class ParameterException(LayerError):
    pass


class Layer[T: Array](Layerable[T]):
    pass


def select(name: str) -> type[Layer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
