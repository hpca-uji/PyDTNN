"""
PyDTNN Layer base class
"""


from pydtnn.layer_base import LayerBase
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array


class LayerError(ValueError):
    pass


class ParameterException(LayerError):
    pass


class Layer[T: Array](LayerBase[T]):
    pass


def select(name: str) -> type[Layer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
