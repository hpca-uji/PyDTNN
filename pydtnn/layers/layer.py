"""
PyDTNN Layer base class
"""


from pydtnn.layer import LayerAndActivationBase
from pydtnn.utils import find_component
from pydtnn.utils.types import Array


class LayerError(ValueError):
    pass


class ParameterException(LayerError):
    pass


class Layer[T: Array](LayerAndActivationBase[T]):

    @property
    def canonical_name_with_id(self) -> str:
        return f"{self._id_prefix}{self.canonical_name}"


def select(name: str) -> type[Layer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
