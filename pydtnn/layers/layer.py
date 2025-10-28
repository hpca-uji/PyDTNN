"""
PyDTNN Layer base class
"""


from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.utils.types import Array


class LayerError(ValueError):
    pass


class ParameterException(LayerError):
    pass


class Layer[T: Array](LayerAndActivationBase[T]):

    @property
    def canonical_name_with_id(self) -> str:
        return f"{self._id_prefix}{self.canonical_name}"
