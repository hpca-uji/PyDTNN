"""
PyDTNN Layer base class
"""

from abc import ABC

from .layer_and_activation_base import LayerAndActivationBase
from ..backends import PromoteToBackendMixin


class Layer(PromoteToBackendMixin, LayerAndActivationBase, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def canonical_name_with_id(self) -> str:
        return f"{self._id_prefix}{self.canonical_name}"
