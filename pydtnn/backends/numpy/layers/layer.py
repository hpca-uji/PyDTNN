import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.layers.layer import Layer
from pydtnn.libs import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydtnn.model import Model


if TYPE_CHECKING:
    import numpy as np


class LayerNumpy(Layer[np.ndarray], LayerableNumpy):
    """
    Extends a Layer class with the attributes and methods required by CPU Layers.
    """

    model: "Model[np.ndarray]"

    @property
    def _ary_prop(self) -> set[str]:
        return {*self.grad_vars.keys(), *self.grad_vars.values()}

    def _export_prop(self, key: str):
        if key not in self._ary_prop:
            return super()._export_prop(key)

        ary = getattr(self, key)
        return np.asarray(ary, dtype=np.float64, order="C", copy=True)

    def _import_prop(self, key: str, value) -> None:
        if key not in self._ary_prop:
            return super()._import_prop(key, value)

        ary = getattr(self, key)
        ary[:] = value
