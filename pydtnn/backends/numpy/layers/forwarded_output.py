"""Numpy's Setted layer implementation."""

import logging
from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np
from pydtnn.layers.forwarded_output import ForwardedOutput
from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.utils.constants import ArrayShape


__all__ = ("ForwardedOutputNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class ForwardedOutputNumpy(ForwardedOutput[np.ndarray], LayerNumpy):
    """Numpy's implementation of the 'Setted Output' layer"""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        super()._model_init(prev_shape, x)
        self.seted_forward_output = np.asarray(
            self.forward_initializer(self.shape, self.model.param_dtype, self.model.random), order="C"
        )
        self.memory_used += self.seted_forward_output.nbytes

        self.seted_backward_output = np.asarray(
            self.backward_initializer(self.shape, self.model.param_dtype, self.model.random), order="C"
        )
        self.memory_used += self.seted_backward_output.nbytes
