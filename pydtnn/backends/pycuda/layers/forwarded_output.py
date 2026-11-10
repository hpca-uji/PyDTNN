"""PyCuda's Setted layer implementation."""

import logging

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.layers.forwarded_output import ForwardedOutput
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import TensorArray

__all__ = ("ForwardedOutputPycuda",)

logger = logging.getLogger(__name__)


class ForwardedOutputPycuda(ForwardedOutput[TensorArray], LayerPycuda):
    """Numpy's implementation of the 'Setted Output' layer"""

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        self.seted_forward_output = TensorArray.to_gpu(
            self.forward_initializer(self.shape, self.model.param_dtype, self.model.random),
            self.model.tensor_format,
            self.model.cudnn_dtype,
            TensorArray.TensorType.TENSOR,
        )
        self.memory_used += self.seted_forward_output.nbytes

        self.seted_backward_output = TensorArray.to_gpu(
            self.backward_initializer(self.shape, self.model.param_dtype, self.model.random),
            self.model.tensor_format,
            self.model.cudnn_dtype,
            TensorArray.TensorType.TENSOR,
        )
        self.memory_used += self.seted_backward_output.nbytes
