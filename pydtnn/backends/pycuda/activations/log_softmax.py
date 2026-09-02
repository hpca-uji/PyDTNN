"""PyCUDA backend implementation for the Log Softmax activation function."""

import logging

from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.backends.pycuda.activations.softmax import SoftmaxPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.libs import cudnn as cudnn
from pydtnn.utils.constants import ArrayShape

__all__ = ("LogSoftmaxPycuda",)

logger = logging.getLogger(__name__)


class LogSoftmaxPycuda(LogSoftmax[TensorArray], SoftmaxPycuda):
    """PyCUDA-accelerated Log Softmax activation layer using cuDNN."""

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize layer buffers and cuDNN parameters."""
        super()._model_init(prev_shape, x)
        self.algo = cudnn.cudnnSoftmaxAlgorithm["CUDNN_SOFTMAX_LOG"]
