"""PyCUDA implementation of the categorical hinge metric."""
import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.categorical_hinge import CategoricalHinge

__all__ = ("CategoricalHingePycuda",)

logger = logging.getLogger(__name__)


class CategoricalHingePycuda(CategoricalHinge[TensorArray], MetricPycuda):
    """Categorical hinge metric implementation for PyCUDA backends."""
    def _model_init(self) -> None:
        """Initializes the metric buffers on the GPU."""
        super()._model_init()
        self.res = TensorArray.new_zeros(shape=(1,), dtype=np.dtype(self.model.dtype), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_res = TensorArray.new_zeros(shape=(self.model.batch_size,), dtype=np.dtype(self.model.dtype), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """Computes the categorical hinge loss between predictions and targets.

        Args:
            y_pred: Predicted tensor values.
            y_targ: Ground truth tensor values.

        Returns:
            The computed categorical hinge loss as a float.
        """
        n = y_pred.shape[0]

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(y_pred.shape[1])

        self.kernel(y_targ.ary, y_pred.ary, self.res.ary, self.local_res.ary, n, num_classes, grid=self.grid, block=self.block, stream=self.model.stream)
        return float(self.res.get())