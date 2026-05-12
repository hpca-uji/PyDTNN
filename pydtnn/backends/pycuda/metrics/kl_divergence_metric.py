"""
PyCUDA implementation of the Kullback-Leibler divergence metric.
"""
import logging

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric

__all__ = ("KLDivergenceMetricPycuda",)

logger = logging.getLogger(__name__)


class KLDivergenceMetricPycuda(KLDivergenceMetric[TensorArray], MetricPycuda):
    """
    Computes the Kullback-Leibler divergence between predictions and targets using PyCUDA.
    """
    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Calculates the KL divergence metric on the GPU.

        Args:
            y_pred: Predicted probability distribution.
            y_targ: Target probability distribution.

        Returns:
            The computed KL divergence value as a float.
        """
        self.kernel(
            y_targ, y_pred, self.cost, np.int32(self.model.batch_size), np.int32(self.shape[1]), np.float32(self.eps), grid=self.model.cuda_grid, block=self.model.cuda_block, stream=self.model.stream
        )
        return float(gpuarray.sum(self.cost).get() / self.model.batch_size)