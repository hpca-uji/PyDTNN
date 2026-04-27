from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pycuda import gpuarray  # type: ignore
import numpy as np
import logging
logger = logging.getLogger(__name__)


class KLDivergenceMetricPycuda(KLDivergenceMetric[TensorArray], MetricPycuda):

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        self.kernel(y_targ, y_pred, self.cost,
                    np.int32(self.model.batch_size), np.int32(self.shape[1]),
                    np.float32(self.eps),
                    grid=self.model.cuda_grid, block=self.model.cuda_block,
                    stream=self.model.stream)
        return float(gpuarray.sum(self.cost).get()) / self.model.batch_size
