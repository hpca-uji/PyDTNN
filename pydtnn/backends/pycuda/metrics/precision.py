from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.precision import Precision
from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
import numpy as np
import logging
logger = logging.getLogger(__name__)


class PrecisionPycuda(Precision[TensorArray], MetricPycuda):

    def _model_init(self) -> None:
        super()._model_init()
        target_classes = self.model.output_shape[0]
        self.precision = TensorArray.new_zeros(shape=(1, ), dtype=np.dtype(np.int32),
                                               tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_precision = TensorArray.new_zeros(shape=(target_classes, ), dtype=np.dtype(np.int32),
                                                     tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:

        target_classes = self.model.output_shape[0]

        target_classes = np.int32(target_classes)

        self.precision.fill(0)
        self.local_precision.fill(0)

        self.kernel(self.precision.ary, self.conf_matrix_metric.conf_matrix.ary,
                    self.local_precision.ary, target_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)

        return float(self.precision[0])
